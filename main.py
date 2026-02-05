import inspect
import json
import logging
import os
import secrets
import subprocess
import threading
from typing import get_type_hints

from jinja2 import Environment, FileSystemLoader
from openai import OpenAI
from time import sleep

logging.basicConfig(level=logging.CRITICAL, format="%(asctime)s %(name)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("agency")

templates = Environment(loader=FileSystemLoader(os.path.dirname(__file__)))

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)


def _parse_parameter_descriptions(docstring):
    parameter_descriptions = {}
    in_args_section = False

    for line in docstring.split("\n"):
        stripped = line.strip()

        if stripped.lower() in ("args:", "arguments:", "parameters:"):
            in_args_section = True
            continue

        if not in_args_section:
            continue

        if not stripped or (not line[0].isspace() if line else True):
            break

        if ":" not in stripped:
            continue

        name, description = stripped.split(":", 1)
        parameter_descriptions[name.strip()] = description.strip()

    return parameter_descriptions


def tool(function=None, *, name=None, description=None):
    def decorator(function):
        type_map = {str: "string", int: "integer", float: "number", bool: "boolean"}

        tool_name = name or function.__name__
        hints = get_type_hints(function)
        signature = inspect.signature(function)
        docstring = inspect.getdoc(function) or ""

        tool_description = description or (
            docstring.strip().split("\n")[0] if docstring.strip() else function.__name__
        )
        parameter_descriptions = _parse_parameter_descriptions(docstring)

        properties = {}
        required = []

        for parameter_name, parameter in signature.parameters.items():
            property_schema = {
                "type": type_map.get(hints.get(parameter_name), "string")
            }

            if parameter_name in parameter_descriptions:
                property_schema["description"] = parameter_descriptions[parameter_name]

            properties[parameter_name] = property_schema

            if parameter.default is inspect.Parameter.empty:
                required.append(parameter_name)

        function.schema = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": tool_description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
        return function

    if function is not None:
        return decorator(function)

    return decorator


class Agent:
    def __init__(self, tools, model, name, description=None):
        self.id = secrets.token_hex(3)
        self.tools = tools
        self.model = model
        self.name = name
        self.description = description
        self.messages = []

    def with_tools(self, tools):
        agent = Agent(self.tools + tools, self.model, self.name, self.description)
        agent.messages = self.messages
        return agent

    def run(
        self,
        on_before_iteration=None,
        on_after_iteration=None,
        on_tool_call=None,
        on_message=None,
        **kwargs,
    ):
        tool_map = {
            function.schema["function"]["name"]: function for function in self.tools
        }
        schemas = [function.schema for function in self.tools] or None
        on_tool_call = on_tool_call or (lambda fn, **kw: fn(**kw))

        system_prompt = {
            "role": "system",
            "content": templates.get_template("agent_system.jinja").render(agent=self),
        }

        while True:
            if on_before_iteration:
                on_before_iteration()

            response = client.chat.completions.create(
                model=self.model,
                messages=[system_prompt, *self.messages],
                tools=schemas,
                **kwargs,
            )
            message = response.choices[0].message
            self.messages.append(message)

            if not message.tool_calls:
                logger.info("[%s] %s", self.name, message.content)
                if on_message:
                    on_message(message)
                return message

            for tool_call in message.tool_calls:
                function = tool_map.get(tool_call.function.name)

                if not function:
                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": f"Unknown tool: {tool_call.function.name}",
                        }
                    )
                    continue

                arguments = json.loads(tool_call.function.arguments)
                logger.info(
                    "[%s] %s(%s)", self.name, tool_call.function.name, arguments
                )
                result = on_tool_call(function, **arguments)

                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result),
                    }
                )

            if on_after_iteration:
                on_after_iteration()


class AgentSeat:
    def __init__(self, agent):
        self.agent = agent
        self.thread = None
        self.inbox = []


class Agency:
    def __init__(
        self,
        agents,
        on_agent_status_change=None,
        on_agent_before_iteration=None,
        on_agent_after_iteration=None,
        on_agent_tool_call=None,
        on_agent_message=None,
    ):
        self.seats = [AgentSeat(agent) for agent in agents]
        self.on_agent_status_change = on_agent_status_change
        self.on_agent_before_iteration = on_agent_before_iteration
        self.on_agent_after_iteration = on_agent_after_iteration
        self.on_agent_tool_call = on_agent_tool_call
        self.on_agent_message = on_agent_message

    def find_seat(self, agent_id):
        return next((seat for seat in self.seats if seat.agent.id == agent_id), None)

    def create_toolkit(self, agent):
        roster = "\n".join(
            f"* {seat.agent.name} ({seat.agent.id}) - {seat.agent.description}"
            for seat in self.seats
            if seat.agent is not agent
        )

        @tool(
            name="SendMessage",
            description=f"Send a message to another agent. Available agents:\n{roster}",
        )
        def send_message(agent_id: str, body: str):
            """Send a message to another agent.

            Args:
                agent_id: The 6-character ID of the target agent.
                body: The message body to send.
            """
            seat = self.find_seat(agent_id)
            if not seat:
                return f"Agent {agent_id} not found"
            seat.inbox.append((agent, body))
            if seat.thread is None:
                self.run(seat.agent)
            return f"Message sent to {agent_id}"

        return [send_message]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        for seat in self.seats:
            if seat.thread is not None:
                seat.thread.join()

    def run(self, agent, **kwargs):
        seat = self.find_seat(agent.id)
        extended = agent.with_tools(self.create_toolkit(agent))

        def with_agent(callback):
            if callback is None:
                return None
            return lambda *args, **kwargs: callback(agent, *args, **kwargs)

        def target(**kwargs):
            logger.info("[%s] Waking up", agent.name)

            def handle_before_iteration():
                for sender, body in seat.inbox:
                    extended.messages.append(
                        {
                            "role": "user",
                            "content": templates.get_template(
                                "message_notification.jinja"
                            ).render(agent=sender, body=body),
                        }
                    )
                seat.inbox.clear()
                if self.on_agent_before_iteration:
                    self.on_agent_before_iteration(agent)

            extended.run(
                on_before_iteration=handle_before_iteration,
                on_after_iteration=with_agent(self.on_agent_after_iteration),
                on_tool_call=with_agent(self.on_agent_tool_call),
                on_message=with_agent(self.on_agent_message),
                **kwargs,
            )
            seat.thread = None
            logger.info("[%s] Going to sleep", agent.name)
            if self.on_agent_status_change:
                self.on_agent_status_change(seat, False)

        seat.thread = threading.Thread(
            target=target,
            kwargs=kwargs,
            daemon=True,
        )
        seat.thread.start()
        if self.on_agent_status_change:
            self.on_agent_status_change(seat, True)


@tool(name="Bash")
def bash(command: str):
    """Execute a bash command and return the output.

    Args:
        command: The bash command to execute.
    """
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout + result.stderr


def main():
    alice = Agent([], model="kimi-k2.5:cloud", name="Alice")
    bob = Agent(
        [bash],
        model="kimi-k2.5:cloud",
        name="Bob",
        description="Has access to the computer.",
    )
    hannah = Agent(
        [],
        model="kimi-k2.5:cloud",
        name="Hannah",
    )
    
    while True:
        try:
            user_input = input("> ")
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        alice.messages.append({"role": "user", "content": user_input})

        def on_agent_message(agent, message):
            print(f"{agent.name}: {message.content.strip()}")

        def on_agent_tool_call(agent, fn, **kwargs):
            name = fn.schema["function"]["name"]
            if name == "SendMessage":
                receiver = next(
                    s.agent.name
                    for s in agency.seats
                    if s.agent.id == kwargs["agent_id"]
                )
                print(f"{agent.name} to {receiver}: '{kwargs['body']}'")
                return fn(**kwargs)

            args = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
            result = fn(**kwargs)
            first_line = str(result).split("\n", 1)[0]
            print(f"{name}({args}) -> {first_line}")
            return result

        with Agency(
            agents=[alice, bob, hannah],
            on_agent_message=on_agent_message,
            on_agent_tool_call=on_agent_tool_call,
        ) as agency:
            agency.run(alice)

        sleep(1)


if __name__ == "__main__":
    main()
