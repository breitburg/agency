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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("agency")

templates = Environment(loader=FileSystemLoader(os.path.dirname(__file__)))

client = OpenAI(
    base_url="https://ollama.com/v1",
    api_key=os.environ.get("OLLAMA_API_KEY"),
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

    def run(self, **kwargs):
        tool_map = {
            function.schema["function"]["name"]: function for function in self.tools
        }
        schemas = [function.schema for function in self.tools] or None

        system_prompt = {
            "role": "system",
            "content": templates.get_template("agent_system.jinja").render(agent=self),
        }

        while True:
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
                result = function(**arguments)

                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result),
                    }
                )


class Agency:
    def __init__(self, agents):
        self.agents = {agent: None for agent in agents}

    def create_toolkit(self, agent):
        roster = "\n".join(
            f"* {a.name} ({a.id}) - {a.description}"
            for a in self.agents
            if a is not agent
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
            target = next((a for a in self.agents if a.id == agent_id), None)
            if not target:
                return f"Agent {agent_id} not found"
            target.messages.append(
                {
                    "role": "user",
                    "content": templates.get_template(
                        "message_notification.jinja"
                    ).render(agent=agent, body=body),
                }
            )
            if self.agents[target] is None:
                self.run(target)
            return f"Message sent to {agent_id}"

        return [send_message]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        for thread in self.agents.values():
            if thread is not None:
                thread.join()

    def run(self, agent, **kwargs):
        extended = agent.with_tools(self.create_toolkit(agent))

        def target(**kwargs):
            logger.info("[%s] Waking up", agent.name)
            extended.run(**kwargs)
            self.agents[agent] = None
            logger.info("[%s] Going to sleep", agent.name)

        thread = threading.Thread(
            target=target,
            kwargs=kwargs,
            daemon=True,
        )
        thread.start()
        self.agents[agent] = thread


@tool(name="Bash")
def bash(command: str):
    """Execute a bash command and return the output.

    Args:
        command: The bash command to execute.
    """
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout + result.stderr


def main():
    alice = Agent([], model="glm-4.7:cloud", name="Alice")
    bob = Agent([bash], model="glm-4.7:cloud", name="Bob", description="Has access to the computer.")

    with Agency(agents=[alice, bob]) as agency:
        while True:
            try:
                user_input = input("> ")
            except (EOFError, KeyboardInterrupt):
                break

            alice.messages.append({"role": "user", "content": user_input})
            agency.run(alice)
            agency.agents[alice].join()


if __name__ == "__main__":
    main()
