import json
import logging
import secrets

from openai import OpenAI

from agency._templates import get_environment

logger = logging.getLogger("agency")


class Agent:
    def __init__(self, tools, model, name, description=None, instructions=None, client=None, tags=None):
        self.id = secrets.token_hex(3)
        self.tools = tools
        self.model = model
        self.name = name
        self.description = description
        self.instructions = instructions
        self.client = client or OpenAI()
        self.tags = tags or []
        self.messages = []

    def with_tools(self, tools):
        agent = Agent(self.tools + tools, self.model, self.name, self.description, self.instructions, self.client, self.tags)
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
        tool_map = {function.schema["function"]["name"]: function for function in self.tools}
        schemas = [function.schema for function in self.tools] or None
        on_tool_call = on_tool_call or (lambda fn, **kw: fn(**kw))

        system_prompt = {
            "role": "system",
            "content": get_environment().get_template("agent_system.jinja").render(agent=self),
        }

        while True:
            if on_before_iteration:
                on_before_iteration()

            response = self.client.chat.completions.create(
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
                logger.info("[%s] %s(%s)", self.name, tool_call.function.name, arguments)
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
