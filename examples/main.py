import logging
import subprocess
from time import sleep

from openai import OpenAI

from agency import Agent, Agency, tool

logging.basicConfig(level=logging.CRITICAL, format="%(asctime)s %(name)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)


@tool(name="Bash")
def bash(command: str):
    """Execute a bash command and return the output.

    Args:
        command: The bash command to execute.
    """
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout + result.stderr


def main():
    alice = Agent([], model="kimi-k2.5:cloud", name="Alice", client=client)
    bob = Agent(
        [bash],
        model="kimi-k2.5:cloud",
        name="Bob",
        description="Has access to the computer.",
        client=client,
    )
    hannah = Agent(
        [],
        model="kimi-k2.5:cloud",
        name="Hannah",
        client=client,
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
