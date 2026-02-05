# Agency

A minimal multi-agent orchestration framework built on the OpenAI chat completions API. Agents run in parallel threads, use tools defined as plain Python functions, and communicate with each other through message passing.

## Installation

```bash
uv add git+https://github.com/breitburg/agency
```

## Concepts

`@tool()` is a decorator that turns a Python function into a tool the model can call. It inspects the function's type hints and docstring to generate the JSON schema automatically.

`Agent` is a stateful wrapper around a model. It holds a message history and a list of tools, and runs an agentic loop: prompt the model, execute any requested tool calls, repeat until the model produces a final text response.

`Agency` is a coordinator for multiple agents. It gives every agent a `SendMessage` tool so they can talk to each other, runs each agent in its own thread, and delivers messages through per-agent inboxes.

## Usage

### Defining tools

Decorate a function with `@tool()`. Parameter types (`str`, `int`, `float`, `bool`) and descriptions (from the `Args:` section of the docstring) are picked up automatically.

```python
from agency import tool

@tool()
def get_weather(city: str):
    """Get the current weather for a city.

    Args:
        city: Name of the city.
    """
    return f"It's sunny in {city}."
```

The `name` and `description` keyword arguments can override the defaults:

```python
@tool(name="Search", description="Search the web.")
def search(query: str):
    ...
```

### Running a single agent

```python
from agency import Agent

agent = Agent(
    tools=[get_weather],
    model="gpt-5.2",
    name="Assistant",
)

agent.messages.append({"role": "user", "content": "What's the weather in Berlin?"})
response = agent.run()
print(response.content)
```

### Callbacks

`Agent.run()` accepts callbacks to observe or intercept the agentic loop. These can be used to hook the library up to a UI, display interactions in real time, or ask the user for tool-calling permissions before execution.

```python
agent.run(
    on_before_iteration=lambda: print("thinking..."),
    on_after_iteration=lambda: print("done with this step"),
    on_tool_call=lambda fn, **kw: fn(**kw),  # default behavior
    on_message=lambda msg: print(msg.content),
)
```

### Multi-agent with Agency

`Agency` wires agents together. Each agent automatically receives a `SendMessage` tool listing the other agents and their IDs. Sending a message to a sleeping agent wakes it up in a new thread.

```python
from agency import Agent, Agency, tool
import subprocess

@tool(name="Bash")
def bash(command: str):
    """Execute a bash command.

    Args:
        command: The bash command to execute.
    """
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout + result.stderr

researcher = Agent([], model="gpt-5.2", name="Researcher")
coder = Agent(
    [bash],
    model="gpt-5.2",
    name="Coder",
    description="Has access to bash.",
)

researcher.messages.append({"role": "user", "content": "Find out what OS we're on."})

with Agency(
    agents=[researcher, coder],
    on_agent_message=lambda agent, msg: print(f"{agent.name}: {msg.content}"),
) as agency:
    agency.run(researcher)
```

The researcher can ask the coder to run a command by calling `SendMessage`, and the coder will wake up, execute it, and reply back.

### Agency callbacks

Agency accepts the same callbacks as `Agent.run()`, prefixed with `on_agent_`, with the agent instance injected as the first argument:

| Callback | Signature | When |
|---|---|---|
| `on_agent_status_change` | `(seat, is_running)` | Agent thread starts or stops |
| `on_agent_before_iteration` | `(agent)` | Before each LLM request |
| `on_agent_after_iteration` | `(agent)` | After tool execution |
| `on_agent_tool_call` | `(agent, fn, **kwargs)` | On every tool call (return the result) |
| `on_agent_message` | `(agent, message)` | Agent produces a final response |
