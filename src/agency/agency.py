import logging
import threading

from agency._templates import get_environment
from agency.tool import tool


logger = logging.getLogger("agency")


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
                            "content": get_environment()
                            .get_template("message_notification.jinja")
                            .render(agent=sender, body=body),
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
