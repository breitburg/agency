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

    def find_channel_seats(self, channel, exclude=None):
        return [
            seat for seat in self.seats
            if channel in seat.agent.tags and seat.agent is not exclude
        ]

    def create_toolkit(self, agent):
        agent_roster = "\n".join(
            f"* {seat.agent.name} ({seat.agent.id}) - {seat.agent.description}"
            for seat in self.seats
            if seat.agent is not agent
        )

        all_tags = {tag for seat in self.seats for tag in seat.agent.tags}
        channel_roster = "\n".join(
            f"* #{tag} - {len(members)} agent(s): {', '.join(s.agent.name for s in members)}"
            for tag in sorted(all_tags)
            if (members := self.find_channel_seats(tag, exclude=agent))
        )

        description = f"Send a message to an agent or channel. Available agents:\n{agent_roster}"
        if channel_roster:
            description += f"\n\nAvailable channels:\n{channel_roster}"

        @tool(
            name="SendMessage",
            description=description,
        )
        def send_message(recipient: str, body: str):
            """Send a message to an agent or channel.

            Args:
                recipient: The 6-character ID of the target agent, or a #channel name.
                body: The message body to send.
            """
            if recipient.startswith("#"):
                channel = recipient[1:]
                seats = self.find_channel_seats(channel, exclude=agent)
                if not seats:
                    return f"Channel #{channel} not found or has no other members"
                for seat in seats:
                    seat.inbox.append((agent, body))
                    if seat.thread is None:
                        self.run(seat.agent)
                names = ", ".join(s.agent.name for s in seats)
                return f"Message sent to #{channel} ({len(seats)} agent(s): {names})"

            seat = self.find_seat(recipient)
            if not seat:
                return f"Agent {recipient} not found"
            seat.inbox.append((agent, body))
            if seat.thread is None:
                self.run(seat.agent)
            return f"Message sent to {seat.agent.name} ({seat.agent.id})"

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
