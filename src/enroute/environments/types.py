"""Observation and State models for environments.

Subclass these on each environment. Reward and termination stay on
:class:`~enroute.environments.env.StepResult`, not on the observation.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Observation(BaseModel):
    """What the agent is allowed to see.

    Authors subclass this and implement :meth:`render` when the prompt
    should not be a raw dump of the fields. :meth:`~enroute.environments.env.Environment.reset`
    and :meth:`~enroute.environments.env.Environment.step` call
    :meth:`~enroute.environments.env.Environment.observe` internally —
    do not call ``observe`` to drive an agent.

    Attributes:
        text: Default rendered view. Structured subclasses may ignore this.
        metadata: Extra visible fields that do not need a typed attribute.
    """

    text: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    def render(self) -> str:
        """Return the string the policy sees in the conversation.

        Returns:
            ``text``, or a subclass-specific prompt string.
        """
        return self.text

    def __str__(self) -> str:
        """Return :meth:`render` so tests and logs can print an observation."""
        return self.render()


class State(BaseModel):
    """Internal episode state. May include hidden fields (secrets, labels).

    Attributes:
        seed: Optional RNG seed from the task.
        metadata: Extra internal fields that do not need a typed attribute.
    """

    seed: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
