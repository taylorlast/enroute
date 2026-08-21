"""Closed-corpus library: search, read, then answer.

Research actions have no immediate reward. The episode return is whether the
submitted answer matches the hidden fact. A trainer uses discounted returns
to credit search/read — the same pattern as researching, then posting, then
getting likes later.

``research`` is a hierarchical tool: it calls ``search`` then ``read``. That
is still one Decision; inner calls are recorded as children for a future
skill trainer.
"""

from __future__ import annotations

from typing import Any

from enroute import Environment, TaskData
from enroute.environments import Observation, State, tool

CORPUS: list[dict[str, str]] = [
    {
        "doc_id": "d1",
        "title": "City council notes",
        "body": "The city council is voting tonight on the river path.",
    },
    {
        "doc_id": "d2",
        "title": "Cafe review",
        "body": "The best cortado in town is still at Grove.",
    },
    {
        "doc_id": "d3",
        "title": "Shipping log",
        "body": "Bob ships on Saturday. Do not tell his future self.",
    },
]

INSTRUCTIONS = (
    "You are a librarian. Search the catalog, read a document if you need "
    "the fact, then submit an answer. You may call research to search and "
    "read in one step. Do not guess."
)


class LibraryState(State):
    """Question, hidden expected phrase, corpus, and submitted answer."""

    question: str = ""
    expected: str = ""
    submitted: str | None = None
    docs: list[dict[str, str]] = []


class LibraryObservation(Observation):
    """Question and whether an answer has been submitted."""

    question: str = ""
    status: str = "not submitted"

    def render(self) -> str:
        return (
            f"Question: {self.question}\n"
            f"Use search and read before answering. Status: {self.status}."
        )


class LibraryEnv(Environment[LibraryObservation, LibraryState]):
    """Tiny document store the agent can search and read."""

    name = "library"
    version = "0.1.0"
    system_prompt = INSTRUCTIONS
    max_turns = 8

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.state = LibraryState()
        self.scorer(self._correct, name="correct")
        self.tasks(self._default_tasks)

    def setup(self, task: TaskData) -> None:
        """Load the question and a fixed corpus.

        Args:
            task: Task with ``input`` (question) and ``expected`` (key phrase).
        """
        super().setup(task)
        self.state = LibraryState(
            seed=self.seed,
            question=str(getattr(task, "input", "") or ""),
            expected=str(getattr(task, "expected", "") or "").lower(),
            submitted=None,
            docs=list(CORPUS),
        )

    def observe(self) -> LibraryObservation:
        """Build the briefing the policy may see. Called by reset/step."""
        submitted = self.state.submitted
        status = "not submitted" if submitted is None else f"submitted: {submitted}"
        return LibraryObservation(question=self.state.question, status=status)

    def done(self) -> bool:
        """Return whether an answer has been submitted."""
        return self.state.submitted is not None

    def snapshot(self) -> dict[str, Any]:
        """Return question, answer, and whether it matches."""
        return {
            "question": self.state.question,
            "expected": self.state.expected,
            "answer": self.state.submitted,
            "correct": self.score(),
        }

    def score(self) -> float:
        """Terminal correctness in ``{0, 1}``."""
        if not self.state.submitted or not self.state.expected:
            return 0.0
        return 1.0 if self.state.expected in self.state.submitted.lower() else 0.0

    def step_reward(self, tool_name: str, result: Any) -> float | None:
        """No dense reward — research is credited by the trainer."""
        return None

    @tool
    def search(self, query: str) -> dict[str, Any]:
        """Search the library. Returns matching titles, not full text."""
        q = query.lower()
        hits = [
            {"doc_id": doc["doc_id"], "title": doc["title"]}
            for doc in self.state.docs
            if q in doc["title"].lower() or q in doc["body"].lower()
        ]
        return {"query": query, "hits": hits}

    @tool
    def read(self, doc_id: str) -> dict[str, Any]:
        """Read the full text of one document."""
        for doc in self.state.docs:
            if doc["doc_id"] == doc_id:
                return dict(doc)
        return {"error": f"unknown document {doc_id}"}

    @tool
    def research(self, query: str) -> dict[str, Any]:
        """Search, then read the first hit. Nested search/read are recorded."""
        found = self.search(query)
        hits = found.get("hits") or []
        if not hits:
            return {"query": query, "hits": [], "doc": None}
        doc = self.read(str(hits[0]["doc_id"]))
        return {"query": query, "hits": hits, "doc": doc}

    @tool
    def answer(self, text: str) -> dict[str, Any]:
        """Submit the final answer and end the episode."""
        self.state.submitted = text
        return {"ok": True, "answer": text}

    def _correct(self, rollout: object) -> float:
        env = getattr(rollout, "env", None)
        return float(env.score()) if env is not None else 0.0

    def _default_tasks(self) -> list[TaskData]:
        return [
            TaskData(
                task_id="river-path",
                input="What is the city voting on tonight?",
                expected="river path",
                metadata={"seed": 1},
            ),
            TaskData(
                task_id="cortado",
                input="Where is the best cortado?",
                expected="grove",
                metadata={"seed": 1},
            ),
        ]


def make_env() -> LibraryEnv:
    """Build the research-then-answer environment."""
    return LibraryEnv()
