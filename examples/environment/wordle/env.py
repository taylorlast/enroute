"""Wordle: the environment *is* the game. The LLM only calls guess."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from enroute import Environment, TaskData
from enroute.environments import Observation, State, tool

try:
    from .words import is_allowed, pattern, pick_answer
except ImportError:
    from words import is_allowed, pattern, pick_answer

MAX_GUESSES = 6
_MARK_RANK = {".": 0, "Y": 1, "G": 2}

INSTRUCTIONS = (
    "You are playing Wordle. The only action is the guess tool. "
    "Submit a valid 5-letter English word each turn. Read the board: "
    "G is correct place, Y is wrong place, . is absent. "
    "You have six valid guesses. Do not reveal that you are an LLM."
)


class WordleRow(BaseModel):
    """One submitted guess and its mark pattern."""

    guess: str = ""
    pattern: str = ""


class WordleState(State):
    """Hidden puzzle plus the board. ``secret`` is not in the observation."""

    secret: str = ""
    rows: list[WordleRow] = []
    solved: bool = False
    seen_marks: dict[str, str] = {}
    last_new_greens: int = 0
    last_new_yellows: int = 0
    last_invalid: bool = False


class WordleObservation(Observation):
    """Visible board, keyboard, and guesses left."""

    board: str = ""
    guesses_left: int = MAX_GUESSES
    letters: dict[str, str] = {}

    def render(self) -> str:
        return self.board


class WordleEnv(Environment[WordleObservation, WordleState]):
    """A single Wordle puzzle. The policy only calls :meth:`guess`."""

    name = "wordle"
    version = "0.1.0"
    system_prompt = INSTRUCTIONS
    max_turns = 8

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.state = WordleState()
        self.scorer(self._solved_score, name="solved")
        self.tasks(self._default_tasks)

    def setup(self, task: TaskData) -> None:
        """Start a puzzle from ``task.expected`` or ``task.metadata["seed"]``.

        Args:
            task: Task data. Pin a word with ``expected``; otherwise pick by seed.
        """
        super().setup(task)
        pinned = str(getattr(task, "expected", "") or "").strip().lower()
        secret = pinned or pick_answer(int(self.seed or 0))
        self.state = WordleState(seed=self.seed, secret=secret)

    def observe(self) -> WordleObservation:
        """Build the board the policy may see. Called by reset/step."""
        lines = ["Wordle. Guess a valid 5-letter word.", "Board:"]
        if not self.state.rows:
            lines.append("  (empty)")
        for i, row in enumerate(self.state.rows, start=1):
            spaced = " ".join(row.pattern)
            lines.append(f"  {i}. {row.guess.upper()}  {spaced}")
        lines.append(f"Guesses left: {self.guesses_left}")
        letters = dict(self.state.seen_marks)
        if letters:
            keys = " ".join(f"{letter.upper()}:{mark}" for letter, mark in sorted(letters.items()))
            lines.append(f"Letters: {keys}")
        return WordleObservation(
            board="\n".join(lines),
            guesses_left=self.guesses_left,
            letters=letters,
        )

    def done(self) -> bool:
        """Return whether the puzzle is over."""
        return self.state.solved or len(self.state.rows) >= MAX_GUESSES

    def score(self) -> float:
        """Terminal return: ``0`` if unsolved, else fewer guesses score higher."""
        if not self.state.solved:
            return 0.0
        return (7 - len(self.state.rows)) / 6

    def step_reward(self, tool_name: str, result: Any) -> float | None:
        """Dense guide: new greens/yellows, small penalty for invalid guesses."""
        if tool_name != "guess":
            return None
        if self.state.last_invalid:
            return -0.05
        return 0.1 * self.state.last_new_greens + 0.03 * self.state.last_new_yellows

    @property
    def guesses_left(self) -> int:
        """Valid guesses remaining."""
        return max(0, MAX_GUESSES - len(self.state.rows))

    @property
    def secret(self) -> str:
        """Hidden answer (not in the observation)."""
        return self.state.secret

    @property
    def solved(self) -> bool:
        """Whether the current puzzle is solved."""
        return self.state.solved

    @property
    def rows(self) -> list[WordleRow]:
        """Submitted rows."""
        return self.state.rows

    @tool
    def guess(self, word: str) -> dict[str, Any]:
        """Submit a 5-letter guess. Invalid words do not consume a turn."""
        self.state.last_new_greens = 0
        self.state.last_new_yellows = 0
        self.state.last_invalid = False
        cleaned = word.strip().lower()
        if len(cleaned) != 5:
            self.state.last_invalid = True
            return {"error": "guess must be 5 letters", "guess": cleaned}
        if not is_allowed(cleaned):
            self.state.last_invalid = True
            return {"error": "not in the word list", "guess": cleaned}
        if self.done():
            return {"error": "puzzle is over", "guess": cleaned}

        marks = pattern(cleaned, self.state.secret)
        self.state.rows.append(WordleRow(guess=cleaned, pattern=marks))
        self.state.solved = marks == "GGGGG"
        self._record_new_marks(cleaned, marks)
        return {
            "guess": cleaned,
            "pattern": marks,
            "solved": self.state.solved,
            "guesses_left": self.guesses_left,
        }

    def _record_new_marks(self, word: str, marks: str) -> None:
        for letter, mark in zip(word, marks, strict=True):
            previous = self.state.seen_marks.get(letter, ".")
            if _MARK_RANK[mark] > _MARK_RANK[previous]:
                if mark == "G" and previous != "G":
                    self.state.last_new_greens += 1
                elif mark == "Y" and previous == ".":
                    self.state.last_new_yellows += 1
                self.state.seen_marks[letter] = mark
            elif letter not in self.state.seen_marks:
                self.state.seen_marks[letter] = mark

    def _solved_score(self, rollout: object) -> float:
        env = getattr(rollout, "env", None)
        return float(env.score()) if env is not None else 0.0

    def _default_tasks(self) -> list[TaskData]:
        return [
            TaskData(
                task_id="seed-1",
                input="Play Wordle. Use guess.",
                metadata={"seed": 1},
            ),
            TaskData(
                task_id="seed-2",
                input="Play Wordle. Use guess.",
                metadata={"seed": 2},
            ),
            TaskData(
                task_id="crane",
                input="Play Wordle. Use guess.",
                expected="crane",
                metadata={"seed": 0},
            ),
        ]


def make_env() -> WordleEnv:
    """Build a Wordle environment. The model is plugged in at rollout time."""
    return WordleEnv()
