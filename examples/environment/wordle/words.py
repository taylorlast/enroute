"""Load official Wordle lists and score guesses.

Reads ``data/answers.txt`` (secrets) and ``data/allowed.txt`` (legal guesses).
Does not invent or fetch word lists.
"""

from __future__ import annotations

from pathlib import Path

_DATA = Path(__file__).resolve().parent / "data"


def _load_words(path: Path) -> tuple[str, ...]:
    words: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        word = raw.strip().lower()
        if word:
            words.append(word)
    return tuple(words)


ANSWERS: tuple[str, ...] = _load_words(_DATA / "answers.txt")
ALLOWED: frozenset[str] = frozenset(_load_words(_DATA / "allowed.txt")) | frozenset(ANSWERS)


def pick_answer(seed: int) -> str:
    """Pick a secret from ``ANSWERS`` using ``seed``.

    Args:
        seed: Index into the answer list (wrapped).

    Returns:
        A 5-letter answer word.
    """
    if not ANSWERS:
        raise RuntimeError("answers.txt is empty")
    return ANSWERS[int(seed) % len(ANSWERS)]


def is_allowed(word: str) -> bool:
    """Return whether ``word`` is a legal guess.

    Args:
        word: Candidate guess.

    Returns:
        ``True`` if the word is in the allowed set (including all answers).
    """
    return word.lower() in ALLOWED


def pattern(guess: str, secret: str) -> str:
    """Score ``guess`` against ``secret`` with official duplicate-letter rules.

    Greens are assigned first. Remaining letters in the secret can mark
    yellows; leftovers are gray. Marks: ``G``, ``Y``, ``.``

    Args:
        guess: The player's 5-letter guess.
        secret: The hidden answer.

    Returns:
        A 5-character pattern string.

    Examples:
        >>> pattern("crane", "crane")
        'GGGGG'
        >>> pattern("abide", "speed")
        '...YY'
    """
    guess = guess.lower()
    secret = secret.lower()
    marks = ["."] * 5
    remaining: list[str] = list(secret)
    for i, letter in enumerate(guess):
        if letter == secret[i]:
            marks[i] = "G"
            remaining[i] = ""
    for i, letter in enumerate(guess):
        if marks[i] == "G":
            continue
        if letter in remaining:
            marks[i] = "Y"
            remaining[remaining.index(letter)] = ""
    return "".join(marks)
