"""Wordle environment: the env is the game, the LLM is the policy."""

from __future__ import annotations

from .env import WordleEnv, make_env

__all__ = ["WordleEnv", "make_env"]
