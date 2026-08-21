from __future__ import annotations

import pytest

from enroute.tracing.schema import Outcome, ParsedAction, Trace


def _episode(*actions: str, reward: float | None = 1.0) -> Trace:
    trace = Trace(outcome=Outcome(reward=reward) if reward is not None else None)
    for name in actions:
        trace.add_decision(parsed_action=[ParsedAction(name=name)])
    return trace


def test_sparse_outcome_returns() -> None:
    trace = _episode("search", "read", "answer")
    assert trace.decision_rewards(source="outcome") == [0.0, 0.0, 1.0]
    assert trace.returns(gamma=1.0) == [1.0, 1.0, 1.0]
    assert trace.returns(gamma=0.9) == pytest.approx([0.81, 0.9, 1.0])


def test_credit_decision_then_events_returns() -> None:
    trace = _episode("search", "tweet", reward=0.0)
    event = trace.credit(0.8, name="likes", reason="overnight", decision_index=1)
    assert event.name == "likes"
    assert event.value == 0.8
    assert trace.decisions()[1].reward_events[-1].name == "likes"
    assert trace.decision_rewards(source="events") == [0.0, 0.8]
    assert trace.returns(gamma=0.5, source="events") == pytest.approx([0.4, 0.8])


def test_credit_episode_increments_outcome() -> None:
    trace = _episode("answer", reward=1.0)
    trace.credit(0.3, name="reviewer")
    assert trace.outcome is not None
    assert trace.outcome.reward == pytest.approx(1.3)
    assert trace.outcome.scores["reviewer"] == pytest.approx(0.3)


def test_credit_negative_index_and_both() -> None:
    trace = _episode("search", "answer", reward=1.0)
    trace.credit(0.4, name="reviewer", decision_index=-1)
    assert trace.decision_rewards(source="both") == pytest.approx([0.0, 1.4])
    assert trace.returns(gamma=1.0, source="both") == pytest.approx([1.4, 1.4])


def test_credit_out_of_range() -> None:
    trace = Trace()
    with pytest.raises(IndexError):
        trace.credit(1.0, decision_index=0)
    trace.add_decision(parsed_action=[ParsedAction(name="a")])
    with pytest.raises(IndexError):
        trace.credit(1.0, decision_index=3)
