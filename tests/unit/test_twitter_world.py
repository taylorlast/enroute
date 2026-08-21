from __future__ import annotations

import sys
from pathlib import Path

from enroute import TaskData

_TWITTER = Path(__file__).resolve().parents[2] / "examples" / "environment" / "twitter"
sys.path.insert(0, str(_TWITTER))
sys.modules.pop("env", None)
from env import TwitterEnv, make_env  # noqa: E402

sys.modules.pop("env", None)
sys.path.remove(str(_TWITTER))


def _env(seed: int = 42, goal: str = "reply_to_mentions") -> TwitterEnv:
    env = TwitterEnv()
    env.setup(
        TaskData(
            task_id="t",
            input="act",
            metadata={"seed": seed, "account": "agent", "goal": goal},
        )
    )
    return env


def test_reset_is_deterministic() -> None:
    a = _env(42)
    b = _env(42)
    assert a.snapshot()["mentions"] == b.snapshot()["mentions"]
    assert list(a.state.posts) == list(b.state.posts)


def test_tweet_like_follow_reply() -> None:
    env = _env()
    posted = env.tweet("hello from the sim")
    post_id = posted["post"]["post_id"]
    assert posted["post"]["author"] == "agent"

    liked = env.like(post_id)
    assert liked["likes"] == 1
    env.unlike(post_id)
    assert env.posts[post_id].likes == set()

    followed = env.follow("alice")
    assert followed["following"] == "alice"
    assert "alice" in env.users["agent"].following
    env.unfollow("alice")
    assert "alice" not in env.users["agent"].following

    mention = env._mentions()[0]
    reply = env.reply(mention.post_id, "on it")
    assert reply["post"]["reply_to"] == mention.post_id


def test_quote_repost_bookmark_media() -> None:
    env = _env()
    timeline = env.view_timeline()["posts"]
    target = timeline[0]["post_id"]
    env.repost(target)
    quoted = env.quote(target, "adding context")
    assert quoted["post"]["quote_of"] == target
    env.bookmark(target)
    image = env.create_image("sunset")
    video = env.create_video("walkthrough")
    tweeted = env.tweet("media", media_ids=[image["media_id"]])
    assert tweeted["post"]["media"][0]["type"] == "image"
    assert video["type"] == "video"


def test_goal_scores() -> None:
    env = _env(goal="reply_to_mentions")
    assert env.score("reply_to_mentions") < 1.0
    for mention in env._mentions():
        env.reply(mention.post_id, "ack")
    assert env.score("reply_to_mentions") == 1.0
    assert env.done() is True

    likes_env = _env(goal="get_n_likes")
    likes_env.task.metadata["target_likes"] = 2
    post = likes_env.tweet("please like")["post"]["post_id"]
    likes_env.posts[post].likes.add("alice")
    likes_env.posts[post].likes.add("bob")
    assert likes_env.score("get_n_likes") == 1.0


def test_make_env_registers_tools() -> None:
    env = make_env()
    assert env.name == "twitter-account"
    assert env.version == "0.1.0"
    names = {t.function.name for t in env.tool_defs}
    for required in {
        "view_timeline",
        "view_profile",
        "view_post",
        "view_notifications",
        "follow",
        "unfollow",
        "tweet",
        "reply",
        "like",
        "unlike",
        "repost",
        "quote",
        "bookmark",
        "create_image",
        "create_video",
    }:
        assert required in names
    tasks = list(env.iter_tasks())
    assert tasks[0].metadata["goal"] == "reply_to_mentions"
