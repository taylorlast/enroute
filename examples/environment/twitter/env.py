"""In-memory Twitter / X simulator the agent acts on."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import ConfigDict, Field

from enroute import Environment, TaskData
from enroute.environments import Observation, State, tool

INSTRUCTIONS = (
    "You operate this Twitter account. Use tools to look around and act the "
    "way a person would: read the timeline and notifications, then follow, "
    "like, reply, post, quote, or bookmark. Prefer doing the goal over chatting."
)


@dataclass
class User:
    handle: str
    name: str
    bio: str
    following: set[str] = field(default_factory=set)
    followers: set[str] = field(default_factory=set)


@dataclass
class Post:
    post_id: str
    author: str
    text: str
    created: int
    likes: set[str] = field(default_factory=set)
    reposts: set[str] = field(default_factory=set)
    bookmarks: set[str] = field(default_factory=set)
    reply_to: str | None = None
    quote_of: str | None = None
    media: list[dict[str, str]] = field(default_factory=list)


class TwitterState(State):
    """Social graph and account the tools act on."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    account: str = "agent"
    tick: int = 0
    next_id: int = 1
    users: dict[str, User] = Field(default_factory=dict)
    posts: dict[str, Post] = Field(default_factory=dict)
    media: dict[str, dict[str, str]] = Field(default_factory=dict)
    seen_notifications: set[str] = Field(default_factory=set)
    goal: str | None = None


class TwitterObservation(Observation):
    """Account briefing the policy may see."""

    account: str = "agent"
    goal: str | None = None
    followers: int = 0
    following: int = 0
    mentions_waiting: int = 0
    briefing: str = ""

    def render(self) -> str:
        return self.briefing


class TwitterEnv(Environment[TwitterObservation, TwitterState]):
    """Deterministic graph of users, posts, and notifications.

    Seed via ``task.metadata["seed"]``. The agent account is
    ``task.metadata["account"]`` (default ``agent``).
    """

    name = "twitter-account"
    version = "0.1.0"
    system_prompt = INSTRUCTIONS
    max_turns = 12

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.state = TwitterState()
        self.scorer(self._goal_progress, name="goal_progress")
        self.tasks(self._default_tasks)

    @property
    def account(self) -> str:
        return self.state.account

    @account.setter
    def account(self, value: str) -> None:
        self.state.account = value

    @property
    def tick(self) -> int:
        return self.state.tick

    @tick.setter
    def tick(self, value: int) -> None:
        self.state.tick = value

    @property
    def users(self) -> dict[str, User]:
        return self.state.users

    @property
    def posts(self) -> dict[str, Post]:
        return self.state.posts

    @property
    def media(self) -> dict[str, dict[str, str]]:
        return self.state.media

    @property
    def _next_id(self) -> int:
        return self.state.next_id

    @_next_id.setter
    def _next_id(self, value: int) -> None:
        self.state.next_id = value

    @property
    def _seen_notifications(self) -> set[str]:
        return self.state.seen_notifications

    def setup(self, task: TaskData) -> None:
        """Build a small social graph from the task seed.

        Args:
            task: Task data with optional ``account``, ``seed``, and ``goal``.
        """
        super().setup(task)
        metadata = getattr(task, "metadata", None) or {}
        goal = metadata.get("goal")
        self.state = TwitterState(
            seed=self.seed,
            account=str(metadata.get("account") or "agent").lstrip("@"),
            goal=str(goal) if goal else None,
        )
        self._seed_graph()

    def observe(self) -> TwitterObservation:
        """Build the briefing the policy may see. Called by reset/step."""
        me = self.users[self.account]
        mentions = self._mentions()
        timeline = self._timeline_posts()[:5]
        goal = self._goal()
        lines = [
            f"You operate @{self.account}. {me.bio}",
            f"Goal: {goal or 'none (explore and act as a person would).'}",
            f"Followers: {len(me.followers)}  Following: {len(me.following)}",
            f"Mentions waiting: {len(mentions)}",
        ]
        if timeline:
            lines.append("Recent timeline:")
            for post in timeline:
                lines.append(f"- {post.post_id} @{post.author}: {post.text}")
        if mentions:
            lines.append("Mentions:")
            for post in mentions:
                lines.append(f"- {post.post_id} @{post.author}: {post.text}")
        briefing = "\n".join(lines)
        return TwitterObservation(
            account=self.account,
            goal=goal,
            followers=len(me.followers),
            following=len(me.following),
            mentions_waiting=len(mentions),
            briefing=briefing,
        )

    def done(self) -> bool:
        """Return whether the current goal is fully satisfied.

        Returns:
            ``True`` when every mention has a reply (``reply_to_mentions``).
        """
        if self._goal() != "reply_to_mentions":
            return False
        return self._mention_reply_rate() >= 1.0

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot.

        Returns:
            Account stats, post ids, and goal progress.
        """
        me = self.users.get(self.account)
        return {
            "seed": self.seed,
            "account": self.account,
            "goal": self._goal(),
            "tick": self.tick,
            "followers": len(me.followers) if me else 0,
            "following": len(me.following) if me else 0,
            "own_posts": [p.post_id for p in self.posts.values() if p.author == self.account],
            "likes_on_own_posts": self._likes_on_own_posts(),
            "mentions": [p.post_id for p in self._mentions()],
            "mention_reply_rate": self._mention_reply_rate(),
        }

    def step_reward(self, tool_name: str, result: Any) -> float | None:
        """Dense reward for useful actions.

        Args:
            tool_name: Tool that just ran.
            result: Tool payload.

        Returns:
            A small positive reward for successful writes, else ``None``.
        """
        if isinstance(result, dict) and result.get("error"):
            return -0.05
        if tool_name in {"tweet", "reply", "quote", "like", "repost", "follow", "bookmark"}:
            return 0.05
        return None

    def score(self, goal: str | None = None) -> float:
        """Terminal score for a named goal.

        Args:
            goal: Goal id. Defaults to the task goal.

        Returns:
            A float in ``[0, 1]``.
        """
        goal = goal or self._goal()
        if goal == "reply_to_mentions":
            return self._mention_reply_rate()
        if goal == "get_n_likes":
            target = 3
            metadata = getattr(self.task, "metadata", None) or {}
            raw = metadata.get("target_likes")
            if raw is not None:
                target = int(raw)
            return min(1.0, self._likes_on_own_posts() / max(target, 1))
        return 0.0

    @tool
    def view_timeline(self, limit: int = 10) -> dict[str, Any]:
        """Show the home timeline (posts from accounts you follow, newest first)."""
        posts = self._timeline_posts()[: max(1, min(limit, 50))]
        return {"posts": [self._post_view(p) for p in posts]}

    @tool
    def view_profile(self, handle: str) -> dict[str, Any]:
        """View a user profile and their recent posts."""
        user = self._user(handle)
        if user is None:
            return {"error": f"unknown user @{handle.lstrip('@')}"}
        posts = [p for p in self.posts.values() if p.author == user.handle]
        posts.sort(key=lambda p: p.created, reverse=True)
        return {
            "handle": user.handle,
            "name": user.name,
            "bio": user.bio,
            "followers": len(user.followers),
            "following": len(user.following),
            "you_follow": user.handle in self.users[self.account].following,
            "follows_you": self.account in user.following,
            "recent_posts": [self._post_view(p) for p in posts[:5]],
        }

    @tool
    def view_post(self, post_id: str) -> dict[str, Any]:
        """View a single post and its replies."""
        post = self.posts.get(post_id)
        if post is None:
            return {"error": f"unknown post {post_id}"}
        replies = [p for p in self.posts.values() if p.reply_to == post_id]
        return {**self._post_view(post), "replies": [self._post_view(p) for p in replies]}

    @tool
    def view_notifications(self) -> dict[str, Any]:
        """View mentions and replies directed at this account."""
        mentions = self._mentions()
        for post in mentions:
            self._seen_notifications.add(post.post_id)
        return {"mentions": [self._post_view(p) for p in mentions]}

    @tool
    def follow(self, handle: str) -> dict[str, Any]:
        """Follow a user."""
        user = self._user(handle)
        if user is None:
            return {"error": f"unknown user @{handle.lstrip('@')}"}
        if user.handle == self.account:
            return {"error": "cannot follow yourself"}
        self.users[self.account].following.add(user.handle)
        user.followers.add(self.account)
        self.tick += 1
        return {"ok": True, "following": user.handle}

    @tool
    def unfollow(self, handle: str) -> dict[str, Any]:
        """Unfollow a user."""
        user = self._user(handle)
        if user is None:
            return {"error": f"unknown user @{handle.lstrip('@')}"}
        self.users[self.account].following.discard(user.handle)
        user.followers.discard(self.account)
        self.tick += 1
        return {"ok": True, "unfollowed": user.handle}

    @tool
    def tweet(self, text: str, media_ids: list[str] | None = None) -> dict[str, Any]:
        """Publish a tweet from this account."""
        post = self._create_post(self.account, text, media_ids=media_ids)
        return {"ok": True, "post": self._post_view(post)}

    @tool
    def reply(self, post_id: str, text: str) -> dict[str, Any]:
        """Reply to a post."""
        if post_id not in self.posts:
            return {"error": f"unknown post {post_id}"}
        post = self._create_post(self.account, text, reply_to=post_id)
        return {"ok": True, "post": self._post_view(post)}

    @tool
    def like(self, post_id: str) -> dict[str, Any]:
        """Like a post."""
        post = self.posts.get(post_id)
        if post is None:
            return {"error": f"unknown post {post_id}"}
        post.likes.add(self.account)
        self.tick += 1
        return {"ok": True, "likes": len(post.likes)}

    @tool
    def unlike(self, post_id: str) -> dict[str, Any]:
        """Remove a like from a post."""
        post = self.posts.get(post_id)
        if post is None:
            return {"error": f"unknown post {post_id}"}
        post.likes.discard(self.account)
        self.tick += 1
        return {"ok": True, "likes": len(post.likes)}

    @tool
    def repost(self, post_id: str) -> dict[str, Any]:
        """Repost / retweet a post."""
        post = self.posts.get(post_id)
        if post is None:
            return {"error": f"unknown post {post_id}"}
        post.reposts.add(self.account)
        self.tick += 1
        return {"ok": True, "reposts": len(post.reposts)}

    @tool
    def quote(self, post_id: str, text: str) -> dict[str, Any]:
        """Quote-tweet a post with your own commentary."""
        if post_id not in self.posts:
            return {"error": f"unknown post {post_id}"}
        post = self._create_post(self.account, text, quote_of=post_id)
        self.posts[post_id].reposts.add(self.account)
        return {"ok": True, "post": self._post_view(post)}

    @tool
    def bookmark(self, post_id: str) -> dict[str, Any]:
        """Bookmark a post for later."""
        post = self.posts.get(post_id)
        if post is None:
            return {"error": f"unknown post {post_id}"}
        post.bookmarks.add(self.account)
        self.tick += 1
        return {"ok": True, "bookmarked": post_id}

    @tool
    def create_image(self, prompt: str) -> dict[str, Any]:
        """Create an image attachment (placeholder id in this simulator)."""
        media_id = self._new_id("img")
        self.media[media_id] = {
            "media_id": media_id,
            "type": "image",
            "prompt": prompt,
            "url": f"sim://image/{media_id}",
        }
        self.tick += 1
        return self.media[media_id]

    @tool
    def create_video(self, prompt: str) -> dict[str, Any]:
        """Create a video attachment (placeholder id in this simulator)."""
        media_id = self._new_id("vid")
        self.media[media_id] = {
            "media_id": media_id,
            "type": "video",
            "prompt": prompt,
            "url": f"sim://video/{media_id}",
        }
        self.tick += 1
        return self.media[media_id]

    def _goal_progress(self, rollout: object) -> float:
        env = getattr(rollout, "env", None)
        task = getattr(rollout, "task", None)
        if env is None or task is None:
            return 0.0
        return float(env.score(task.metadata.get("goal")))

    def _default_tasks(self) -> list[TaskData]:
        return [
            TaskData(
                task_id="reply-mentions-1",
                input="Reply to every mention.",
                metadata={"seed": 42, "account": "agent", "goal": "reply_to_mentions"},
            ),
            TaskData(
                task_id="get-likes-1",
                input="Post something people will like. Target 3 likes.",
                metadata={
                    "seed": 42,
                    "account": "agent",
                    "goal": "get_n_likes",
                    "target_likes": 3,
                },
            ),
        ]

    def _seed_graph(self) -> None:
        people = [
            (self.account, self.account.title(), "Operator of this account."),
            ("alice", "Alice", "Writes about coffee and cities."),
            ("bob", "Bob", "Builds things. Tweets too much."),
            ("cara", "Cara", "Asks good questions."),
            ("news", "Daily News", "Headlines, no hot takes."),
        ]
        for handle, name, bio in people:
            self.users[handle] = User(handle=handle, name=name, bio=bio)

        self._follow("alice", "news")
        self._follow("alice", "bob")
        self._follow("bob", "alice")
        self._follow("cara", "news")
        self._follow("cara", self.account)
        self._follow(self.account, "news")

        news_one = self._create_post("news", "Markets opened mixed. Weather stays mild.")
        alice_one = self._create_post("alice", "Best cortado in town is still at Grove.")
        self._create_post(
            "bob", "Shipping on a Saturday. Don't tell my future self.", reply_to=alice_one.post_id
        )
        self._create_post("cara", f"hey @{self.account} what should I read this week?")
        self._create_post("news", "City council votes tonight on the river path.")
        self._create_post("alice", f"@{self.account} you coming to the pop-up tomorrow?")
        news_one.likes.add("alice")
        news_one.likes.add("bob")
        alice_one.likes.add("bob")

    def _follow(self, src: str, dest: str) -> None:
        self.users[src].following.add(dest)
        self.users[dest].followers.add(src)

    def _create_post(
        self,
        author: str,
        text: str,
        *,
        reply_to: str | None = None,
        quote_of: str | None = None,
        media_ids: list[str] | None = None,
    ) -> Post:
        post_id = self._new_id("p")
        media = [self.media[mid] for mid in (media_ids or []) if mid in self.media]
        post = Post(
            post_id=post_id,
            author=author,
            text=text,
            created=self.tick,
            reply_to=reply_to,
            quote_of=quote_of,
            media=media,
        )
        self.posts[post_id] = post
        self.tick += 1
        return post

    def _timeline_posts(self) -> list[Post]:
        following = self.users[self.account].following
        posts = [
            p for p in self.posts.values() if p.author in following or p.author == self.account
        ]
        posts.sort(key=lambda p: p.created, reverse=True)
        return posts

    def _mentions(self) -> list[Post]:
        token = f"@{self.account}"
        found: list[Post] = []
        for post in self.posts.values():
            if post.author == self.account:
                continue
            mentioned = token in post.text
            parent = self.posts.get(post.reply_to) if post.reply_to else None
            replied_to_me = parent is not None and parent.author == self.account
            if mentioned or replied_to_me:
                found.append(post)
        found.sort(key=lambda p: p.created, reverse=True)
        return found

    def _mention_reply_rate(self) -> float:
        mentions = self._mentions()
        if not mentions:
            return 1.0
        replied = 0
        for mention in mentions:
            if any(
                p.author == self.account and p.reply_to == mention.post_id
                for p in self.posts.values()
            ):
                replied += 1
        return replied / len(mentions)

    def _likes_on_own_posts(self) -> int:
        return sum(len(p.likes) for p in self.posts.values() if p.author == self.account)

    def _post_view(self, post: Post) -> dict[str, Any]:
        return {
            "post_id": post.post_id,
            "author": post.author,
            "text": post.text,
            "likes": len(post.likes),
            "reposts": len(post.reposts),
            "bookmarks": len(post.bookmarks),
            "reply_to": post.reply_to,
            "quote_of": post.quote_of,
            "media": post.media,
            "you_liked": self.account in post.likes,
        }

    def _user(self, handle: str) -> User | None:
        return self.users.get(handle.lstrip("@"))

    def _goal(self) -> str | None:
        if self.state.goal:
            return self.state.goal
        metadata = getattr(self.task, "metadata", None) or {}
        goal = metadata.get("goal")
        return str(goal) if goal else None

    def _new_id(self, prefix: str) -> str:
        value = f"{prefix}{self._next_id}"
        self._next_id += 1
        return value


def make_env() -> TwitterEnv:
    """Build twitter-account@0.1.0 with reply-to-mentions and likes goals."""
    return TwitterEnv()
