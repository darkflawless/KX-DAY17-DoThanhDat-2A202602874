from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A: Baseline Agent.

    Requirements:
    - Within-session memory only (keyed strictly by `thread_id`)
    - No persistent `User.md`
    - Forgets long-term facts across new threads
    """

    def __init__(
        self, config: LabConfig | None = None, force_offline: bool = False
    ) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None

        if not self.force_offline and self.config.model.api_key:
            try:
                self.langchain_agent = self._maybe_build_langchain_agent()
            except Exception:
                self.langchain_agent = None

    def reply(
        self, user_id: str, thread_id: str, message: str
    ) -> dict[str, Any]:
        """Return agent response and token accounting."""
        if self.langchain_agent and not self.force_offline:
            try:
                # Live path if model and agent are available
                res = self.langchain_agent.invoke(
                    {"input": message},
                    config={"configurable": {"thread_id": thread_id}},
                )
                output = res.get("output", "")
                tokens = estimate_tokens(output)
                session = self.sessions.setdefault(thread_id, SessionState())
                session.messages.append({"role": "user", "content": message})
                session.messages.append({"role": "assistant", "content": output})
                prompt_tok = sum(
                    estimate_tokens(m["content"]) for m in session.messages
                )
                session.token_usage += tokens
                session.prompt_tokens_processed += prompt_tok
                return {
                    "reply": output,
                    "tokens": tokens,
                    "prompt_tokens": prompt_tok,
                }
            except Exception:
                pass

        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str | None = None) -> int:
        if thread_id is not None:
            return self.sessions.get(thread_id, SessionState()).token_usage
        return sum(s.token_usage for s in self.sessions.values())

    def prompt_token_usage(self, thread_id: str | None = None) -> int:
        if thread_id is not None:
            return self.sessions.get(
                thread_id, SessionState()
            ).prompt_tokens_processed
        return sum(s.prompt_tokens_processed for s in self.sessions.values())

    def compaction_count(self, thread_id: str | None = None) -> int:
        # Baseline agent has no compact memory
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Simple offline behavior for Baseline Agent."""
        session = self.sessions.setdefault(thread_id, SessionState())

        # Baseline accumulates ALL messages in thread to form prompt
        prompt_tokens = sum(
            estimate_tokens(m["content"]) for m in session.messages
        ) + estimate_tokens(message)
        session.prompt_tokens_processed += prompt_tokens
        session.messages.append({"role": "user", "content": message})

        # Baseline only has context from current thread
        reply_text = (
            "Tôi là Baseline Agent. Trong phiên này tôi đã ghi nhận tin nhắn của bạn."
        )

        resp_tokens = estimate_tokens(reply_text)
        session.token_usage += resp_tokens
        session.messages.append({"role": "assistant", "content": reply_text})

        return {
            "reply": reply_text,
            "tokens": resp_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _maybe_build_langchain_agent(self):
        """Optional wire-up for live agent."""
        try:
            model = build_chat_model(self.config.model)
            return model
        except Exception:
            return None
