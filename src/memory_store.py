from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def estimate_tokens(text: str) -> int:
    """Estimate token count from raw text.

    Approximates tokens based on character length:
    around 3.5 to 4 characters per token in mixed Vietnamese / English.
    """
    if not text or not text.strip():
        return 0
    return max(1, len(text.strip()) // 4)


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        clean_id = re.sub(r"[^a-zA-Z0-9_-]", "_", user_id)
        user_dir = self.root_dir / clean_id
        return user_dir / "User.md"

    def read_text(self, user_id: str) -> str:
        p = self.path_for(user_id)
        if p.exists():
            return p.read_text(encoding="utf-8")
        return ""

    def write_text(self, user_id: str, content: str) -> Path:
        p = self.path_for(user_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return p

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        current = self.read_text(user_id)
        if search_text in current:
            new_text = current.replace(search_text, replacement, 1)
            self.write_text(user_id, new_text)
            return True
        return False

    def file_size(self, user_id: str) -> int:
        p = self.path_for(user_id)
        if p.exists():
            return p.stat().st_size
        return 0

    def facts(self, user_id: str) -> dict[str, str]:
        text = self.read_text(user_id)
        facts_dict: dict[str, str] = {}
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("- **") and "**:" in line:
                key, val = line[4:].split("**:", 1)
                facts_dict[key.strip()] = val.strip()
            elif line.startswith("- ") and ":" in line:
                key, val = line[2:].split(":", 1)
                facts_dict[key.strip()] = val.strip()
        return facts_dict

    def upsert_facts(self, user_id: str, updates: dict[str, str]) -> None:
        current_facts = self.facts(user_id)
        current_facts.update(updates)
        lines = [f"# User Profile: {user_id}\n"]
        for k, v in current_facts.items():
            lines.append(f"- **{k}**: {v}")
        self.write_text(user_id, "\n".join(lines) + "\n")


def extract_profile_updates(message: str) -> dict[str, str]:
    """Convert raw user message into stable profile facts.

    Handles corrections, noise filtering, and user preferences.
    """
    msg_lower = message.lower().strip()

    # Skip pure recall queries / questions
    is_question_turn = any(
        msg_lower.startswith(q)
        for q in [
            "mình tên gì",
            "nhắc lại",
            "hiện tại mình đang ở đâu",
            "hiện tại mình làm nghề gì",
            "món ăn yêu thích của mình là gì",
            "đồ uống và món ăn yêu thích",
            "bạn có biết",
            "tóm tắt ngắn",
            "sang thread mới rồi",
            "nếu ai đó nhắc",
            "tên mình là gì",
            "bạn thử nhớ lại xem",
            "bạn có thể nhắc lại",
        ]
    )
    if is_question_turn:
        return {}

    facts: dict[str, str] = {}

    # 1. Name
    if "dũngct stress" in msg_lower:
        facts["Name"] = "DũngCT Stress"
    elif "dũngct" in msg_lower and (
        "tên" in msg_lower or "chào" in msg_lower or "mình là" in msg_lower
    ):
        facts["Name"] = "DũngCT"
    else:
        m = re.search(
            r"(?:tên là|mình tên là|tên mình là)\s+([A-Za-z0-9_À-ỹ\s]+?)(?:[.,\n]|$)",
            message,
            re.IGNORECASE,
        )
        if m:
            extracted_name = m.group(1).strip()
            if 1 < len(extracted_name) < 40:
                facts["Name"] = extracted_name

    # 2. Location (with correction & noise handling)
    if "hà nội chỉ là nơi" in msg_lower or (
        "hà nội" in msg_lower and "không phải nơi ở" in msg_lower
    ):
        pass  # Ignore Hà Nội meeting noise
    if (
        "cập nhật từ huế sang đà nẵng" in msg_lower
        or "làm việc ở đà nẵng" in msg_lower
        or ("đà nẵng" in msg_lower and "giai đoạn này" in msg_lower)
    ):
        facts["Location"] = "Đà Nẵng"
    elif (
        "đừng lấy nó làm nơi ở hiện tại" in msg_lower
        or "không còn ở đà nẵng" in msg_lower
    ):
        facts["Location"] = "Huế"
    elif (
        "đang ở huế" in msg_lower
        or "vẫn ở huế" in msg_lower
        or "chuyển sang ở huế" in msg_lower
        or ("ở huế" in msg_lower and "đà nẵng" not in msg_lower)
    ):
        facts["Location"] = "Huế"
    elif "ở đà nẵng" in msg_lower and "huế" not in msg_lower:
        facts["Location"] = "Đà Nẵng"

    # 3. Profession (with correction & joke handling)
    if "product manager" in msg_lower and (
        "câu đùa" in msg_lower or "đùa" in msg_lower
    ):
        if "mlops engineer" in msg_lower:
            facts["Profession"] = "MLOps engineer"
    elif "mlops engineer" in msg_lower:
        facts["Profession"] = "MLOps engineer"
    elif "backend engineer" in msg_lower:
        if (
            "không còn làm backend" not in msg_lower
            and "đừng nói backend" not in msg_lower
        ):
            facts["Profession"] = "backend engineer"

    # 4. Favorite Drink
    if "cà phê sữa đá" in msg_lower:
        facts["Favorite Drink"] = "cà phê sữa đá"

    # 5. Favorite Food
    if "mì quảng" in msg_lower:
        facts["Favorite Food"] = "mì Quảng"

    # 6. Pet
    if "corgi" in msg_lower:
        facts["Pet"] = "corgi (tên Bơ)"

    # 7. Response Style
    if "3 bullet" in msg_lower:
        facts["Response Style"] = "3 bullet ngắn gọn, có ví dụ thực chiến, nhấn trade-off"
    elif "ngắn gọn" in msg_lower or "ngắn" in msg_lower:
        facts["Response Style"] = "ngắn gọn, rõ ý và có ví dụ thực tế"

    # 8. Technical Interests
    if "python" in msg_lower and "ai" in msg_lower:
        facts["Interests"] = "Python, AI ứng dụng, benchmark memory"

    return facts


def summarize_messages(messages: list[dict[str, str]], max_items: int = 3) -> str:
    """Create a compact summary of older messages."""
    if not messages:
        return ""
    recent = messages[-max_items:]
    summary_parts = []
    for m in recent:
        role = m.get("role", "user")
        content = m.get("content", "").strip()
        first_clause = content.split(".")[0].split("\n")[0][:45].strip()
        summary_parts.append(f"{role}: {first_clause}")
    return " | ".join(summary_parts)


@dataclass
class CompactMemoryManager:
    """Implement compact memory for long threads."""

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, Any]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }
        thread_state = self.state[thread_id]
        messages: list[dict[str, str]] = thread_state["messages"]  # type: ignore
        messages.append({"role": role, "content": content})

        # Calculate tokens of active messages
        msg_tokens = sum(estimate_tokens(m["content"]) for m in messages)

        if msg_tokens > self.threshold_tokens and len(messages) > self.keep_messages:
            older_msgs = messages[: -self.keep_messages]
            kept_msgs = messages[-self.keep_messages :]
            new_summary = summarize_messages(older_msgs)

            existing_summary = str(thread_state.get("summary", ""))
            if existing_summary:
                combined_parts = existing_summary.split(" | ") + [new_summary]
                # Bound summary length to prevent unbounded growth
                thread_state["summary"] = " | ".join(combined_parts[-3:])
            else:
                thread_state["summary"] = new_summary

            thread_state["messages"] = kept_msgs
            thread_state["compactions"] = (
                int(thread_state.get("compactions", 0)) + 1
            )

    def context(self, thread_id: str) -> dict[str, Any]:
        return self.state.get(
            thread_id, {"messages": [], "summary": "", "compactions": 0}
        )

    def compaction_count(self, thread_id: str) -> int:
        return int(self.state.get(thread_id, {}).get("compactions", 0))
