from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore


def make_config(tmp_path: Path) -> LabConfig:
    """Build an isolated config for tests."""
    cfg = load_config()
    cfg.base_dir = tmp_path
    cfg.state_dir = tmp_path / "state"
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    (cfg.state_dir / "profiles").mkdir(parents=True, exist_ok=True)
    cfg.compact_threshold_tokens = 50
    cfg.compact_keep_messages = 2
    return cfg


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify `User.md` can be created, updated, and edited."""
    profiles_dir = tmp_path / "profiles"
    store = UserProfileStore(profiles_dir)

    # 1. Write initial profile
    written_path = store.write_text(
        "user123", "# User Profile\n\n- **Name**: Alice\n- **City**: Đà Nẵng\n"
    )
    assert written_path.exists()
    assert store.file_size("user123") > 0

    # 2. Read text
    content = store.read_text("user123")
    assert "Alice" in content
    assert "Đà Nẵng" in content

    # 3. Edit text (simulate correction)
    changed = store.edit_text("user123", "Đà Nẵng", "Huế")
    assert changed is True
    updated_content = store.read_text("user123")
    assert "Huế" in updated_content
    assert "Đà Nẵng" not in updated_content


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify long threads trigger compaction."""
    manager = CompactMemoryManager(threshold_tokens=50, keep_messages=2)
    thread_id = "test-thread-compact"

    # Append messages until compaction triggers
    for i in range(6):
        manager.append(
            thread_id,
            "user",
            f"Lượt {i}: Đoạn nội dung rất dài kiểm tra ngưỡng compaction kích hoạt bộ nhớ.",
        )

    assert manager.compaction_count(thread_id) > 0
    ctx = manager.context(thread_id)
    assert len(ctx["messages"]) <= 2
    assert len(ctx["summary"]) > 0


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify advanced remembers across sessions and baseline does not."""
    cfg = make_config(tmp_path)
    baseline = BaselineAgent(cfg, force_offline=True)
    advanced = AdvancedAgent(cfg, force_offline=True)

    # Session 1: Provide personal facts
    intro = "Chào bạn, mình tên là DũngCT, hiện ở Huế và uống cà phê sữa đá."
    baseline.reply("user_test", "session-1", intro)
    advanced.reply("user_test", "session-1", intro)

    # Session 2: Ask recall question in a completely fresh thread
    question = "Mình tên gì, ở đâu và thích uống gì?"
    base_reply = baseline.reply("user_test", "session-2-fresh", question)["reply"]
    adv_reply = advanced.reply("user_test", "session-2-fresh", question)["reply"]

    # Baseline forgets across new threads
    assert "DũngCT" not in base_reply
    assert "Huế" not in base_reply

    # Advanced recalls via persistent User.md
    assert "DũngCT" in adv_reply
    assert "Huế" in adv_reply
    assert "cà phê sữa đá" in adv_reply


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compare prompt load of baseline vs advanced on a long thread."""
    cfg = make_config(tmp_path)
    baseline = BaselineAgent(cfg, force_offline=True)
    advanced = AdvancedAgent(cfg, force_offline=True)

    long_thread_id = "stress-thread-load"
    long_turns = [
        f"Lượt số {i}: Đoạn nội dung rất dài nhằm mục đích đẩy số lượng token của hội thoại lên cao "
        f"và kích hoạt cơ chế nén bộ nhớ compaction sau một số lượt nhất định."
        for i in range(12)
    ]

    for turn in long_turns:
        baseline.reply("user_stress", long_thread_id, turn)
        advanced.reply("user_stress", long_thread_id, turn)

    base_prompt_load = baseline.prompt_token_usage(long_thread_id)
    adv_prompt_load = advanced.prompt_token_usage(long_thread_id)

    # Advanced should have triggered compaction and kept prompt token load lower than baseline
    assert advanced.compaction_count(long_thread_id) > 0
    assert adv_prompt_load < base_prompt_load
