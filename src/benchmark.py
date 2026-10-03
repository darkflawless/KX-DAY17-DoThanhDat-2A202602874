from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tabulate import tabulate

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read JSON conversations from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    """Return fraction of expected facts appearing in the answer."""
    if not expected:
        return 1.0
    ans_lower = answer.lower()
    matches = sum(1 for exp in expected if exp.lower() in ans_lower)
    return matches / len(expected)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Lightweight response quality score combining recall coverage and response structure."""
    rec = recall_points(answer, expected)
    length_bonus = min(1.0, len(answer.strip()) / 60.0)
    return round(0.75 * rec + 0.25 * length_bonus, 2)


def run_agent_benchmark(
    agent_name: str,
    agent: BaselineAgent | AdvancedAgent,
    conversations: list[dict[str, Any]],
    config: LabConfig,
) -> BenchmarkRow:
    """Evaluate one agent over multiple conversations and recall questions."""
    all_recall_scores: list[float] = []
    all_quality_scores: list[float] = []

    # Record initial memory size
    initial_memory: dict[str, int] = {}
    if hasattr(agent, "memory_file_size"):
        for conv in conversations:
            uid = conv["user_id"]
            if uid not in initial_memory:
                initial_memory[uid] = agent.memory_file_size(uid)

    for conv in conversations:
        user_id = conv["user_id"]
        conv_id = conv["id"]

        # Feed turns within primary conversation thread
        for turn in conv["turns"]:
            agent.reply(user_id, conv_id, turn)

        # Ask recall questions in fresh cross-session threads
        for q_idx, rq in enumerate(conv.get("recall_questions", [])):
            recall_thread_id = f"{conv_id}_recall_{q_idx}"
            reply_dict = agent.reply(user_id, recall_thread_id, rq["question"])
            answer = reply_dict.get("reply", "")
            expected = rq.get("expected_contains", [])

            score = recall_points(answer, expected)
            quality = heuristic_quality(answer, expected)
            all_recall_scores.append(score)
            all_quality_scores.append(quality)

    agent_tokens = agent.token_usage()
    prompt_tokens = agent.prompt_token_usage()
    avg_recall = (
        sum(all_recall_scores) / len(all_recall_scores)
        if all_recall_scores
        else 0.0
    )
    avg_quality = (
        sum(all_quality_scores) / len(all_quality_scores)
        if all_quality_scores
        else 0.0
    )
    compactions = agent.compaction_count()

    memory_growth = 0
    if hasattr(agent, "memory_file_size"):
        for uid, init_sz in initial_memory.items():
            curr_sz = agent.memory_file_size(uid)
            memory_growth += max(0, curr_sz - init_sz)

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=agent_tokens,
        prompt_tokens_processed=prompt_tokens,
        recall_score=round(avg_recall, 3),
        response_quality=round(avg_quality, 3),
        memory_growth_bytes=memory_growth,
        compactions=compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Print tabulated benchmark results matching the 6 required columns."""
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    table_data = []
    for r in rows:
        table_data.append(
            [
                r.agent_name,
                f"{r.agent_tokens_only:,}",
                f"{r.prompt_tokens_processed:,}",
                f"{r.recall_score * 100:.1f}%",
                f"{r.response_quality:.2f}",
                f"{r.memory_growth_bytes:,}",
                r.compactions,
            ]
        )
    return tabulate(table_data, headers=headers, tablefmt="github")


def main() -> None:
    """Run both Standard Benchmark and Long-Context Stress Benchmark."""
    root = Path(__file__).resolve().parent.parent
    config = load_config(root)

    standard_path = config.data_dir / "conversations.json"
    stress_path = config.data_dir / "advanced_long_context.json"

    # 1. Standard Benchmark
    print("=" * 80)
    print("Standard Benchmark (data/conversations.json)")
    print("=" * 80)
    if (config.state_dir / "profiles").exists():
        shutil.rmtree(config.state_dir / "profiles")

    std_conversations = load_conversations(standard_path)
    baseline_std = BaselineAgent(config, force_offline=True)
    advanced_std = AdvancedAgent(config, force_offline=True)

    row_base_std = run_agent_benchmark(
        "Baseline Agent", baseline_std, std_conversations, config
    )
    row_adv_std = run_agent_benchmark(
        "Advanced Agent", advanced_std, std_conversations, config
    )
    print(format_rows([row_base_std, row_adv_std]))
    print()

    # 2. Long-Context Stress Benchmark
    print("=" * 80)
    print("Long-Context Stress Benchmark (data/advanced_long_context.json)")
    print("=" * 80)
    if (config.state_dir / "profiles").exists():
        shutil.rmtree(config.state_dir / "profiles")

    stress_conversations = load_conversations(stress_path)
    baseline_stress = BaselineAgent(config, force_offline=True)
    advanced_stress = AdvancedAgent(config, force_offline=True)

    row_base_stress = run_agent_benchmark(
        "Baseline Agent", baseline_stress, stress_conversations, config
    )
    row_adv_stress = run_agent_benchmark(
        "Advanced Agent", advanced_stress, stress_conversations, config
    )
    print(format_rows([row_base_stress, row_adv_stress]))
    print()


if __name__ == "__main__":
    main()
