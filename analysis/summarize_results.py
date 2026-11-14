"""Score the published KG-RAG response files and print one accuracy table.

The response files in ``data/results`` were produced and published by the KG-RAG authors
(Baranzini Lab, UCSF). This script only reads them. It needs nothing beyond the Python
standard library, so it runs without a GPU, an API key or the project's requirements.

    python analysis/summarize_results.py
    python analysis/summarize_results.py --format json

Scoring rules
-------------
True/false: the response must contain exactly one of ``True`` / ``False`` (repeats of the same
value are fine). A response with both values, with neither, or with "Don't know" is counted as
wrong and reported in the ``unparsed`` column. Only questions present in the curated benchmark
file are scored, with the label taken from that file.

Multiple choice: the value of the ``"answer"`` key in the response is compared with the correct
option. Extraction follows ``notebooks/mcq_analysis.ipynb``, including its handling of nested
answers.

Accuracy is ``correct / questions`` over the full file. The KG-RAG paper reports bootstrap means
over samples of 150 questions, so its numbers can differ slightly from the ones printed here.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

MODELS = {
    "Llama-2-13b": "Llama_2_13b_chat_hf",
    "GPT-3.5-Turbo": "gpt_35_turbo",
    "GPT-4": "gpt_4",
}

TRUE_FALSE_FILES = {
    "prompt": "{model}_prompt_based_true_false_binary_response.csv",
    "kg_rag": (
        "{model}_PubMedBert_and_entity_recognition_based_node_retrieval_rag_based"
        "_true_false_binary_response.csv"
    ),
}

MCQ_FILES = {
    "prompt": "{model}_prompt_based_mcq_from_monarch_and_robokop_response.csv",
    "kg_rag": (
        "{model}_PubMedBert_entity_recognition_based_node_retrieval_rag_based"
        "_mcq_from_monarch_and_robokop_response.csv"
    ),
}

_TRUE_FALSE = re.compile(r"True|False|Don't know")


@dataclass
class Score:
    """Accuracy of one response file."""

    questions: int
    correct: int
    unparsed: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.questions if self.questions else 0.0


def extract_true_false(response: str) -> str | None:
    """Return ``"True"`` or ``"False"``, or None when the response has no single answer."""
    found = set(_TRUE_FALSE.findall(response))
    if found == {"True"} or found == {"False"}:
        return found.pop()
    return None


def extract_mcq(response: str) -> str | None:
    """Return the value of the ``"answer"`` key, or None when there is none.

    Follows the extraction in ``notebooks/mcq_analysis.ipynb``: the answer is the quoted value
    on the line of the ``"answer":`` key. When that value is a nested object (some Llama
    responses), the answer is the first value inside it.
    """
    key = '"answer":'
    if key not in response:
        return None
    lines = response.split(key)[-1].split("\n")
    value = lines[0].strip()
    if value == "{":
        if len(lines) < 2:
            return None
        value = lines[1].split(":")[-1]
        if '"' not in value:
            return None
    parts = value.split('"')
    answer = parts[1].strip() if len(parts) > 1 else value.strip()
    return answer or None


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def score_true_false(responses: list[dict[str, str]], benchmark: list[dict[str, str]]) -> Score:
    """Score true/false responses against the curated benchmark labels."""
    labels = {row["text"]: row["label"].strip() for row in benchmark}
    questions = correct = unparsed = 0
    for row in responses:
        label = labels.get(row["question"])
        if label is None:
            continue
        questions += 1
        answer = extract_true_false(row["llm_answer"])
        if answer is None:
            unparsed += 1
        elif answer == label:
            correct += 1
    return Score(questions, correct, unparsed)


def score_mcq(responses: list[dict[str, str]]) -> Score:
    """Score multiple-choice responses against the ``correct_answer`` column."""
    questions = correct = unparsed = 0
    for row in responses:
        questions += 1
        answer = extract_mcq(row["llm_answer"])
        if answer is None:
            unparsed += 1
        elif answer == row["correct_answer"].strip():
            correct += 1
    return Score(questions, correct, unparsed)


@dataclass
class Comparison:
    """Prompt-only and KG-RAG scores of one model on one task."""

    model: str
    task: str
    prompt: Score
    kg_rag: Score

    def to_dict(self) -> dict[str, object]:
        record = asdict(self)
        record["prompt"]["accuracy"] = self.prompt.accuracy
        record["kg_rag"]["accuracy"] = self.kg_rag.accuracy
        return record


def summarize(results_dir: Path, benchmark_dir: Path) -> list[Comparison]:
    """Score every model on both tasks, with and without KG-RAG."""
    benchmark = read_rows(benchmark_dir / "true_false_questions.csv")
    comparisons = []
    for name, prefix in MODELS.items():
        true_false = {
            method: score_true_false(
                read_rows(results_dir / pattern.format(model=prefix)), benchmark
            )
            for method, pattern in TRUE_FALSE_FILES.items()
        }
        comparisons.append(Comparison(name, "True/False", **true_false))
        mcq = {
            method: score_mcq(read_rows(results_dir / pattern.format(model=prefix)))
            for method, pattern in MCQ_FILES.items()
        }
        comparisons.append(Comparison(name, "Multiple choice", **mcq))
    return comparisons


def render_markdown(comparisons: list[Comparison]) -> str:
    lines = [
        "| Model | Task | Questions | Prompt only | KG-RAG | Change | Unparsed (prompt / KG-RAG) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        change = (row.kg_rag.accuracy - row.prompt.accuracy) * 100
        lines.append(
            f"| {row.model} | {row.task} | {row.prompt.questions} "
            f"| {row.prompt.accuracy:.1%} | {row.kg_rag.accuracy:.1%} | {change:+.1f} pts "
            f"| {row.prompt.unparsed} / {row.kg_rag.unparsed} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "data" / "results")
    parser.add_argument("--benchmark-dir", type=Path, default=REPO_ROOT / "data" / "benchmark_data")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    args = parser.parse_args(argv)

    try:
        comparisons = summarize(args.results_dir, args.benchmark_dir)
    except FileNotFoundError as error:
        print(f"error: missing file: {error.filename}", file=sys.stderr)
        return 1

    if args.format == "json":
        print(json.dumps([row.to_dict() for row in comparisons], indent=2))
    else:
        print(render_markdown(comparisons))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
