"""Cheap checks that the files the KG-RAG scripts read are present and well formed."""

import csv
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))


def count_rows(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as handle:
        return len(list(csv.DictReader(handle)))


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: path.name)
def test_notebook_is_valid_json_with_cells(notebook):
    content = json.loads(notebook.read_text(encoding="utf-8"))
    assert content["cells"]


def test_config_files_parse_and_have_the_keys_the_code_reads():
    yaml = pytest.importorskip("yaml")
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    for key in (
        "CONTEXT_VOLUME",
        "QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD",
        "QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY",
        "VECTOR_DB_PATH",
        "NODE_CONTEXT_PATH",
        "SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL",
        "SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL",
    ):
        assert key in config

    prompts = yaml.safe_load((ROOT / "system_prompts.yaml").read_text(encoding="utf-8"))
    assert "DISEASE_ENTITY_EXTRACTION" in prompts


def test_benchmark_files_have_the_expected_size():
    benchmark = ROOT / "data" / "benchmark_data"
    assert count_rows(benchmark / "true_false_questions.csv") == 311
    assert count_rows(benchmark / "mcq_questions.csv") == 306
