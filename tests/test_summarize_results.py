import json
from pathlib import Path

import pytest
import summarize_results as sr

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
BENCHMARK = ROOT / "data" / "benchmark_data"


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ('{"answer": "True"}', "True"),
        ('{\r\n  "answer": "False"\r\n}', "False"),
        ("True. The statement is True.", "True"),
        ("It could be True or False", None),
        ("Don't know", None),
        ("", None),
    ],
)
def test_extract_true_false(response, expected):
    assert sr.extract_true_false(response) == expected


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ('{"answer": "HLA-B"}', "HLA-B"),
        ('{\r\n  "answer": "NOD2"\r\n}', "NOD2"),
        ('  {\r\n  "answer": {\r\n    "gene": "TERT",\r\n    "disease": "x"\r\n  }\r\n}', "TERT"),
        ('{"answer": " STAT4 "}', "STAT4"),
        ("The gene is HLA-B", None),
        ('{"answer": {', None),
    ],
)
def test_extract_mcq(response, expected):
    assert sr.extract_mcq(response) == expected


def test_score_true_false_uses_benchmark_labels_and_skips_other_questions():
    benchmark = [{"text": "q1", "label": "True"}, {"text": "q2", "label": "False"}]
    responses = [
        {"question": "q1", "label": "ignored", "llm_answer": '{"answer": "True"}'},
        {"question": "q2", "label": "ignored", "llm_answer": '{"answer": "True"}'},
        {"question": "not in benchmark", "label": "True", "llm_answer": '{"answer": "True"}'},
    ]
    assert sr.score_true_false(responses, benchmark) == sr.Score(questions=2, correct=1, unparsed=0)


def test_score_true_false_counts_unparsed_as_wrong():
    benchmark = [{"text": "q1", "label": "True"}]
    responses = [{"question": "q1", "label": "True", "llm_answer": "I am not sure"}]
    score = sr.score_true_false(responses, benchmark)
    assert score == sr.Score(questions=1, correct=0, unparsed=1)
    assert score.accuracy == 0.0


def test_score_mcq():
    responses = [
        {"question": "a", "correct_answer": "HLA-B", "llm_answer": '{"answer": "HLA-B"}'},
        {"question": "b", "correct_answer": "HLA-B", "llm_answer": '{"answer": "NOD2"}'},
        {"question": "c", "correct_answer": "HLA-B", "llm_answer": "no json here"},
    ]
    score = sr.score_mcq(responses)
    assert score == sr.Score(questions=3, correct=1, unparsed=1)
    assert score.accuracy == pytest.approx(1 / 3)


def test_empty_score_has_zero_accuracy():
    assert sr.Score(questions=0, correct=0, unparsed=0).accuracy == 0.0


def test_published_results_are_scored_as_expected():
    rows = {(row.model, row.task): row for row in sr.summarize(RESULTS, BENCHMARK)}
    assert len(rows) == 6
    for (_, task), row in rows.items():
        expected = 311 if task == "True/False" else 306
        assert row.prompt.questions == row.kg_rag.questions == expected

    gpt4_mcq = rows[("GPT-4", "Multiple choice")]
    assert (gpt4_mcq.prompt.correct, gpt4_mcq.kg_rag.correct) == (209, 227)
    llama_mcq = rows[("Llama-2-13b", "Multiple choice")]
    assert (llama_mcq.prompt.correct, llama_mcq.kg_rag.correct) == (94, 161)
    # KG-RAG is at least as accurate as prompting alone for every model and task.
    assert all(row.kg_rag.accuracy >= row.prompt.accuracy for row in rows.values())


def test_main_prints_markdown_and_json(capsys):
    assert sr.main([]) == 0
    table = capsys.readouterr().out
    assert table.startswith("| Model | Task |")
    assert len(table.strip().splitlines()) == 8

    assert sr.main(["--format", "json"]) == 0
    records = json.loads(capsys.readouterr().out)
    assert len(records) == 6
    assert {"model", "task", "prompt", "kg_rag"} <= records[0].keys()
    assert 0 <= records[0]["kg_rag"]["accuracy"] <= 1


def test_main_reports_a_missing_results_folder(tmp_path, capsys):
    assert sr.main(["--results-dir", str(tmp_path)]) == 1
    assert "missing file" in capsys.readouterr().err
