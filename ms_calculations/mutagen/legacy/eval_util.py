import sqlite3
import json
from typing import NamedTuple, List
from pathlib import Path


TEST_RESULT_TABLE_CREATION = """
CREATE TABLE test_results (
    sut_name VARCHAR,
    sut_training INT,
    dataset TEXT,
    sample_id INT8,
    label INT,
    output INT,
    result BOOLEAN,
    confidence FLOAT,
    latent_space FLOAT[43],
    training_time FLOAT,
    evaluation_time FLOAT,
    is_duplicate BOOLEAN,
)
"""

TEST_RESULT_TABLE_INSERT = """
INSERT INTO test_results (
    sut_name,
    sut_training,
    dataset,
    sample_id,
    label,
    output,
    result,
    confidence,
    latent_space
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


class TestResult(NamedTuple):
    sut_name: str
    sut_training: int
    dataset: str
    sample_id: int
    label: int
    output: int
    result: int
    confidence: float
    latent_space: str


def write_output_db(result_path: Path, results: List[TestResult]):
    file_path = result_path / 'eval.db'
    if file_path.exists():
        file_path.unlink()

    with sqlite3.connect(file_path) as conn:
        conn.execute(TEST_RESULT_TABLE_CREATION)
        conn.executemany(TEST_RESULT_TABLE_INSERT, results)
        conn.commit()
        conn.close()


def write_eval_result(mutant_folder: Path, name: str, acc: float):
    print(f"{name: <10} {mutant_folder.name: <30} {acc:07.3%}")

    eval_json = mutant_folder / 'eval.json'
    j = {} if not eval_json.exists() else json.loads(eval_json.read_text())
    j[name] = acc
    eval_json.write_text(json.dumps(j, indent=4))
