import argparse
import sqlite3
from typing import NamedTuple, List
from pathlib import Path

from eval_util import write_eval_result

QUERY = """
SELECT
    sut_name,
    AVG(CASE WHEN relation_result = 'True' THEN 1 ELSE 0 END) as acc
FROM mtc_results
GROUP BY sut_name
"""


class SutResult(NamedTuple):
    sut_name: str
    acc: float


def run(result_path: Path, db_path: Path):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(QUERY)
        results = [SutResult(*r) for r in cursor.fetchall()]

    print(results)

    output_path = result_path / "killable_mutants"
    # TODO: extract mutant name from SUT name
    for result in results:
        write_eval_result(output_path / result.sut_name, name="metamorphic", acc=result.acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results_path')
    parser.add_argument('db_path')
    args = parser.parse_args()

    run(Path(args.results_path), Path(args.db_path))
