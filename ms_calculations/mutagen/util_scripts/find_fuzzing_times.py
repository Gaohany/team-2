from pathlib import Path
import json
from datetime import datetime


def get_time(p: Path):
    c = json.loads(p.read_text())
    print("  ", c['__version_entry__'][0]['__Time__'])
    parsed_date = datetime.strptime(c['__version_entry__'][0]['__Time__'], '%Y-%m-%d %H:%M:%S')
    return parsed_date


def run():
    fuzzing_root = Path('data', 'fuzzing', 'lenet')

    for sub_folder in sorted(fuzzing_root.iterdir()):
        if not sub_folder.is_dir():
            continue

        a = sub_folder / 'crashes_rec'

        dates = [
            get_time(p) for p in a.glob('*.json')
        ]

        mi = min(dates)
        ma = max(dates)

        ra = ma - mi

        print(sub_folder.name, ra)


if __name__ == '__main__':
    run()