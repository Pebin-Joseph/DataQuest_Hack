import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def run_scripted(data_dir: Path, interval_s: float) -> None:
    """Deterministic sequence: add -> query -> update -> delete.

    This is ideal for a hackathon demo because the behavior is predictable:
    - A new fact appears
    - Then the fact changes
    - Then the source is deleted
    """

    scenario = [
        (
            "ADD",
            "breaking_news.txt",
            """TITLE: Breaking Update (v1)\nTIME: {ts}\n\nCompany X announced Product Alpha will launch on 15 Jan 2026.\nInvestors reacted positively.\n""",
        ),
        (
            "UPDATE",
            "breaking_news.txt",
            """TITLE: Correction (v2)\nTIME: {ts}\n\nCORRECTION: Company X clarified Product Alpha will launch on 20 Jan 2026 (not 15 Jan).\n""",
        ),
        (
            "ADD",
            "market_note.txt",
            """TITLE: Market Note\nTIME: {ts}\n\nAnalysts expect short-term volatility after the corrected launch date.\n""",
        ),
        (
            "DELETE",
            "breaking_news.txt",
            "",
        ),
    ]

    while True:
        for action, filename, template in scenario:
            fpath = data_dir / filename
            if action == "DELETE":
                try:
                    fpath.unlink(missing_ok=True)
                    print(f"SIM DELETE {filename}")
                except Exception as e:
                    print(f"SIM DELETE FAIL {filename}: {e}")
            elif action in ("ADD", "UPDATE"):
                text = template.format(ts=_now_iso())
                _write(fpath, text)
                print(f"SIM {action} {filename}")
            time.sleep(interval_s)


def run_random(data_dir: Path, interval_s: float, delete_prob: float) -> None:
    """Randomly adds/updates/deletes small docs to simulate a live stream."""

    topics = [
        "Pathway Live AI",
        "Hackathon Judging",
        "Streaming RAG",
        "Real-time Updates",
        "Docs Change Events",
    ]

    filenames = [f"live_item_{i:02d}.txt" for i in range(1, 11)]

    while True:
        existing = [p for p in data_dir.glob("live_item_*.txt") if p.is_file()]

        # delete sometimes
        if existing and random.random() < delete_prob:
            victim = random.choice(existing)
            try:
                victim.unlink(missing_ok=True)
                print(f"SIM DELETE {victim.name}")
            except Exception as e:
                print(f"SIM DELETE FAIL {victim.name}: {e}")
            time.sleep(interval_s)
            continue

        # otherwise add/update
        name = random.choice(filenames)
        topic = random.choice(topics)
        version = random.randint(1, 999)
        text = (
            f"TITLE: {topic}\n"
            f"TIME: {_now_iso()}\n"
            f"VERSION: {version}\n\n"
            f"Update: {topic} received new information at {_now_iso()}.\n"
            f"Key point: streaming systems must handle add/update/delete without restarts.\n"
        )
        _write(data_dir / name, text)
        print(f"SIM UPSERT {name}")
        time.sleep(interval_s)


def main() -> None:
    data_dir = Path(os.getenv("SIM_DATA_DIR", "data"))
    interval_s = float(os.getenv("SIM_INTERVAL_S", "6"))
    mode = os.getenv("SIM_MODE", "scripted").lower().strip()
    delete_prob = float(os.getenv("SIM_DELETE_PROB", "0.15"))

    print(
        "SIM START",
        {"mode": mode, "data_dir": str(data_dir), "interval_s": interval_s, "delete_prob": delete_prob},
    )

    data_dir.mkdir(parents=True, exist_ok=True)

    if mode == "random":
        run_random(data_dir, interval_s, delete_prob)
    else:
        run_scripted(data_dir, interval_s)


if __name__ == "__main__":
    main()
