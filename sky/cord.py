"""Write CORD-v2's receipts (naver-clova-ix/cord-v2, CC-BY-4.0) as OCR conversations:
the user's turn the receipt's photo, the reply its text, a row of words a line.

    python sky/cord.py data/cord           # train, validation and test .parquet

A receipt's text is its words, by the rows CORD's annotators drew: rows top to
bottom, by their words' middle, and each row's words left to right.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

PROMPT = "Read the text in the image."


def text(ground_truth: str) -> str:
    """Return a receipt's text: its rows of words, top to bottom, left to right."""
    rows = defaultdict(list)
    for line in json.loads(ground_truth)["valid_line"]:
        for word in line["words"]:
            q = word["quad"]
            rows[word["row_id"]].append(
                ((q["y1"] + q["y3"]) / 2, q["x1"], word["text"])
            )
    ordered = sorted(rows.values(), key=lambda ws: sum(w[0] for w in ws) / len(ws))
    return "\n".join(
        " ".join(t for _, _, t in sorted(ws, key=lambda w: w[1])) for ws in ordered
    )


def conversation(image: bytes, reply: str) -> list[dict]:
    """Return the conversation of reading `image` as `reply`."""
    user = [
        {"type": "image", "image": image, "text": None},
        {"type": "text", "image": None, "text": PROMPT},
    ]
    return [
        {"role": "user", "content": user},
        {
            "role": "assistant",
            "content": [{"type": "text", "image": None, "text": reply}],
        },
    ]


def main(out: str) -> None:
    folder = Path(snapshot_download("naver-clova-ix/cord-v2", repo_type="dataset"))
    Path(out).mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        rows = []
        for f in sorted((folder / "data").glob(f"{split}-*.parquet")):
            for r in pq.read_table(f).to_pylist():
                reply = text(r["ground_truth"])
                rows.append({"messages": conversation(r["image"]["bytes"], reply)})
        pq.write_table(pa.Table.from_pylist(rows), Path(out) / f"{split}.parquet")
        print(f"{split}: {len(rows)} receipts")


if __name__ == "__main__":
    main(sys.argv[1])
