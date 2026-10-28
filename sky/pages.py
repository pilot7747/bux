"""Write conversations about rendered pages of text, for a smoke run of OCR.

python sky/pages.py data/ocr/pages.parquet 512
"""

import io
import random
import sys

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

WORDS = [
    "invoice",
    "total",
    "due",
    "receipt",
    "memo",
    "ship",
    "order",
    "paid",
    "net",
    "tax",
    "amount",
    "date",
    "page",
]


def page(rng: random.Random) -> tuple[bytes, str]:
    """Return a PNG of a few lines of random words, and its text."""
    lines = [
        " ".join(rng.choices(WORDS, k=rng.randint(3, 8)))
        for _ in range(rng.randint(2, 6))
    ]
    text = "\n".join(lines)
    image = Image.new(
        "RGB", (rng.choice([480, 640, 800]), rng.choice([240, 320, 480])), "white"
    )
    ImageDraw.Draw(image).text((16, 16), text, fill="black")
    out = io.BytesIO()
    image.save(out, "PNG")
    return out.getvalue(), text


def main(path: str, n: int, seed: int = 0) -> None:
    rng, rows = random.Random(seed), []
    for _ in range(n):
        data, text = page(rng)
        user = [
            {"type": "image", "image": data, "text": None},
            {"type": "text", "image": None, "text": "Read the text in the image."},
        ]
        reply = [{"type": "text", "image": None, "text": text}]
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": reply},
                ]
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), path)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 0)
