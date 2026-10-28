"""Read pages with a model, greedily, in vLLM or transformers, and print its character
error rate: the edit distance from its reply to the page's text, over the text's
length.

    python sky/ocr_eval.py MODEL_DIR pages.parquet NAME [vllm|transformers|bux] [MAX_PIXELS]

``bux`` reads with a model bux exported as its own code (``bux export RUN OUT``),
one page at a time.
"""

import base64
import re
import sys
from typing import Any

import pyarrow.parquet as pq

THOUGHT = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def distance(a: str, b: str) -> int:
    """Return the edit distance from a to b, in characters."""
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        before, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            before, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, before + (x != y))
    return row[-1]


def by_vllm(model: str, chats: list) -> list[str]:
    """Return the model's replies to `chats`, by vLLM."""
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, max_model_len=4096, limit_mm_per_prompt={"image": 1})
    outs = llm.chat(chats, SamplingParams(temperature=0, max_tokens=1024))
    return [o.outputs[0].text for o in outs]


def by_transformers(
    model: str, chats: list, max_pixels: int | None = None
) -> list[str]:
    """Return the model's replies to `chats`, by transformers, 16 at a time, its
    images resized to at most `max_pixels`, if given."""
    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model, padding_side="left")
    if max_pixels is not None:
        size = dict(processor.image_processor.size)
        processor.image_processor.size = size | {"longest_edge": max_pixels}
    net = Qwen3_5ForConditionalGeneration.from_pretrained(
        model, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    out = []
    for i in range(0, len(chats), 16):
        inputs = processor.apply_chat_template(
            chats[i : i + 16],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to("cuda")
        with torch.no_grad():
            ids = net.generate(**inputs, max_new_tokens=1024, do_sample=False)
        new = ids[:, inputs["input_ids"].shape[1] :]
        out += processor.batch_decode(new, skip_special_tokens=False)
    return [o.split("<|im_end|>")[0] for o in out]


def by_bux(model: str, chats: list) -> list[str]:
    """Return the replies to `chats` of a model bux exported as its own code, by
    transformers, one page at a time."""
    import importlib
    import io

    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoTokenizer

    net: Any = AutoModelForCausalLM.from_pretrained(  # remote code, of no type
        model, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda"
    )
    vision = importlib.import_module(type(net).__module__.rsplit(".", 1)[0] + ".vision")
    tok = AutoTokenizer.from_pretrained(model)
    assert tok is not None
    token = tok.convert_tokens_to_ids("<|image_pad|>")
    prompt = tok(
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        "Read the text in the image.<|im_end|>\n<|im_start|>assistant\n"
    ).input_ids
    out = []
    for chat in chats:
        url = chat[0]["content"][0]["image_url"]["url"]
        data = base64.b64decode(url.split(",", 1)[1])
        page = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
        ids, images = vision.inputs(prompt, [page], token, net.config.bux_images)
        images = vision.Images(*(x.to("cuda") for x in images))
        with torch.no_grad():
            got = net.generate(
                input_ids=ids.to("cuda"),
                images=images,
                max_new_tokens=1024,
                do_sample=False,
            )
        out.append(str(tok.decode(got[0, ids.shape[1] :])).split("<|im_end|>")[0])
    return out


def main(
    model: str, path: str, name: str, backend: str, max_pixels: int | None
) -> None:
    rows = pq.read_table(path).to_pylist()
    chats, texts = [], []
    for r in rows:
        user, reply = r["messages"]
        image = next(p["image"] for p in user["content"] if p["type"] == "image")
        url = "data:image/png;base64," + base64.b64encode(image).decode()
        chats.append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": url}},
                        {"type": "text", "text": "Read the text in the image."},
                    ],
                }
            ]
        )
        texts.append(reply["content"][0]["text"])
    if backend == "transformers":
        replies = by_transformers(model, chats, max_pixels)
    else:
        replies = {"vllm": by_vllm, "bux": by_bux}[backend](model, chats)
    said = [THOUGHT.sub("", s).strip() for s in replies]
    errors = sum(distance(s, t) for s, t in zip(said, texts, strict=True))
    exact = sum(s == t for s, t in zip(said, texts, strict=True))
    print(
        f"{name}: CER {errors / sum(map(len, texts)):.4f}, {exact}/{len(texts)} exact"
    )
    print(f"{name}: e.g. {said[0]!r} for {texts[0]!r}")


if __name__ == "__main__":
    backend = sys.argv[4] if len(sys.argv) > 4 else "vllm"
    most = int(sys.argv[5]) if len(sys.argv) > 5 else None
    main(*sys.argv[1:4], backend, most)
