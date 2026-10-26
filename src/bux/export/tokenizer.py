"""A run's tokenizer in an exported directory, so that ``AutoTokenizer`` loads it from
there, and its special tokens in ``generation_config.json``, so that ``generate``
starts, stops and pads as the run's documents do.

The tokenizer is the one ``tokenizer(name=...)`` names: a Hub repo's, its chat
template and settings with it, or a ``tokenizer.json``; with bux's `bos` and `eos`,
its `added` tokens after the vocabulary, and the end of a document as its padding
where it has none. Each special token keeps the id bux gave it, or the export fails.
"""

from pathlib import Path
from typing import Any

from bux.data import tokenizer as bux_tokenizer
from bux.train.config import TokenizerConfig


def save(config: TokenizerConfig, out: Path | str, vocab_size: int) -> Any:
    """Write the tokenizer of `config` and the generation config to `out`.

    Args:
        config: The run's ``tokenizer`` setting.
        out: The exported model's directory.
        vocab_size: The model's vocabulary, which the tokenizer's ids must fit.

    Returns:
        The tokenizer, as transformers loads it from `out`.

    Raises:
        ValueError: transformers' tokenizer gives a special token another id than
            bux's, or more ids than the model has.
    """
    from transformers import AutoTokenizer, GenerationConfig

    made = _made(config)
    made.save_pretrained(out)
    tok: Any = AutoTokenizer.from_pretrained(out)
    _check_ids(tok, config, vocab_size)
    GenerationConfig(
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    ).save_pretrained(out)
    return tok


def _made(config: TokenizerConfig) -> Any:
    """Return transformers' tokenizer of `config`, with bux's special tokens."""
    from tokenizers import AddedToken
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    name = config.name
    tok: Any = (
        PreTrainedTokenizerFast(tokenizer_file=name)
        if Path(name).is_file()
        else AutoTokenizer.from_pretrained(name)
    )
    added = [AddedToken(t, special=True, normalized=False) for t in config.added]
    tok.add_tokens(added, special_tokens=True)
    tok.bos_token, tok.eos_token = config.bos, config.eos
    if tok.pad_token is None:
        tok.pad_token = config.eos
    return tok


def _check_ids(tok: Any, config: TokenizerConfig, vocab_size: int) -> None:
    """Raise unless `tok` gives bux's special tokens bux's ids, within the model's."""
    ours = bux_tokenizer.load(config)
    for token in (config.bos, config.eos, *config.added):
        theirs = tok.convert_tokens_to_ids(token)
        if theirs != ours.special(token):
            raise ValueError(
                f"transformers' tokenizer of {config.name} gives {token} the id "
                f"{theirs}, and bux's {ours.special(token)}"
            )
    if len(tok) > vocab_size:
        raise ValueError(
            f"the tokenizer makes {len(tok)} ids, and the model has {vocab_size}"
        )
