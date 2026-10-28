"""The inspector's forms: a node or a setting, field by field, with where the file
writes each field and whether the page can set it there.

`inspect` returns a ``Form`` of the page's JSON API::

    {"title": "gqa", "subtitle": "attn", "doc": "Grouped-query attention; ...",
     "where": {"span": [104, 165], "file": None, "line": 6},
     "fields": [{"key": "num_kv_heads", "type": "int", "value": "8",
                 "written": "8", "input": "int", "editable": True, ...}, ...]}

A field says what `bux.studio.edits` would do with it: ``editable`` is False, with a
``note`` saying why, where a ``set`` of it raises, such as a value that a macro's
parameter sets. Its ``key`` is what a ``set`` of the same target takes.
"""

import inspect as pyinspect
import typing
from collections.abc import Mapping

from pydantic.fields import FieldInfo

from bux import config as cfg
from bux.lang import ir, reference
from bux.lang import syntax as ast
from bux.lang.compiler import SETTINGS, TRAIN_SETTINGS, Written
from bux.lang.printer import field_order
from bux.studio.palette import as_source
from bux.studio.targets import (
    EditError,
    File,
    Place,
    amendable,
    blocked,
    count,
    extent,
    node_at,
    node_path,
    options,
    place,
    tag,
)
from bux.train import config as tcfg

__all__ = ["inspect"]

# Settings written as sums, which the page edits as text.
_WHOLE = (tcfg.MixtureConfig, tcfg.ObjectiveConfig)

_MODEL = {
    "d_model": "the width of the residual stream",
    "vocab_size": "the token ids that embeddings and heads take",
    "residual": "the residual algebra that every [f] merges by; none: x + f(x)",
    "init": "how the weight matrices start, by role; none: normal(std=0.02)",
}
_KINDS = {
    ir.Seq: ("sequence", "A composition: each part takes what the one before makes."),
    ir.Residual: ("residual", "A residual branch, [f]: x + f(x)."),
    ir.Sum: ("sum", "Parallel branches on the same input, summed: f + g."),
    ir.Prod: ("product", "Parallel branches, multiplied: f * g gates or scales."),
    ir.Neg: ("negation", "A negated branch, -f."),
    ir.Const: ("constant", "A constant branch, such as the 1 of 1 - f."),
    ir.Identity: ("identity", "The identity, _: the input as it is."),
    ir.Repeat: ("repeat", "Fresh copies of a block in sequence: block^n."),
    ir.Tap: ("tap", "@name pushes the stream onto a tap."),
    ir.Ref: ("read", "$name reads the top of a tap, and $name! pops it."),
    ir.Image: (
        "image",
        (
            "An encoder of the row's images, image(f, token=...): its outputs replace "
            "the image tokens."
        ),
    ),
}


def inspect(
    source: str, target: dict, *, picks: Mapping[str, int], path: str | None
) -> dict:
    """Return the inspector's form of `target` in the run that `picks` selects.

    Args:
        source: The file's text.
        target: ``{"node": id}``, ``{"setting": name}`` or ``{"model": true}``, as
            the page's JSON API writes them.
        picks: The option each axis of the sweep takes, by axis id.
        path: Where the file is, which ``use`` searches from; None if not saved.

    Returns:
        The form, as JSON: its title, subtitle, doc, where, and fields.

    Raises:
        EditError: For a file that does not compile, or a target it does not have.
    """
    f = File(source, picks, path)
    if not isinstance(target, dict):
        raise EditError("a target is {node}, {setting} or {model}")
    if "node" in target:
        return _node_form(f, str(target["node"]))
    if "setting" in target:
        return _setting_form(f, str(target["setting"]))
    if target.get("model"):
        return _model_form(f)
    raise EditError("a target is {node}, {setting} or {model}")


def _form(
    title: str, subtitle: str | None, doc: str, where: dict | None, fields: list
) -> dict:
    return {
        "title": title,
        "subtitle": subtitle,
        "doc": doc,
        "where": where,
        "fields": fields,
    }


def _doc(cls: type) -> str:
    """Return the first paragraph of a class's own docstring, on one line."""
    doc = cls.__dict__.get("__doc__") or ""
    return " ".join(pyinspect.cleandoc(doc).split("\n\n")[0].split())


def _default(info: FieldInfo) -> str | None:
    if info.is_required():
        return None
    try:
        return reference.default_text(info)
    except TypeError:
        return None


def _input(annotation: object, value: object) -> tuple[str, list[str] | None]:
    """Return how the page takes a value of this type, and a choice's options."""
    if isinstance(value, cfg.Config) and not isinstance(value, _WHOLE):
        return "config", None
    members = reference.members(annotation)
    literals = [
        str(v)
        for m in members
        if typing.get_origin(m) is typing.Literal
        for v in typing.get_args(m)
    ]
    none = type(None) in members
    rest = [
        m
        for m in members
        if typing.get_origin(m) is not typing.Literal and m is not type(None)
    ]
    if literals and not rest:
        return "choice", literals + ["none"] * none
    if rest == [bool]:
        return ("choice", ["true", "false", "none"]) if none else ("bool", None)
    if rest and all(m is int for m in rest):
        return "int", None
    if rest and all(m in (int, float) for m in rest):
        return "float", None
    return "text", None


def _field(
    key: str,
    annotation: object,
    value: object,
    *,
    doc: str,
    info: FieldInfo | None = None,
    **more: object,
) -> dict:
    kind, choices = _input(annotation, value)
    field = {
        "key": key,
        "type": reference.type_text(annotation),
        "doc": doc,
        "required": info.is_required() if info is not None else True,
        "default": _default(info) if info is not None else None,
        "value": as_source(value),
        "written": None,
        "where": None,
        "input": kind,
        "options": choices,
        "sweep": None,
        "editable": True,
        "note": None,
        "fields": None,
    }
    return field | more


# --- Configs -------------------------------------------------------------------------


def _config_fields(
    f: File,
    config: cfg.Config,
    top: Written | None,
    amend: str | None,
    path: tuple[str, ...] = (),
    shown: tuple[str, ...] = (),
) -> list[dict]:
    """Return the fields of `config`, the one at `path` of the config written at
    `top`, keyed as `shown` then `path` dotted."""
    cls = type(config)
    notes = reference.notes(cls)
    out = []
    for name in field_order(cls):
        if name == "type":
            continue
        info, value, key = cls.model_fields[name], getattr(config, name), (*path, name)
        found = place(f, top, key)
        editable, note = _editable(f, found, top, amend)
        nested = (
            _config_fields(f, value, top, amend, key, shown)
            if isinstance(value, cfg.Config) and not isinstance(value, _WHOLE)
            else None
        )
        out.append(
            _field(
                ".".join((*shown, *key)),
                info.annotation,
                value,
                doc=notes.get(name, ""),
                info=info,
                editable=editable,
                note=note,
                fields=nested,
                **_written(f, found),
            )
        )
    return out


def _written(f: File, found: Place) -> dict:
    """Return the written, where and sweep of a field that `found` places."""
    if found.found is None:
        return {}
    value = found.found.arg.value
    return {
        "written": f.text(extent(f.source, value)),
        "where": f.where(found.found.arg.span),
        "sweep": options(f.source, value) if isinstance(value, ast.Alt) else None,
    }


def _editable(
    f: File, found: Place, top: Written | None, amend: str | None
) -> tuple[bool, str | None]:
    """Return whether a ``set`` of the field that `found` places applies, as
    `bux.studio.edits` sets it, and a note on where it lands, or why it does not."""
    site = found.found
    if site is not None and f.local(site.arg.span):
        why = blocked(f, site)
        return why is None, why
    span = site.arg.span if site is not None else top.span if top is not None else None
    module = f.module(span) if span is not None and not f.local(span) else None
    if found.into is not None:
        return True, (f"written in {module}; set here, in this file" if site else None)
    if amend is not None:
        return True, (
            f"written in {module}; set here, it amends {amend}" if module else None
        )
    return False, f"written in {module or 'a module'}, with no name to amend it by"


# --- Nodes ---------------------------------------------------------------------------


def _node_form(f: File, node_id: str) -> dict:
    node = node_at(f, node_id)
    span = f.trace.nodes.get(node_path(node_id))
    where = f.where(span) if span is not None else None
    if isinstance(node, ir.Layer):
        return _layer_form(f, node, where)
    kind, doc = _KINDS.get(type(node), ("node", ""))
    fields = []
    if isinstance(node, ir.Repeat):
        fields.append(_times(f, node_id, node))
    st = f.trace.statements.get(node.label) if node.label is not None else None
    if st is not None and st.params is None:
        fields.append(_definition(f, st, "block"))
    if isinstance(node, ir.Residual) and f.trace.arch.residual is not None:
        doc += f" It merges by {as_source(f.trace.arch.residual)}."
    if node.label is not None:
        return _form(node.label, kind, doc, where, fields)
    return _form(kind, None, doc, where, fields)


def _layer_form(f: File, node: ir.Layer, where: dict | None) -> dict:
    config = node.config
    if isinstance(config, cfg.ActivationConfig):
        return _form(config.fn, node.label, "An activation, elementwise.", where, [])
    top = f.trace.written(config)
    fields = _config_fields(f, config, top, amendable(f, node.label))
    where = f.where(top.span) if top is not None else where
    return _form(tag(config), node.label, _doc(type(config)), where, fields)


def _times(f: File, node_id: str, node: ir.Repeat) -> dict:
    field = _field(
        "times", int, node.times, doc="how many fresh copies, one after another"
    )
    try:
        written = count(f, node_id)
    except EditError as err:
        return field | {"editable": False, "note": str(err)}
    if written is None:
        return field
    repeat = written.repeat.expr
    assert isinstance(repeat, ast.BinOp)
    span = extent(f.source, repeat.right)
    return field | {
        "written": f.text(span),
        "where": f.where(span),
        "editable": written.number is not None,
        "note": None if written.number is not None else f"the count is {written.why}",
    }


def _definition(f: File, st: ast.Statement, kind: str) -> dict:
    """Return the field of a definition as a whole: its text."""
    span = extent(f.source, st.body)
    note = None
    if not f.local(st.span):
        note = f"written in {f.module(st.span)}; a definition here takes its place"
    return _field(st.name, str, "", doc=f"the {kind}'s definition") | {
        "type": kind,
        "value": f.text(span),
        "written": f.text(span),
        "where": f.where(st.span),
        "note": note,
    }


# --- Settings ------------------------------------------------------------------------


def _model_form(f: File) -> dict:
    arch = f.trace.arch
    fields = [
        _setting_field(f, name, ir.Architecture.model_fields[name], getattr(arch, name))
        for name in SETTINGS
    ]
    doc = "The model's settings: its width, vocabulary, residual algebra and init."
    return _form("model", None, doc, None, fields)


def _setting_form(f: File, name: str) -> dict:
    st = f.trace.statements.get(name)
    where = f.where(st.span) if st is not None else None
    if name in SETTINGS:
        info = ir.Architecture.model_fields[name]
        field = _setting_field(f, name, info, getattr(f.trace.arch, name))
        return _form(name, None, _MODEL[name], where, [field])
    if name not in TRAIN_SETTINGS:
        if st is None or st.params is not None:
            raise EditError(f"no setting {name!r}")
        return _form(name, None, "", where, [_definition(f, st, "setting")])
    if f.trace.train is None:
        raise EditError("this file is a model, not a run: it has no training settings")
    info = tcfg.TrainConfig.model_fields[name]
    value = getattr(f.trace.train, name)
    doc = reference.notes(tcfg.TrainConfig).get(name, "")
    if isinstance(value, tcfg.MixtureConfig):
        return _form(name, "mix", doc, where, [_mixture(f, name, info, value)])
    if isinstance(value, cfg.Config) and not isinstance(value, _WHOLE):
        top = f.trace.written(value)
        where = f.where(top.span) if top is not None else where
        fields = _config_fields(f, value, top, name)
        return _form(tag(value), name, _doc(type(value)), where, fields)
    field = _setting_field(f, name, info, value)
    if isinstance(value, tuple) and all(isinstance(v, cfg.Config) for v in value):
        field["fields"] = [_item(f, i, item) for i, item in enumerate(value)]
    return _form(name, None, doc, where, [field])


def _setting_field(f: File, name: str, info: FieldInfo, value: object) -> dict:
    """Return the field of a top-level setting, as a whole."""
    st = f.trace.statements.get(name)
    notes = reference.notes(tcfg.TrainConfig)
    doc = _MODEL.get(name) or notes.get(name, "")
    field = _field(name, info.annotation, value, doc=doc, info=info)
    if isinstance(value, cfg.Config) and not isinstance(value, _WHOLE):
        top = f.trace.written(value)
        field["fields"] = _config_fields(f, value, top, name, shown=(name,))
    if st is None:
        return field
    if st.amends:  # its definition with amendments: the latest call writes it
        top = f.trace.written(value) if isinstance(value, cfg.Config) else None
        span = top.span if top is not None else st.span
        return field | {"where": f.where(span)}
    span = extent(f.source, st.body)
    note = None
    if not f.local(st.span):
        note = f"written in {f.module(st.span)}; a value here takes its place"
    sweep = options(f.source, st.body) if isinstance(st.body, ast.Alt) else None
    return field | {
        "written": f.text(span),
        "where": f.where(st.span),
        "sweep": sweep,
        "note": note,
    }


def _mixture(f: File, name: str, info: FieldInfo, mix: tcfg.MixtureConfig) -> dict:
    """Return the field of a mix of data sources: its text, and its sources read-only."""
    field = _setting_field(f, name, info, mix)
    note = "a mix of sources: edit it as text, such as 0.8 * web + 0.2 * code"
    sources = [
        _field(f"sources.{i}", str, "", doc=f"source {i + 1}, with its weight")
        | {
            "type": "source",
            "value": f"{part.weight!r} * {as_source(part.source)}",
            "editable": False,
            "note": "read-only: edit the mix",
        }
        for i, part in enumerate(mix.sources)
    ]
    return field | {"input": "text", "note": field["note"] or note, "fields": sources}


def _item(f: File, i: int, item: cfg.Config) -> dict:
    """Return the field of one config of a list, such as a phase of the curriculum."""
    top = f.trace.written(item)
    field = _field(str(i), type(item), item, doc=_doc(type(item)))
    field["fields"] = _config_fields(f, item, top, None, shown=(str(i),))
    if top is None:
        return field | {"editable": False, "note": "not written as a call"}
    return field | {"written": f.text(top.span), "where": f.where(top.span)}
