"""Language server for .bux files: ``bux-lsp`` speaks LSP over stdio.

Diagnostics, completion and hover come from ``bux.lang.lsp.features``; this module
only converts between LSP positions and offsets into the source.
"""

import sys

try:
    from lsprotocol import types as lsp
    from pygls.lsp.server import LanguageServer
    from pygls.workspace import TextDocument
except ImportError:  # pragma: no cover
    sys.exit(
        "bux-lsp needs pygls: install bux with the 'lsp' extra (uv sync --extra lsp)"
    )

from bux.lang.lsp import features

_KINDS = {
    "setting": lsp.CompletionItemKind.Field,
    "value": lsp.CompletionItemKind.EnumMember,
    "layer": lsp.CompletionItemKind.Class,
    "config": lsp.CompletionItemKind.Struct,
    "activation": lsp.CompletionItemKind.Function,
    "definition": lsp.CompletionItemKind.Variable,
    "macro": lsp.CompletionItemKind.Function,
    "parameter": lsp.CompletionItemKind.Variable,
    "tap": lsp.CompletionItemKind.Reference,
    "keyword": lsp.CompletionItemKind.Keyword,
}

server = LanguageServer("bux", "0.1.0")


def _range(doc: TextDocument, start: int, end: int) -> lsp.Range:
    return lsp.Range(
        start=doc.client_position_at_offset(start),
        end=doc.client_position_at_offset(end),
    )


def _markdown(text: str) -> lsp.MarkupContent:
    return lsp.MarkupContent(kind=lsp.MarkupKind.Markdown, value=text)


def _publish(ls: LanguageServer, uri: str) -> None:
    doc = ls.workspace.get_text_document(uri)
    diagnostics = [
        lsp.Diagnostic(
            range=_range(doc, d.start, d.end),
            message=d.message,
            severity=lsp.DiagnosticSeverity.Error,
            source="bux",
        )
        for d in features.diagnostics(doc.source, doc.path)
    ]
    ls.text_document_publish_diagnostics(
        lsp.PublishDiagnosticsParams(
            uri=uri, version=doc.version, diagnostics=diagnostics
        )
    )


@server.feature(lsp.TEXT_DOCUMENT_DID_OPEN)
def _did_open(ls: LanguageServer, params: lsp.DidOpenTextDocumentParams) -> None:
    _publish(ls, params.text_document.uri)


@server.feature(lsp.TEXT_DOCUMENT_DID_CHANGE)
def _did_change(ls: LanguageServer, params: lsp.DidChangeTextDocumentParams) -> None:
    _publish(ls, params.text_document.uri)


@server.feature(lsp.TEXT_DOCUMENT_DID_CLOSE)
def _did_close(ls: LanguageServer, params: lsp.DidCloseTextDocumentParams) -> None:
    ls.text_document_publish_diagnostics(
        lsp.PublishDiagnosticsParams(uri=params.text_document.uri, diagnostics=[])
    )


@server.feature(
    lsp.TEXT_DOCUMENT_COMPLETION,
    lsp.CompletionOptions(trigger_characters=["(", ",", ".", "=", "@", "$", '"']),
)
def _completion(ls: LanguageServer, params: lsp.CompletionParams) -> lsp.CompletionList:
    doc = ls.workspace.get_text_document(params.text_document.uri)
    offset = doc.offset_at_position(params.position)
    items = [
        lsp.CompletionItem(
            label=c.label,
            kind=_KINDS.get(c.kind),
            detail=c.detail or None,
            documentation=_markdown(c.documentation) if c.documentation else None,
            insert_text=c.insert_text,
            sort_text=f"{c.sort}{c.label}",
        )
        for c in features.complete(doc.source, offset, doc.path)
    ]
    return lsp.CompletionList(is_incomplete=False, items=items)


@server.feature(lsp.TEXT_DOCUMENT_HOVER)
def _hover(ls: LanguageServer, params: lsp.HoverParams) -> lsp.Hover | None:
    doc = ls.workspace.get_text_document(params.text_document.uri)
    found = features.hover(
        doc.source, doc.offset_at_position(params.position), doc.path
    )
    if found is None:
        return None
    return lsp.Hover(
        contents=_markdown(found.markdown), range=_range(doc, found.start, found.end)
    )


def main() -> None:
    server.start_io()


if __name__ == "__main__":
    main()
