"""Tests for text_formatter — must match Cigno's text-formatter.ts output."""

from cigno_code.data.text_formatter import format_embedding_text, SymbolForEmbedding


def test_full_symbol():
    symbol = SymbolForEmbedding(
        symbol_name="getUserById",
        symbol_type="function",
        signature="getUserById(id: string)",
        doc_comment="Fetch a user by their unique ID.",
        body="function getUserById(id: string) { return db.users.find(id); }",
        file_path="src/users/repository.ts",
    )
    result = format_embedding_text(symbol)
    assert result == (
        "file: src/users/repository.ts\n"
        "function: getUserById\n"
        "signature: getUserById(id: string)\n"
        "documentation: Fetch a user by their unique ID.\n"
        "body: function getUserById(id: string) { return db.users.find(id); }"
    )


def test_no_file_path():
    symbol = SymbolForEmbedding(
        symbol_name="sort",
        symbol_type="function",
        signature="sort(arr: number[])",
        doc_comment="",
        body="function sort(arr) { return arr.sort(); }",
    )
    result = format_embedding_text(symbol)
    assert result.startswith("function: sort")
    assert "file:" not in result


def test_no_doc_comment():
    symbol = SymbolForEmbedding(
        symbol_name="AVLTree",
        symbol_type="class",
        signature="AVLTree",
        doc_comment="",
        body="class AVLTree { ... }",
    )
    result = format_embedding_text(symbol)
    assert "documentation:" not in result


def test_signature_same_as_name_excluded():
    symbol = SymbolForEmbedding(
        symbol_name="main",
        symbol_type="function",
        signature="main",
        doc_comment="",
        body="fn main() {}",
    )
    result = format_embedding_text(symbol)
    assert "signature:" not in result


def test_body_truncation():
    long_body = "x" * 1000
    symbol = SymbolForEmbedding(
        symbol_name="big",
        symbol_type="function",
        signature="big()",
        doc_comment="",
        body=long_body,
    )
    result = format_embedding_text(symbol)
    body_line = [line for line in result.split("\n") if line.startswith("body:")][0]
    # "body: " = 6 chars + 512 body chars
    assert len(body_line) == 6 + 512
