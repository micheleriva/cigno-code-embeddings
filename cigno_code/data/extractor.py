"""
Tree-sitter based symbol extractor.

Extracts functions, classes, methods, and other symbols from source code
using tree-sitter grammars. Mirrors the extraction logic in Cigno's
@cigno/extractor package.
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING

import tree_sitter

from .languages import LanguageConfig, LANGUAGE_CONFIG
from .text_formatter import SymbolForEmbedding

if TYPE_CHECKING:
    from tree_sitter import Node

logger = logging.getLogger(__name__)

# Cache loaded languages
_language_cache: dict[str, tree_sitter.Language] = {}
_parser: tree_sitter.Parser | None = None


def _get_parser() -> tree_sitter.Parser:
    global _parser
    if _parser is None:
        _parser = tree_sitter.Parser()
    return _parser


def _get_language(config: LanguageConfig) -> tree_sitter.Language:
    """Load and cache a tree-sitter language grammar."""
    if config.name not in _language_cache:
        module = importlib.import_module(config.grammar_module)
        grammar_fn = getattr(module, config.grammar_func)
        lang = tree_sitter.Language(grammar_fn())
        _language_cache[config.name] = lang
    return _language_cache[config.name]


def _get_node_text(node: Node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _find_name(node: Node, config: LanguageConfig, source: bytes) -> str | None:
    """Extract the symbol name from an AST node."""
    # Try the configured name field first
    name_node = node.child_by_field_name(config.name_field)
    if name_node:
        # For C/C++ declarators, dig into nested pointer/function declarators
        while name_node.type in ("pointer_declarator", "function_declarator"):
            inner = name_node.child_by_field_name("declarator")
            if inner:
                name_node = inner
            else:
                break
        return _get_node_text(name_node, source)

    # Fallback: look for an identifier-like child
    for child in node.children:
        if child.type in ("identifier", "value_name", "simple_identifier", "object_reference"):
            return _get_node_text(child, source)

    return None


def _find_doc_comment(node: Node, source: bytes, config: LanguageConfig) -> str:
    """Find a doc comment immediately preceding a symbol node."""
    if config.doc_comment_node is None:
        return ""

    # For Python, look for a docstring as the first child expression_statement
    if config.name == "python":
        body = node.child_by_field_name("body")
        if body and body.child_count > 0:
            first = body.children[0]
            if first.type == "expression_statement" and first.child_count > 0:
                string_node = first.children[0]
                if string_node.type == "string":
                    text = _get_node_text(string_node, source)
                    # Strip triple quotes
                    for q in ('"""', "'''"):
                        if text.startswith(q) and text.endswith(q):
                            return text[3:-3].strip()
                    return text.strip('"').strip("'").strip()
        return ""

    # For other languages, look at the previous sibling
    prev = node.prev_named_sibling
    if prev is None:
        return ""

    text = _get_node_text(prev, source)

    # Check if it's a doc comment
    if config.name in ("java",) and text.startswith("/**"):
        # Javadoc: strip /** ... */
        return text[3:-2].strip().replace("\n * ", "\n").replace("\n *", "\n").strip()
    if config.name == "rust" and text.startswith("///"):
        return text.lstrip("/").strip()
    if text.startswith("//"):
        return text.lstrip("/").strip()
    if text.startswith("/*"):
        return text[2:-2].strip()

    return ""


def _build_signature(node: Node, source: bytes, config: LanguageConfig) -> str:
    """Build a function/method signature string."""
    params = node.child_by_field_name("parameters")
    if params is None:
        params = node.child_by_field_name("formal_parameters")

    name = _find_name(node, config, source) or ""

    if params:
        return f"{name}{_get_node_text(params, source)}"
    return name


def _classify_symbol_type(node_type: str, config: LanguageConfig) -> str:
    """Map a tree-sitter node type to a human-readable symbol type."""
    if node_type in config.function_nodes:
        if "method" in node_type:
            return "method"
        return "function"
    if node_type in config.class_nodes:
        if "interface" in node_type:
            return "interface"
        if "struct" in node_type:
            return "struct"
        if "enum" in node_type:
            return "enum"
        if "trait" in node_type:
            return "trait"
        if "impl" in node_type:
            return "impl"
        if "type_alias" in node_type:
            return "type"
        if "namespace" in node_type:
            return "namespace"
        if "union" in node_type:
            return "union"
        return "class"
    return "symbol"


def extract_symbols(
    source_code: str,
    language: str,
    file_path: str | None = None,
) -> list[SymbolForEmbedding]:
    """Extract symbols from source code using tree-sitter.

    Args:
        source_code: The full source code text.
        language: Language name (must be a key in LANGUAGE_CONFIG).
        file_path: Optional file path for context.

    Returns:
        List of extracted symbols ready for embedding.
    """
    # Normalize aliases to match LANGUAGE_CONFIG keys
    lang_aliases = {"c++": "cpp", "c#": "csharp", "c-sharp": "csharp", "bash": "shell", "objective-c": "objc"}
    lang_key = lang_aliases.get(language, language)
    config = LANGUAGE_CONFIG.get(lang_key)
    if config is None:
        logger.warning(f"Unsupported language: {language}")
        return []

    try:
        lang = _get_language(config)
    except (ImportError, AttributeError) as e:
        logger.warning(f"Failed to load grammar for {language}: {e}")
        return []

    parser = _get_parser()
    parser.language = lang

    source_bytes = source_code.encode("utf-8")
    tree = parser.parse(source_bytes)

    symbols: list[SymbolForEmbedding] = []
    target_types = set(config.all_symbol_nodes)

    # Iterative DFS to avoid RecursionError on deeply nested files
    stack: list[Node] = [tree.root_node]
    while stack:
        node = stack.pop()

        if node.type in target_types:
            name = _find_name(node, config, source_bytes)
            if name is not None and len(name) >= 2:
                body = _get_node_text(node, source_bytes)
                symbol_type = _classify_symbol_type(node.type, config)
                signature = _build_signature(node, source_bytes, config)
                doc_comment = _find_doc_comment(node, source_bytes, config)

                symbols.append(
                    SymbolForEmbedding(
                        symbol_name=name,
                        symbol_type=symbol_type,
                        signature=signature,
                        doc_comment=doc_comment,
                        body=body,
                        file_path=file_path,
                    )
                )

        # Push children in reverse order so leftmost is processed first
        stack.extend(reversed(node.children))

    return symbols
