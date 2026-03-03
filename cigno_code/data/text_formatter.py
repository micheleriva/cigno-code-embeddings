"""
Text formatter for code symbols — mirrors Cigno's text-formatter.ts exactly.

The training data must match the format the model will see at inference time
in the Cigno CLI, so this must stay in sync with:
  cerca/packages/embedder/src/text-formatter.ts
"""

from dataclasses import dataclass

MAX_BODY_CHARS = 512


@dataclass
class SymbolForEmbedding:
    symbol_name: str
    symbol_type: str  # "function", "class", "method", "interface", etc.
    signature: str
    doc_comment: str
    body: str
    file_path: str | None = None


def format_embedding_text(doc: SymbolForEmbedding) -> str:
    """Format a symbol into text suitable for embedding.

    Matches the output of Cigno's formatEmbeddingText() in text-formatter.ts:
      - file: <path>
      - <type>: <name>
      - signature: <sig>
      - documentation: <doc>
      - body: <body, truncated to 512 chars>
    """
    parts: list[str] = []

    if doc.file_path:
        parts.append(f"file: {doc.file_path}")

    parts.append(f"{doc.symbol_type}: {doc.symbol_name}")

    if doc.signature and doc.signature != doc.symbol_name:
        parts.append(f"signature: {doc.signature}")

    if doc.doc_comment:
        parts.append(f"documentation: {doc.doc_comment}")

    if doc.body:
        body = doc.body[:MAX_BODY_CHARS] if len(doc.body) > MAX_BODY_CHARS else doc.body
        parts.append(f"body: {body}")

    return "\n".join(parts)
