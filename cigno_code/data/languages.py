"""
Language configuration for tree-sitter symbol extraction.

Maps languages to their tree-sitter grammars and the AST node types
that represent extractable symbols (functions, classes, methods, etc.).
"""

from dataclasses import dataclass, field


@dataclass
class LanguageConfig:
    """Configuration for extracting symbols from a single language."""

    name: str
    # Tree-sitter grammar module name (e.g., "tree_sitter_python")
    grammar_module: str
    # File extensions
    extensions: list[str]
    # AST node types that represent functions/methods
    function_nodes: list[str]
    # AST node types that represent classes/structs/interfaces
    class_nodes: list[str]
    # Node type for doc comments (language-specific)
    doc_comment_node: str | None = None
    # How to find the name child within a symbol node
    name_field: str = "name"
    # Function name to call on the grammar module (default: "language")
    # Some grammars expose language_typescript(), language_php(), etc.
    grammar_func: str = "language"

    @property
    def all_symbol_nodes(self) -> list[str]:
        return self.function_nodes + self.class_nodes


# ── Original 8 languages ──────────────────────────────────────────────

LANGUAGE_CONFIG: dict[str, LanguageConfig] = {
    "python": LanguageConfig(
        name="python",
        grammar_module="tree_sitter_python",
        extensions=[".py"],
        function_nodes=["function_definition"],
        class_nodes=["class_definition"],
        doc_comment_node="expression_statement",
        name_field="name",
    ),
    "typescript": LanguageConfig(
        name="typescript",
        grammar_module="tree_sitter_typescript",
        extensions=[".ts", ".tsx"],
        function_nodes=["function_declaration", "method_definition", "arrow_function"],
        class_nodes=["class_declaration", "interface_declaration", "type_alias_declaration"],
        doc_comment_node="comment",
        grammar_func="language_typescript",
    ),
    "javascript": LanguageConfig(
        name="javascript",
        grammar_module="tree_sitter_javascript",
        extensions=[".js", ".jsx"],
        function_nodes=["function_declaration", "method_definition", "arrow_function"],
        class_nodes=["class_declaration"],
        doc_comment_node="comment",
    ),
    "go": LanguageConfig(
        name="go",
        grammar_module="tree_sitter_go",
        extensions=[".go"],
        function_nodes=["function_declaration", "method_declaration"],
        class_nodes=["type_declaration"],
        doc_comment_node="comment",
    ),
    "rust": LanguageConfig(
        name="rust",
        grammar_module="tree_sitter_rust",
        extensions=[".rs"],
        function_nodes=["function_item"],
        class_nodes=["struct_item", "enum_item", "trait_item", "impl_item"],
        doc_comment_node="line_comment",
    ),
    "java": LanguageConfig(
        name="java",
        grammar_module="tree_sitter_java",
        extensions=[".java"],
        function_nodes=["method_declaration", "constructor_declaration"],
        class_nodes=["class_declaration", "interface_declaration", "enum_declaration"],
        doc_comment_node="block_comment",
    ),
    "c": LanguageConfig(
        name="c",
        grammar_module="tree_sitter_c",
        extensions=[".c", ".h"],
        function_nodes=["function_definition"],
        class_nodes=["struct_specifier", "enum_specifier", "union_specifier"],
        doc_comment_node="comment",
        name_field="declarator",
    ),
    "cpp": LanguageConfig(
        name="cpp",
        grammar_module="tree_sitter_cpp",
        extensions=[".cpp", ".hpp", ".cc", ".hh", ".cxx"],
        function_nodes=["function_definition"],
        class_nodes=["struct_specifier", "class_specifier", "enum_specifier", "namespace_definition"],
        doc_comment_node="comment",
        name_field="declarator",
    ),

    # ── 22 new languages ──────────────────────────────────────────────

    "ruby": LanguageConfig(
        name="ruby",
        grammar_module="tree_sitter_ruby",
        extensions=[".rb"],
        function_nodes=["method", "singleton_method"],
        class_nodes=["class", "module"],
        doc_comment_node="comment",
    ),
    "php": LanguageConfig(
        name="php",
        grammar_module="tree_sitter_php",
        extensions=[".php"],
        function_nodes=["function_definition", "method_declaration"],
        class_nodes=["class_declaration", "interface_declaration", "trait_declaration"],
        doc_comment_node="comment",
        grammar_func="language_php",
    ),
    "csharp": LanguageConfig(
        name="csharp",
        grammar_module="tree_sitter_c_sharp",
        extensions=[".cs"],
        function_nodes=["method_declaration", "constructor_declaration"],
        class_nodes=["class_declaration", "interface_declaration", "struct_declaration", "enum_declaration"],
        doc_comment_node="comment",
    ),
    "kotlin": LanguageConfig(
        name="kotlin",
        grammar_module="tree_sitter_kotlin",
        extensions=[".kt", ".kts"],
        function_nodes=["function_declaration"],
        class_nodes=["class_declaration", "object_declaration", "interface_declaration"],
        doc_comment_node="multiline_comment",
    ),
    "swift": LanguageConfig(
        name="swift",
        grammar_module="tree_sitter_swift",
        extensions=[".swift"],
        function_nodes=["function_declaration"],
        class_nodes=["class_declaration", "struct_declaration", "protocol_declaration", "enum_declaration"],
        doc_comment_node="comment",
    ),
    "scala": LanguageConfig(
        name="scala",
        grammar_module="tree_sitter_scala",
        extensions=[".scala", ".sc"],
        function_nodes=["function_definition", "function_declaration"],
        class_nodes=["class_definition", "object_definition", "trait_definition"],
        doc_comment_node="block_comment",
    ),
    "shell": LanguageConfig(
        name="shell",
        grammar_module="tree_sitter_bash",
        extensions=[".sh", ".bash"],
        function_nodes=["function_definition"],
        class_nodes=[],
        doc_comment_node="comment",
    ),
    "lua": LanguageConfig(
        name="lua",
        grammar_module="tree_sitter_lua",
        extensions=[".lua"],
        function_nodes=["function_declaration", "function_definition"],
        class_nodes=[],
        doc_comment_node="comment",
    ),
    "haskell": LanguageConfig(
        name="haskell",
        grammar_module="tree_sitter_haskell",
        extensions=[".hs"],
        function_nodes=["function", "bind"],
        class_nodes=["data_type", "newtype", "class", "instance"],
        doc_comment_node="comment",
    ),
    "julia": LanguageConfig(
        name="julia",
        grammar_module="tree_sitter_julia",
        extensions=[".jl"],
        function_nodes=["function_definition", "short_function_definition"],
        class_nodes=["struct_definition", "abstract_definition"],
        doc_comment_node="string_literal",
    ),
    "elixir": LanguageConfig(
        name="elixir",
        grammar_module="tree_sitter_elixir",
        extensions=[".ex", ".exs"],
        function_nodes=["call"],  # def/defp are call nodes in elixir grammar
        class_nodes=[],
        doc_comment_node="comment",
    ),
    "ocaml": LanguageConfig(
        name="ocaml",
        grammar_module="tree_sitter_ocaml",
        extensions=[".ml", ".mli"],
        function_nodes=["let_binding"],
        class_nodes=["type_definition", "module_binding"],
        doc_comment_node="comment",
        grammar_func="language_ocaml",
        name_field="name",
    ),
    "zig": LanguageConfig(
        name="zig",
        grammar_module="tree_sitter_zig",
        extensions=[".zig"],
        function_nodes=["function_declaration"],
        class_nodes=["struct_declaration"],
        doc_comment_node="doc_comment",
    ),
    "fortran": LanguageConfig(
        name="fortran",
        grammar_module="tree_sitter_fortran",
        extensions=[".f90", ".f95", ".f03", ".f08"],
        function_nodes=["function_statement", "subroutine_statement"],
        class_nodes=[],
        doc_comment_node="comment",
    ),
    "sql": LanguageConfig(
        name="sql",
        grammar_module="tree_sitter_sql",
        extensions=[".sql"],
        function_nodes=["create_view"],
        class_nodes=["create_table"],
        doc_comment_node="comment",
    ),
    "swift": LanguageConfig(
        name="swift",
        grammar_module="tree_sitter_swift",
        extensions=[".swift"],
        function_nodes=["function_declaration"],
        class_nodes=["class_declaration", "protocol_declaration"],
        doc_comment_node="comment",
        name_field="name",
    ),
    "powershell": LanguageConfig(
        name="powershell",
        grammar_module="tree_sitter_powershell",
        extensions=[".ps1", ".psm1"],
        function_nodes=["function_statement"],
        class_nodes=["class_statement", "enum_statement"],
        doc_comment_node="comment",
    ),
    "commonlisp": LanguageConfig(
        name="commonlisp",
        grammar_module="tree_sitter_commonlisp",
        extensions=[".lisp", ".cl", ".lsp"],
        function_nodes=["defun", "defmethod"],
        class_nodes=["defclass", "defstruct"],
        doc_comment_node="comment",
    ),
}


def get_language_for_extension(ext: str) -> LanguageConfig | None:
    """Look up the language config for a file extension."""
    for config in LANGUAGE_CONFIG.values():
        if ext in config.extensions:
            return config
    return None
