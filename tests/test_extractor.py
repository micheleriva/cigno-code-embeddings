"""Tests for tree-sitter symbol extraction."""

import pytest
from cigno_code.data.extractor import extract_symbols


def test_python_function():
    code = '''
def calculate_bmi(weight_kg: float, height_m: float) -> float:
    """Calculate Body Mass Index."""
    return weight_kg / (height_m ** 2)
'''
    symbols = extract_symbols(code, "python")
    assert len(symbols) == 1
    s = symbols[0]
    assert s.symbol_name == "calculate_bmi"
    assert s.symbol_type == "function"
    assert "weight_kg" in s.signature
    assert "Calculate Body Mass Index" in s.doc_comment


def test_python_class():
    code = '''
class BinarySearchTree:
    """A binary search tree implementation."""

    def insert(self, value: int) -> None:
        """Insert a value into the tree."""
        pass

    def search(self, value: int) -> bool:
        pass
'''
    symbols = extract_symbols(code, "python")
    names = {s.symbol_name for s in symbols}
    assert "BinarySearchTree" in names
    assert "insert" in names
    assert "search" in names


def test_go_function():
    code = '''
package main

// FetchUserByID retrieves a user from the database.
func FetchUserByID(ctx context.Context, id string) (*User, error) {
    return db.Find(id)
}
'''
    symbols = extract_symbols(code, "go")
    assert len(symbols) >= 1
    s = [s for s in symbols if s.symbol_name == "FetchUserByID"][0]
    assert s.symbol_type == "function"


def test_empty_source():
    symbols = extract_symbols("", "python")
    assert symbols == []


def test_unsupported_language():
    symbols = extract_symbols("some code", "brainfuck")
    assert symbols == []
