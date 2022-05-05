"""
fortress/fetch_response.py
Ian Kollipara
2022.05.02

Fetch Response Class
"""

# Imports
from typing import Generic, List, TypeVar, Generator
from dataclasses import dataclass

T = TypeVar("T")

@dataclass
class FetchResponse(Generic[T]):
    items: Generator[T, None, None]
    size: int