"""
fortress/fortress.py
Ian Kollipara
2022.05.02

Fortress Class
"""

# Imports


from dataclasses import dataclass, is_dataclass
from types import LambdaType
from typing_extensions import Self
from deta import _Base, Deta
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Type

from .fetch_response import FetchResponse
from .decompile import Decompiler


class Fortress:

    _objects: Dict[str, Type["Fortress"]] = {}
    _bases: Dict[str, _Base] = {}


    def __init_subclass__(cls) -> None:
        cls._objects |= {cls.__name__: cls}
    

    @classmethod
    def create_mappings(cls, deta: Optional[Deta] = None):
        if not deta:
            deta = Deta()
        for base_name in cls._objects.keys():
            cls._bases |= {base_name: deta.Base(base_name)}
    

    @classmethod
    def fetch(cls, query: Optional[Callable[[Self], bool]] = None, limit: int = 1000) -> FetchResponse[Self]:

        if query:
            decompiler = Decompiler(query)
            decompiler.decompile()
            query_ast = decompiler.decompiled_code

Fortress.fetch(lambda x: x in [1,2,3])