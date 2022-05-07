"""
fortress/fortress.py
Ian Kollipara
2022.05.05

Fortress Main Class
"""

# Imports
from dataclasses import dataclass, field
from functools import singledispatchmethod
from typing_extensions import Self
from uuid import uuid4
from deta import Deta, _Base
from typing import Any, Callable, ClassVar, Dict, List, Optional, Type, TypeVar
from .decompile import decompile
from .query_visitor import QueryVisitor
from re import compile as re_compile


def __parse_to_query(lambda_func: Callable[[Any], bool], base_name: str) -> str:
    """Parse the given lambda into a Deta Query.

    This function combines the most important modules of the
    project into a single function. It parses the lambda into
    a query that is then evaluated into a deta query dictionary.
    """

    return QueryVisitor(decompile(lambda_func), base_name, lambda_func.__globals__).generate_query()  # type: ignore


class Fortress:

    """Fortress Main class

    This is the main class. All models inherit from the
    model subclass, and all functionality for inserting, querying, etc.
    is handled within.
    """

    __models: List[Type["Fortress.Model"]] = []
    __bases: Dict[Type["Fortress.Model"], _Base] = {}

    def __init__(self, deta: Optional[Deta] = None) -> None:
        self.deta = deta or Deta()

    @dataclass
    class Model:

        __camel_to_snake = re_compile(r"(?<!^)(?=[A-Z])")
        __table_name__: ClassVar[str]
        __key_func: ClassVar[Callable[[], str]]

        @classmethod
        def __camel_to_snake_transformer(cls, camelcase: str) -> str:
            """Transform a camelcase string to snake case."""

            return cls.__camel_to_snake.sub("_", camelcase).lower()

        def __init_subclass__(
            cls,
            table_name: Optional[str] = None,
            key_func: Optional[Callable[[], str]] = None,
        ) -> None:
            table_name = table_name or cls.__camel_to_snake_transformer(cls.__name__)
            cls.__table_name__ = table_name
            cls.__key_func = key_func or (lambda: str(uuid4()))

            Fortress.__models.append(cls)

        def __post_init__(self):
            self._key = self.__key_func()

        @classmethod
        def fetch(
            cls, query: Optional[Callable[[Self], bool]] = None, *, limit: int = 1000
        ) -> List[Self]:
            """
            Fetch a set of objects from the deta based on the query. If no query is given,
            return all objects possible, or up to the limit (default is 1000).
            """

            query_str = (
                None if not query else eval(__parse_to_query(query, cls.__table_name__))
            )

            items: List[Self] = []

            base = Fortress.__bases[cls]
            res = base.fetch(query_str, limit=1000)
            items += map(self.__init__, res.items)  # type: ignore

            if limit > 1000:
                res = base.fetch(query_str, limit=1000)
                leftover = limit
                while res.last is not None:
                    leftover -= 1000
                    res = base.fetch(query_str, limit=leftover, last=res.last)
                    items += map(self.__init__, res.items)  # type: ignore

            return items

        @classmethod
        def get(cls, obj_key: str) -> Self | None:
            base = Fortress.__bases.get(cls)
            return cls(base.get(obj_key))  # type: ignore

    _T = TypeVar("_T", bound=Model)

    @singledispatchmethod
    @classmethod
    def insert(cls, item: _T | List[_T]):
        """Insert the given item(s) into the base."""

        if not isinstance(item, list):
            base = cls.__bases[type(item)]
            item_dict = item.__dict__
            key = item_dict.pop("_key")
            base.put(item.__dict__, key=key)
            item_dict |= {"_key": key}
        else:
            [cls.insert(i) for i in item]


    @classmethod
    def delete(cls, del_obj: _T):
        """Delete the object in the database."""

        base = cls.__bases[type(del_obj)]

        base.delete(del_obj._key)

    def generate_mappings(self):

        self.__bases = {model: self.deta.Base(model.__table_name__) for model in self.__models}
    