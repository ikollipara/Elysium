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


class Elysium:

    """ Elysium

    Elysium is a Deta ORM. To create a model, subclass the Fortress.Model
    and use the dataclass decorator. This allows for model querying using
    the lambda syntax.
    """

    _models: List[Type["Elysium.Model"]] = []
    _bases: Dict[Type["Elysium.Model"], _Base] = {}

    def __init__(self, deta: Optional[Deta] = None) -> None:
        self.deta = deta or Deta()


    def create_mappings(self):
        """Generate bases for registered models."""

        self._bases = {model: self.deta.Base(model.__name__) for model in self._models}


    @classmethod
    def insert(cls, item: "Elysium.Model" | List["Fortress.Model"]):
        """Insert one or more items into the deta base."""

        base = cls._bases[cls] # type: ignore

        if isinstance(item, list):
            item_dicts = [i.__dict__ for i in item]
            base.put_many(item_dicts) # type: ignore

        else:
            item_dict = item.__dict__.copy()
            key = item_dict.pop("key")
            base.put(item_dict, key)


    @dataclass
    class Model:

        """ Elysium Model

        This is the base class for all Elysium Models. It defines
        how a model is implemented and stored in Elysium.

        To use simply subclass it as Elysium.Model. Ideally the
        model would also be a dataclass. When subclassing, you
        may provide an alternative table name, as well as a callable
        to generate keys from. These keys should be no argument functions
        that result in a string.
        """

        __camel_to_snake = re_compile(r"(?<!^)(?=[A-Z])")
        __table_name__: ClassVar[str]
        __key_func: ClassVar[Callable[[Self], str]]


        def __init_subclass__(cls, table_name: Optional[str] = None, key_generator: Optional[Callable[[], str]] = None) -> None:
            cls.__table_name__ = table_name or cls.__camel_to_snake.sub("_", cls.__name__).lower()
            cls.__key_func = (lambda x: key_generator()) if key_generator else (lambda x: str(uuid4()))

            Elysium._models.append(cls)

        def __post_init__(self):
            self.key = self.__key_func() # type: ignore


        @classmethod
        def fetch(cls, query: Optional[Callable[[Self], bool]] = None, *, limit: int = 1000, offset: int = 0) -> List[Self]:
            """ Fetch a collection of models from the deta base.

            Fetching can be filtered by writing a lambda function to serve as the query. This
            function should take only the class as a parameter, and return a boolean. The amount
            of results can be limited by the limit parameter. You may also offset your results
            through an offset parameter.
            """

            query_str = None if not query else eval(QueryVisitor(decompile(query), cls.__table_name__, query.__globals__).generate_query()) # type: ignore

            items: List[Self] = []

            base = Elysium._bases[cls]
            response = base.fetch(query_str, limit=1000)
            items += [cls(**item) for item in response.items]

            if limit > 1000:
                leftover = limit - 1000

                while leftover <= 0 or response.last is not None:
                    response = base.fetch(query_str, limit=1000)
                    items += [cls(**item) for item in response.items]
                    leftover -= 1000

            return items

        @classmethod
        def get(cls, key: str) -> Self | None:
            """Get an item by its key."""

            if base := Elysium._bases.get(cls):
                if item := base.get(key):
                    return cls(**item)

        @classmethod
        def delete(cls, key: str):
            """Delete the item at the given key."""

            Elysium._bases[cls].delete(key)

        @classmethod
        def update(cls, updated_item: Self):
            """Update the item with the new values."""

            base = Elysium._bases[cls]
            update_dict = updated_item.__dict__.copy()
            key = update_dict.pop("key")
            base.put(update_dict, key)
