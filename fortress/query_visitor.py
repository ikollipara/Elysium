"""
fortress/query_visitor.py
Ian Kollipara
2022.05.02

Query Visitor Class Definition
"""

# Imports
from ast import (
    AST,
    Add,
    And,
    Attribute,
    BinOp,
    BitAnd,
    BitOr,
    BitXor,
    BoolOp,
    Call,
    Compare,
    Constant,
    Div,
    Eq,
    FloorDiv,
    Gt,
    GtE,
    Invert,
    In,
    NotIn,
    LShift,
    Lambda,
    Lt,
    LtE,
    MatMult,
    Mod,
    Mult,
    Name,
    NodeVisitor,
    Not,
    NotEq,
    Pow,
    RShift,
    Slice,
    Sub,
    Subscript,
    Tuple,
    UAdd,
    USub,
    UnaryOp,
    keyword,
)
from typing import Any, Callable, Dict


class QueryVisitor(NodeVisitor):

    """Query Visitor

    This class subclasses ast.NodeVisitor. Its purpose is to
    walk the newly created AST and create the Deta Base query
    from it.
    """

    OP_TABLE = {Eq: "", NotEq: "?ne", Lt: "?lt", LtE: "?lte", Gt: "?gt", GtE: "?gte"}
    UOP_TABLE = {UAdd: "+", USub: "-", Invert: "~", Not: "not"}
    BOP_TABLE = {
        Pow: "**",
        Mult: "*",
        MatMult: "@",
        FloorDiv: "//",
        Div: "/",
        Mod: "%",
        Add: "+",
        Sub: "-",
        LShift: "<<",
        RShift: ">>",
        BitAnd: "&",
        BitOr: "|",
        BitXor: "^",
    }

    def __init__(
        self, query_ast: AST, deta_base_name: str, globals: Dict[str, Any]
    ) -> None:
        self.db_name = deta_base_name
        self.globals = globals
        self.ast = query_ast
        self.param = ""

    def generate_query(self) -> str:
        return self.visit(self.ast)

    def visit_Lambda(self, node: Lambda) -> Any:
        """visit the Lambda Node.

        This function visits the lambda node, and
        takes note of the function parameters, which
        is replaced by self.db_name.
        """

        self.param = node.args.args[0].arg
        return f"{{{self.visit(node.body)}}}"

    def visit_BoolOp(self, node: BoolOp) -> Any:
        """Visit the BoolOp Node.

        This function visits the bool operation node,
        and will generate the different query based on
        whether the function is an AND or an OR.
        """

        if isinstance(node.op, And):
            queries = [f"{self.visit(child)}" for child in node.values]

        else:
            queries = [f"{{{self.visit(child)}}}" for child in node.values]

        return ", ".join(queries)

    def visit_Attribute(self, node: Attribute) -> Any:
        """Visit the Attribute Node.

        Attributes recurse to their children, thus
        creating the correct string for object's values.
        """

        obj = self.visit(node.value)

        if not isinstance(obj, str):
            return getattr(obj, node.attr)

        return f"{obj}.{node.attr}"

    def visit_Name(self, node: Name) -> Any:
        """Visit the Name Node.

        This node can be two things. If the
        value is the param then just return
        the id value. Otherwise attempt to resolve
        the value through the globals dict.
        """

        if node.id == self.param:
            return self.db_name

        elif resolved_val := self.globals.get(node.id):
            if isinstance(resolved_val, str):
                return f"'{resolved_val}'"

            return resolved_val

        else:
            return node.id

    def visit_Constant(self, node: Constant) -> Any:
        """Visit the Constant Node."""

        if isinstance(node.value, str):
            return f"'{node.value}'"
        return node.value

    def visit_Compare(self, node: Compare) -> Any:
        """Visit the Compare Node."""

        if isinstance(node.left, Subscript):
            return self.handle_prefix(node)

        elif isinstance(node.ops[0], (In, NotIn)):
            return self.handle_contains(node)

        else:
            operator = self.OP_TABLE[node.ops[0].__class__]  # type: ignore
            lhs = self.visit(node.left)
            rhs = self.visit(node.comparators[0])
            return f"'{lhs}{operator}': {rhs}"

    def handle_prefix(self, node: Compare):
        """Handle the issues that occur with prefix query.

        To do a prefix query, the list access syntax may
        be used.
        """

        value = self.visit(node.left.value)  # type: ignore
        compare = self.visit(node.comparators[0])

        return f"'{value}?pfx': {compare}"

    def handle_contains(self, node: Compare):
        """Handle issues with contains in Deta.

        Given that deta cannot do `x.name in [1,2,3]`, the
        adjusted query is created with an OR query. Unless
        the the IN is against a string, which works fine.
        """

        lhs = self.visit(node.left)
        rhs = self.visit(node.comparators[0])

        if isinstance(rhs, range):
            return f"'{lhs}?r': [{rhs.start},{rhs.stop}]"

        if self.db_name in rhs:
            if isinstance(node.ops[0], In):
                return f"'{rhs}?contains': {lhs}"

            return f"'{rhs}?!contains': {lhs}"

        else:
            if isinstance(node.ops[0], In):
                orred_queries = [f"{{'{lhs}': {val}}}" for val in rhs]
            else:
                orred_queries = [f"'{lhs}?ne': {val}" for val in rhs]

            return ", ".join(orred_queries)

    def visit_Call(self, node: Call) -> Any:
        """Visit the Call Node.

        This node represents function calls. Since
        Deta only supports two specific function
        calls: range and previous, those will be
        turned into their correct queries, the
        others will be executed and the evaluated
        result is used.
        """

        func = self.visit(node.func)

        keyword_pairs = [self.visit(keyword) for keyword in node.keywords]
        keywords = {key: value for key, value in keyword_pairs}
        args = [self.visit(arg) for arg in node.args]

        if isinstance(func, Callable):
            return func(*args, **keywords)  # type: ignore
        elif isinstance(func, str):
            if resolved_func := getattr(self.globals["__builtins__"], func, None):
                return resolved_func(*args, **keywords)
            else:
                raise NotImplementedError("Method calls on table class not supported")

    def visit_Subscript(self, node: Subscript) -> Any:
        """Visit the Subscript Node.

        This node handles cases like dictionary access
        and list slicing.
        """

        value = self.visit(node.value)

        match node.slice:

            case Slice(lower, upper, step):
                return eval(f"{value}[{lower}:{upper}:{step}]")

            case _:
                slice = self.visit(node.slice)
                return eval(f"{value}[{slice}]")

    def visit_Slice(self, node: Slice) -> Any:
        """Visit the Slice node.

        This handles the case of list slicing.
        """

        return node.lower, node.upper, node.step

    def visit_keyword(self, node: keyword) -> Any:
        """Visit the keyword Node.

        This takes keywords, and creates tuples of (key, value)
        from.
        """

        return (node.arg, self.visit(node.value))

    def visit_Tuple(self, node: Tuple) -> Any:
        """Visit the Tuple Node.

        This builds a tuple from the elements,
        and evaluates it into an actual tuple.
        """

        return [self.visit(el) for el in node.elts]

    def visit_UnaryOp(self, node: UnaryOp) -> Any:
        """Visit the UnaryOp Node.

        Given the list of unary operators, and
        the fact there's not correlation to the queries,
        it's vital that those operators are evaluated
        against their operand.
        """

        if not isinstance(node.op, Not):
            return eval(f"{self.UOP_TABLE[node.op.__class__]}{self.visit(node.operand)}")  # type: ignore
        else:
            return eval(f"not {self.visit(node.operand)}")

    def visit_BinOp(self, node: BinOp) -> Any:
        """Visit the BinOp Node.

        Given that list of binary operators,
        and the fact there's no direct correlation to
        queries, these values must be evaluated.
        """

        operator = self.BOP_TABLE[node.op.__class__]  # type: ignore
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        return eval(f"{lhs} {operator} {rhs}")
