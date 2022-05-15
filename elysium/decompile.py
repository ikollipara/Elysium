"""
decompile.py
Ian Kollipara
2022.04.25

Python Bytecode to AST
"""

# Imports
import ast
import dis
import types
from typing import Any, Callable, Dict, Iterable, List, Optional, TypeVar
from collections.abc import Container
from .query_visitor import QueryVisitor

T = TypeVar("T")


class _Decompiler:
    """_Decompiler is the internal decompiler class.

    _Decompiler is used to take the lambda function queries
    and decompile their bytecode to an AST. This is done to
    preserve strong type checking and a more pythonic way of
    writing queries. This decompiler is designed exclusively
    for use in Fortress, meaning it only decompiles statements
    that fortress cares about: lambdas that result in booleans.

    This class should never be instantiated by the end user.
    """

    compare_operators = {
        "==": ast.Eq(),
        ">": ast.Gt(),
        ">=": ast.GtE(),
        "<=": ast.LtE(),
        "<": ast.Lt(),
        "!=": ast.NotEq(),
    }

    def __init__(self, lambda_func: Callable[[Any], bool]) -> None:
        self.function = lambda_func
        self.bytecode = dis.Bytecode(lambda_func)
        self.stack: List[ast.AST] = []
        self.jumps: List[dis.Instruction] = []
        self.ast: ast.AST = ast.Constant(value=None)
        self.lambda_param = ""

    def __handle_jumps(self, instruction: dis.Instruction):
        """Handle Jump Cases in the Bytecode.

        Given the instance of a jump command, evaluate whether
        the jump occurred due to an And or Or statement.
        """

        jump_argvals = (jump.argval for jump in self.jumps)

        # In the bytecode if a statement is a jump statement, the
        # line to jump to is stored as the argval in the instruction.
        # Thus, to check if a statement is a jump to statement and
        # adjust the stack accordingly, we check if the instruction
        # offset is equal to any jump argvals.
        if instruction.offset in jump_argvals:
            jump: dis.Instruction = next(
                filter(lambda jump: instruction.offset == jump.argval, self.jumps)
            )
            operator = (
                ast.Or()
                if jump.opname in ("JUMP_IF_TRUE_OR_POP", "POP_JUMP_IF_FALSE")
                else ast.And()
            )

            rhs = self.stack.pop()
            lhs = self.stack.pop()
            self.stack.append(ast.BoolOp(op=operator, values=[lhs, rhs]))
            self.jumps.remove(jump)

    def decompile(self):
        """Decompile the provided function to its AST.

        This is the main function of the _Decompiler class. This
        reads each instruction and generates the AST node from it.
        All nodes are stored on the stack, and the resulting AST
        is stored under self.ast.
        """

        for instruction in self.bytecode:

            self.__handle_jumps(instruction)

            try:
                method = getattr(self, instruction.opname.lower())

                method(instruction)
            except AttributeError as e:
                raise NotImplementedError("Unsupported Bytecode") from e

        body = self.stack.pop()
        lambda_node = ast.Lambda(
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg=self.lambda_param)],
                kwonlyargs=[],
                kwdefault=[],
                default=[],
            ),
            body=body,
            type_ignore=[],
        )

        self.ast = lambda_node
        return lambda_node

    def return_value(self, instruction: dis.Instruction):
        """Decompile the RETURN_VALUE instruction.

        This instruction doesn't matter as there's need to know
        what the function returns, that's inferred from the AST.
        """

        pass

    def load_attr(self, instruction: dis.Instruction):
        """Decompile the LOAD_ATTR instruction."""

        value = self.stack.pop()
        attr = ast.Attribute(value=value, attr=instruction.argval, ctx=ast.Load())
        self.stack.append(attr)

    def load_fast(self, instruction: dis.Instruction):
        """Decompile the LOAD_FAST instruction.

        Because LOAD_FAST is only used on the
        lambda's parameter, the lambda parameter
        can be inferred from this instruction's argval.
        """

        self.lambda_param = instruction.argval
        self.stack.append(ast.Name(id=instruction.argval, ctx=ast.Load()))

    def load_global(self, instruction: dis.Instruction):
        """Decompile the LOAD_GLOBAL instruction."""

        self.stack.append(ast.Name(id=instruction.argval, ctx=ast.Load()))

    def load_const(self, instruction: dis.Instruction):
        """Decompile the LOAD_CONST instruction."""

        if (
            not isinstance(instruction.argval, str)
            and isinstance(instruction.argval, Container)
            and isinstance(instruction.argval, Iterable)
        ):
            self.stack.append(
                ast.Tuple(elts=[ast.Constant(value=arg) for arg in instruction.argval])
            )
        else:
            self.stack.append(ast.Constant(value=instruction.argval))

    def call_function(self, instruction: dis.Instruction):
        """Decompile the CALL_FUNCTION instruction."""

        args = [self.stack.pop() for _ in range(instruction.argval)]
        func = self.stack.pop()

        call_op = ast.Call(func=func, args=args, keywords=[])
        self.stack.append(call_op)

    def call_function_kw(self, instruction: dis.Instruction):
        """Decompile the CALL_FUNCTION_KW instruction."""

        arg_names = [node.value for node in self.stack.pop().elts]  # type: ignore
        args = [self.stack.pop() for _ in range(instruction.argval)]
        func = self.stack.pop()

        keywords = [
            ast.keyword(arg=arg, value=value) for arg, value in zip(arg_names, args)
        ]

        try:
            unnamed_args = args[len(keywords) :]
        except IndexError:
            unnamed_args = []

        call_op = ast.Call(func=func, args=unnamed_args, keywords=keywords)

        self.stack.append(call_op)

    def build_slice(self, instruction: dis.Instruction):
        """Decompile the BUILD_SLICE instruction."""

        if instruction.argval == 3:
            step = self.stack.pop()
            upper = self.stack.pop()
            lower = self.stack.pop()

            slice_op = ast.Slice(lower=lower, upper=upper, step=step)
        else:
            upper = self.stack.pop()
            lower = self.stack.pop()

            slice_op = ast.Slice(lower=lower, upper=upper)

        self.stack.append(slice_op)

    def load_method(self, instruction: dis.Instruction):
        """Decompile the LOAD_METHOD instruction."""

        value = self.stack.pop()
        attr_op = ast.Attribute(value=value, attr=instruction.argval, ctx=ast.Load())
        self.stack.append(attr_op)

    def call_method(self, instruction: dis.Instruction):
        """Decompile the CALL_METHOD instruction."""

        args = [self.stack.pop() for _ in range(instruction.argval)]
        func = self.stack.pop()

        call_op = ast.Call(func=func, args=args, keywords=[])
        self.stack.append(call_op)

    def unary_positive(self, instruction: dis.Instruction):
        """Decompile the UNARY_POSITIVE instruction."""

        operand = self.stack.pop()
        unary_positive = ast.UnaryOp(op=ast.UAdd(), operand=operand, type_ignores=[])
        self.stack.append(unary_positive)

    def unary_negative(self, instruction: dis.Instruction):
        """Decompile the UNARY_POSITIVE instruction."""

        operand = self.stack.pop()
        unary_negative = ast.UnaryOp(op=ast.USub(), operand=operand, type_ignores=[])
        self.stack.append(unary_negative)

    def unary_not(self, instruction: dis.Instruction):
        """Decompile the UNARY_NOT instruction."""

        operand = self.stack.pop()
        unary_not = ast.UnaryOp(op=ast.Not(), operand=operand, type_ignores=[])
        self.stack.append(unary_not)

    def unary_invert(self, instruction: dis.Instruction):
        """Decompile the UNARY_NOT instruction."""

        operand = self.stack.pop()
        unary_invert = ast.UnaryOp(op=ast.Invert(), operand=operand, type_ignores=[])
        self.stack.append(unary_invert)

    def binary_power(self, instruction: dis.Instruction):
        """Decompile the BINARY_POWER instruction."""

        rhs = self.stack.pop()
        lhs = self.stack.pop()

        pow_op = ast.BinOp(left=lhs, op=ast.Pow(), right=rhs)
        self.stack.append(pow_op)

    def binary_multiply(self, instruction: dis.Instruction):
        """Decompile the BINARY_MULTIPLY instruction."""

        rhs = self.stack.pop()
        lhs = self.stack.pop()

        mult_op = ast.BinOp(left=lhs, op=ast.Mult(), right=rhs)
        self.stack.append(mult_op)

    def binary_matrix_multiply(self, instruction: dis.Instruction):
        """Decompile the BINARY_MATRIX_MULTIPLY instruction."""

        rhs = self.stack.pop()
        lhs = self.stack.pop()

        mat_mult_op = ast.BinOp(left=lhs, op=ast.MatMult(), right=rhs)
        self.stack.append(mat_mult_op)

    def binary_floor_divide(self, instruction: dis.Instruction):
        """Decompile the BINARY_FLOOR_DIVIDE instruction."""

        rhs = self.stack.pop()
        lhs = self.stack.pop()

        floor_div_op = ast.BinOp(left=lhs, op=ast.FloorDiv(), right=rhs)
        self.stack.append(floor_div_op)

    def binary_true_divide(self, instruction: dis.Instruction):
        """Decompile the BINARY_TRUE_DIVIDE instruction."""

        rhs = self.stack.pop()
        lhs = self.stack.pop()

        true_div_op = ast.BinOp(left=lhs, op=ast.Div(), right=rhs)
        self.stack.append(true_div_op)

    def binary_modulo(self, instruction: dis.Instruction):
        """Decompile the BINARY_MODULO instruction."""

        rhs = self.stack.pop()
        lhs = self.stack.pop()

        mod_op = ast.BinOp(left=lhs, op=ast.Mod(), right=rhs)
        self.stack.append(mod_op)

    def binary_add(self, instruction: dis.Instruction):
        """Decompile the BINARY_ADD instruction."""

        rhs = self.stack.pop()
        lhs = self.stack.pop()

        add_op = ast.BinOp(left=lhs, op=ast.Add(), right=rhs)
        self.stack.append(add_op)

    def binary_subtract(self, instruction: dis.Instruction):
        """Decompile the BINARY_SUBTRACT instruction."""

        rhs = self.stack.pop()
        lhs = self.stack.pop()

        sub_op = ast.BinOp(left=lhs, op=ast.Sub(), right=rhs)
        self.stack.append(sub_op)

    def binary_subscr(self, instruction: dis.Instruction):
        """Decompile the BINARY_SUBSCR instruction."""

        slice = self.stack.pop()
        value = self.stack.pop()

        subscr_op = ast.Subscript(value=value, slice=slice)
        self.stack.append(subscr_op)

    def binary_lshift(self, instruction: dis.Instruction):
        """Decompile the BINARY_LSHIFT instruction."""

        rhs = self.stack.pop()
        lhs = self.stack.pop()

        lshift_op = ast.BinOp(left=lhs, op=ast.LShift(), right=rhs)
        self.stack.append(lshift_op)

    def binary_rshift(self, instruction: dis.Instruction):
        """Decompile the BINARY_RSHIFT instruction."""

        lhs = self.stack.pop()
        rhs = self.stack.pop()

        rshift_op = ast.BinOp(left=lhs, op=ast.RShift(), right=rhs)
        self.stack.append(rshift_op)

    def binary_and(self, instruction: dis.Instruction):
        """Decompile the BINARY_ADD instruction."""

        lhs = self.stack.pop()
        rhs = self.stack.pop()

        bit_and_op = ast.BinOp(left=lhs, op=ast.BitAnd(), right=rhs)
        self.stack.append(bit_and_op)

    def binary_xor(self, instruction: dis.Instruction):
        """Decompile the BINARY_XOR instruction."""

        lhs = self.stack.pop()
        rhs = self.stack.pop()

        bit_xor_op = ast.BinOp(left=lhs, op=ast.BitXor(), right=rhs)
        self.stack.append(bit_xor_op)

    def binary_or(self, instruction: dis.Instruction):
        """Decompile the BINARY_OR instruction."""

        lhs = self.stack.pop()
        rhs = self.stack.pop()

        bit_or_op = ast.BinOp(left=lhs, op=ast.BitOr(), right=rhs)
        self.stack.append(bit_or_op)

    def build_tuple(self, instruction: dis.Instruction):
        """Decompile the BUILD_TUPLE instruction."""

        tuple_op = ast.Tuple(elts=[self.stack.pop() for _ in range(instruction.argval)])
        self.stack.append(tuple_op)

    def build_map(self, instruction: dis.Instruction):
        """Decompile the BUILD_MAP instruction."""

        # Pairs are stored as Value, Key in the tree.
        pairs = [
            (self.stack.pop(), self.stack.pop()) for _ in range(instruction.argval)
        ]
        keys = [p[1] for p in pairs]
        values = [p[0] for p in pairs]

        map_op = ast.Dict(keys=keys, values=values)
        self.stack.append(map_op)

    def build_const_key_map(self, instruction: dis.Instruction):
        """Decompile the BUILD_CONST_KEY_MAP instruction."""

        keys = self.stack.pop().elts  # type: ignore
        values = [self.stack.pop() for _ in range(instruction.argval)]

        map_op = ast.Dict(keys=keys, values=values)
        self.stack.append(map_op)

    def is_op(self, instruction: dis.Instruction):
        """Decompile the IS_OP instruction."""

        rhs = self.stack.pop()
        lhs = self.stack.pop()

        operator = ast.IsNot() if instruction.argval else ast.Is()

        compare_op = ast.Compare(left=lhs, ops=[operator], comparators=[rhs])

        self.stack.append(compare_op)

    def compare_op(self, instruction: dis.Instruction):
        """Decompile the COMPARE_OP instruction.

        This handles all comparision cases, which is why
        the class attribute compare_operators exists.
        """

        rhs = self.stack.pop()
        lhs = self.stack.pop()

        compare_op = ast.Compare(
            left=lhs,
            ops=[self.compare_operators[instruction.argval]],
            comparators=[rhs],
        )

        self.stack.append(compare_op)

    def contains_op(self, instruction: dis.Instruction):
        """Decompile the CONTAINS_OP instruction.

        This handles uses of `in` and `not in`.
        """

        rhs = self.stack.pop()
        lhs = self.stack.pop()

        # If the instruction is "not in" then the
        # instruction's argval is set to 1, that is True.
        operator = ast.NotIn() if instruction.argval else ast.In()

        # Contains is really just specialized comparision
        compare_op = ast.Compare(left=lhs, ops=[operator], comparators=[rhs])
        self.stack.append(compare_op)

    def jump_if_false_or_pop(self, instruction: dis.Instruction):
        """Decompile the JUMP_IF_FALSE_OR_POP instruction.

        Since this handles a jump, its actual understanding
        is handled within self.__handle_jump.
        """

        self.jumps.append(instruction)

    def jump_if_true_or_pop(self, instruction: dis.Instruction):
        """Decompile the JUMP_IF_TRUE_OR_POP instruction.

        Since this handles a jump, its actual understanding
        is handled within self.__handle_jump.
        """

        self.jumps.append(instruction)

    def pop_jump_if_true(self, instruction: dis.Instruction):
        """Decompile the POP_JUMP_IF_TRUE instruction.

        Since this handles a jump, its actual understanding
        is handled within self.__handle_jump.
        """

        self.jumps.append(instruction)

    def pop_jump_if_false(self, instruction: dis.Instruction):
        """Decompile the POP_JUMP_IF_FALSE instruction.

        Since this handles a jump, its actual understanding
        is handled within self.__handle_jump.
        """

        self.jumps.append(instruction)


def decompile(lambda_func: types.LambdaType) -> ast.AST:
    """Decompile the given lambda function into its AST>

    This creates a _Decompiler object and invokes the decompile
    method. This is simply a wrapper.
    """

    return _Decompiler(lambda_func).decompile()
