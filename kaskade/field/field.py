from __future__ import annotations
from numpy import dtype
from kaskade.pyir.typing import DType
from kaskade.pyir import graph
from .grid import Grid
from . import global_ctx as ctx
from ..pyir.node import Node, BinOpNode
from typing import Union, List
from ..pyir.utils import BinaryOpr, biopr_map
from ..pyir import functions as node_functions


class Field():
    """ Physical field defined on a spercific grid
    """

    def __init__(self, grid: Grid, name=None) -> None:
        super().__init__()
        self.size = grid.size
        self.name = name
        self.grid = grid
        self.blob = None
        self.data = None

# Basic mathematical operations
    def __add__(self, other: Union[Field, int, float, complex]) -> Field:
        return BinOpField(BinaryOpr.ADD, self, other)

    def __radd__(self, other: Union[Field, int, float, complex]) -> Field:
        return BinOpField(BinaryOpr.ADD, other, self)

    def __sub__(self, other: Union[Field, int, float, complex]) -> Field:
        return BinOpField(BinaryOpr.SUB, self, other)

    def __rsub__(self, other: Union[Field, int, float, complex]) -> Field:
        return BinOpField(BinaryOpr.SUB, other, self)

    def __mul__(self, other: Union[Field, int, float, complex]) -> Field:
        return BinOpField(BinaryOpr.MUL, self, other)

    def __rmul__(self, other: Union[Field, int, float, complex]) -> Field:
        return BinOpField(BinaryOpr.MUL, other, self)

    def __floordiv__(self, other: Union[Field, int, float, complex]) -> Field:
        return BinOpField(BinaryOpr.IDIV, self, other)

    def __rfloordiv__(self, other: Union[Field, int, float, complex]) -> Field:
        return BinOpField(BinaryOpr.IDIV, other, self)

    def __truediv__(self, other: Union[Field, int, float, complex]) -> Field:
        return BinOpField(BinaryOpr.FDIV, self, other)

    def __rtruediv__(self, other: Union[Field, int, float, complex]) -> Field:
        return BinOpField(BinaryOpr.FDIV, other, self)


def make_coord_field(grid: Grid) -> List[Field]:
    ret = []
    for coord_location in grid.grid_loc:
        ret.append(InputField(grid, name=None, bound_node=coord_location))

    return ret


def _eval(outputs: List[Field]):
    gg = ctx._get_graph()
    inputs = ctx._get_inputs()
    output_node = []
    for field in outputs:
        output_node.append(field.blob)
    output_data = gg.compile_and_run(inputs=inputs, outputs=output_node)
    for idx, field in enumerate(outputs):
        field.data = output_data[idx]


class BinOpField(Field):
    """ Field defined generated from binary operations
    """

    def __init__(self, opr: BinaryOpr, LHS: Field, RHS: Field) -> None:
        if (LHS.grid == RHS.grid):
            third_grid = LHS.grid
        else:
            grid_map = ctx._get_map(LHS.grid, RHS.grid)
            third_grid = grid_map[0]
        super().__init__(third_grid,
                         name=f"field_{biopr_map[opr][DType.Double]}")
        self.ind_node1 = grid_map[1]
        self.ind_node2 = grid_map[2]
        self.ind_node3 = grid_map[3]
        self.blob = BinOpNode(
            opr, LHS.blob[self.ind_node1], RHS.blob[self.ind_node2])


class InputField(Field):
    """ Inputs field with a inputNode blob
    """

    def __init__(self, grid: Grid, name: str = None, bound_node: Node = None) -> None:
        name = f"Input{name}"
        super().__init__(grid, name=name)
        if not bound_node:
            gg = ctx._get_graph()
            self.blob = gg.new(grid.size, name, DType.Double)
        else:
            self.blob = bound_node


def sin(f: Field):
    sin_node = node_functions.sin(f.blob)
    ret_field = Field(f.grid)
    ret_field.blob = sin_node
    return ret_field
