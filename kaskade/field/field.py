from numpy import dtype
from kaskade.pyir.typing import DType
from kaskade.pyir import graph
from .grid import Grid
from . import global_ctx as ctx
from ..pyir.node import Node, BinOpNode
from typing import Union, List
from ..pyir.utils import BinaryOpr, biopr_map


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

    def __add__(self, other):
        return BinOpField(BinaryOpr.ADD, self, other)


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
