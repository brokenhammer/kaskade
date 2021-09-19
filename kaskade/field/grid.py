from ..pyir import graph
from ..pyir.node import Node
from . import global_ctx as ctx
from ..pyir.typing import DType
from functools import partial
from typing import List, Union
import numpy as np


class Grid():
    """ Grid describes the numerical grid points related to physical locations.
    """

    size : int
    dim : int
    grid_loc : Node

    def __init__(self, init_coord: List[Union[list, np.ndarray]]) -> None:
        self.size = len(init_coord[0])
        self.dim = len(init_coord)
        graph = ctx._get_graph()
        self.grid_loc = []
        self.grid_loc_data = init_coord
        for d in range(self.dim):
            assert(len(init_coord[d]) == self.size)
            self.grid_loc.append(graph.new(self.size, f"grid_loc_{d}", DType.Double))
        ctx._add_inputs(self.grid_loc, init_coord)


    def set_interpolate(self, func, **kwargs) -> None:
        self.interpolate_func = partial(func, kwargs)

    # def set_coord(self, coord_vals) -> None:
    #     ctx._add_inputs(self.grid_loc, coord_vals)


### Do not bother design a interpolation, it can be directly implemented with a pure python function with partial function capability
### Do not bother design a mapping function usign LLVM, just pass python arrays to memory.

def meshgrid(g1:Grid, g2:Grid) -> Grid:
    assert(g1.dim == 1 and g2.dim == 1 and g1.size == g2.size)
    ret_grid = Grid(g1.grid_loc_data + g2.grid_loc_data)

    ind1 = []
    ind2 = []
    ind3 = []
    for j in range(5):
        for i in range(5):
            ij=i*5+j
            ind1.append(i)
            ind2.append(j)
            ind3.append(ij)
    ctx._add_map(g1, g2, ret_grid, ind1,ind2,ind3)

    return ret_grid

