from .node import Node, InputNode
import llvmlite.ir as ir
from .utils import LoopCtx, LRValue
from .typing import DType
from . import global_records as gr
import numpy as np
from .code_gen import initialize, finalize_and_return
from typing import Union, Dict, List, Tuple

int_type = ir.IntType(32)


class Graph():
    def __init__(self):
        self._LEFT_nodes = set()
        self._required_nodes = set()
        gr._init()
        self._compiled = False

    def new(
        self,
        size,
        name: str,
        dtype: DType = DType.Float
    ) -> Node:
        tmp = InputNode(size, name, dtype)
        return tmp

    def add_node(self, name: str, node: Node) -> Node:
        node.rename(name)
        node.vtype = LRValue.LEFT
        return node

    def _walk(self, node: Node) -> None:
        if node in self._required_nodes:
            return
        for n in node.dependence:
            self._walk(n)

        self._required_nodes.add(node)

    def _graph_gen(self, builder, inputs, outputs) -> None:
        for node in inputs:
            assert(isinstance(node, InputNode))
        for node in outputs:
            node.vtype = LRValue.LEFT
            self._walk(node)

        for cmd in gr._get():
            if cmd["target"] in self._required_nodes:
                if cmd["type"] == "set":
                    cmd["target"].vtype = LRValue.LEFT

        for node in self._required_nodes:
            if node.vtype == LRValue.LEFT and not node in outputs and not node in inputs:
                node.allocate(builder)

        # Note we ONLY generate the ir but not really allocating memory here!
        for order, node in enumerate(inputs):
            node.set_alloc_from(self._mainfn.args[order])

        for order, node in enumerate(outputs):
            node.set_alloc_from(self._mainfn.args[order+len(inputs)])

        for cmd in gr._get():
            if cmd["target"] in self._required_nodes:
                if cmd["type"] == "make":
                    cmd["target"].code_gen(builder)
                elif cmd["type"] == "set":
                    ind, val = cmd["src"]
                    val.code_gen(builder)
                    if isinstance(ind, Node):
                        ind.code_gen(builder)
                    cmd["target"].code_gen(builder)
                    cmd["target"]._gen_setitem(builder, ind, val)
        gr._clear()
        self._required_nodes.clear()

    def compile_and_run(self, outputs=[], inputs={}) -> List[np.ndarray]:
        if not self._compiled:
            self._mainfn, self._builder = initialize(inputs, outputs)
            self._graph_gen(self._builder, inputs, outputs)
            self._compiled = True

        return finalize_and_return(self._builder, inputs, outputs)
