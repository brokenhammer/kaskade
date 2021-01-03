from .node import Node, InputNode
import llvmlite.ir as ir
from .utils import LoopCtx, LRValue
from .typing import DType
from .code_gen import initialize, finalize_and_return

int_type = ir.IntType(32)

class Graph():
    def __init__(self):
        self._LEFT_nodes = set()
        self._mainfn, self._builder = initialize()
        self._reqired_nodes = set()
        self._compiled = False

    def new(self, size, name: str, dtype=DType.Float):
        tmp = InputNode(size, name, dtype)
        self._LEFT_nodes.add(tmp)
        return tmp

    def add_node(self, name: str, node: Node):
        if node in self._LEFT_nodes:
            print("Warning, node is alredy in the graph, return...")
            return
        node.rename(name)
        self._LEFT_nodes.add(node)
        node.vtype = LRValue.LEFT
        return node

    def _walk(self, node: Node) -> None:
        if node in self._reqired_nodes:
            return
        for n in node.dependence:
            self._walk(n)
        
        self._reqired_nodes.add(node)

    def _graph_gen(self, builder, outputs):
        for node in outputs:
            self._walk(node)

        from . import global_records as gr
        for cmd in gr._get():
            if cmd["target"] in self._reqired_nodes:
                if cmd["type"] == "set":
                    cmd["target"].vtype = LRValue.LEFT
                    self._LEFT_nodes.add(cmd["target"])

        for node in self._reqired_nodes:
            if node.vtype == LRValue.LEFT:
                node.allocate(builder)


        for cmd in gr._get():
            if cmd["target"] in self._reqired_nodes:
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
        self._LEFT_nodes.clear()
        self._reqired_nodes.clear()

    def compile_and_run(self, array=None):
        if not self._compiled:
            array.vtype = LRValue.LEFT
            if not array in self._LEFT_nodes:
                self._LEFT_nodes.add(array)

            self._graph_gen(self._builder, [array])
            self._compiled = True

        return finalize_and_return(self._builder, array)