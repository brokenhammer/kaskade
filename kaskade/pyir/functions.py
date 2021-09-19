from numpy import dtype
from .node import FuncNode, Node
from llvmlite import ir
from .typing import DType, type_map_llvm

def sin(src_node:Node) -> FuncNode:
    return FuncNode(src_node.size, "llvm.sin", src_node.dtype,[src_node])