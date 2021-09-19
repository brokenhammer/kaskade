from numpy import dtype
from ..pyir.graph import Graph
from ..pyir import graph
from ..pyir.typing import DType
global_ctx = None

def _init():
    global global_ctx
    gg = Graph()
    gm = {}
    global_ctx = {"graph": gg, "map": gm, "inputs":{}, "outputs":{}}

def _get_graph():
    global global_ctx
    return global_ctx["graph"]

def _get_inputs():
    global global_ctx
    return global_ctx["inputs"]

def _get_map(g1,g2):
    id1 = id(g1)
    id2 = id(g2)
    global global_ctx
    global_map = global_ctx["map"]
    if (id1,id2) in global_map:
        return global_map[(id1,id2)]

    if (id2,id1) in global_map:
        return global_map[(id2,id1)]

    raise KeyError(f"Cannot found mapping record for {str(g1)} and {str(g2)}")

def _add_inputs(nodes, vals):
    global global_ctx
    global_inputs = global_ctx["inputs"]
    for idx, key in enumerate(nodes):
        global_inputs[key] = vals[idx]

def _add_map(g1,g2, g3, array1, array2,array3):
    assert(len(array1) == len(array2) == len(array3))
    global global_ctx
    global_map = global_ctx["map"]
    id1 = id(g1)
    id2 = id(g2)
    if (id1, id2) in global_map or (id2, id1) in global_map:
        print(f"Warning: failed to set mapping for {str(g1)} and {str(g2)} because it already exists.")
        return
    global_graph = global_ctx["graph"]
    name1 = f"map_{id1}_{id2}_1"
    name2 = f"map_{id1}_{id2}_2"
    name3 = f"map_{id1}_{id2}_3"
    ind_node1 = global_graph.new(len(array1), name1, DType.Int)
    ind_node2 = global_graph.new(len(array2), name2, DType.Int)
    ind_node3 = global_graph.new(len(array3), name3, DType.Int)
    global_map[(id1, id2)] = (g3, ind_node1, ind_node2, ind_node3)
    global_inputs = global_ctx["inputs"]
    global_inputs[ind_node1] = array1
    global_inputs[ind_node2] = array2
    global_inputs[ind_node3] = array3

