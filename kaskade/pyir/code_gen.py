import llvmlite.ir as ir
from ctypes import CFUNCTYPE, c_int, c_float, POINTER, c_double, py_object, byref, pointer, cast
import llvmlite.binding as bd
from .node import Node
from .utils import LoopCtx
from .typing import DType, map_kk_ct, map_kk_np
from typing import Union, Dict, List, Tuple
import numpy as np

int_type = ir.IntType(32)

bd.initialize()
bd.initialize_native_target()
bd.initialize_native_asmprinter()
target = bd.Target.from_triple(bd.get_default_triple())
tm = target.create_target_machine()


def func(name, module, rettype, argtypes):
    func_type = ir.FunctionType(rettype, argtypes, False)
    lfunc = ir.Function(module, func_type, name)
    entry_blk = lfunc.append_basic_block("entry")
    builder = ir.IRBuilder(entry_blk)
    return (lfunc, builder)


def initialize(
    input_arr: Dict[str, Union[np.ndarray, List, Tuple]],
    output_arr: List[Node]
) -> Union[ir.Function, ir.IRBuilder]:
    mod = ir.Module("mymodule")
    params = []
    for n in input_arr:
        params.append(map_kk_ct[n.dtype][1])
    for n in output_arr:
        params.append(map_kk_ct[n.dtype][1])
    mainfn, builder = func("main", mod, int_type, params)
    return mainfn, builder


def finalize(builder) -> None:
    builder.ret(ir.Constant(int_type, 0))
    return


def finalize_and_return(
    builder,
    input_arr: Dict[Node, Union[np.ndarray, List, Tuple]],
    output_arr: List[Node]
) -> List[np.ndarray]:

    finalize(builder)
    mod = builder.block.module
    mem_params = []
    POINTERs = []
    ret_mems = []
    inputs_mems = {}
    for (node, init_val) in input_arr.items():
        if isinstance(init_val, np.ndarray):
            arr_with_type = init_val.astype(map_kk_np[node.dtype])
            arr_c_mem = arr_with_type.ctypes.data_as(
                POINTER(map_kk_ct[node.dtype][0]))
            mem_params.append(arr_c_mem)
            inputs_mems[node] = arr_c_mem
        else:
            mem_type = map_kk_ct[node.dtype][0] * node.size
            input_mem = mem_type(*init_val)
            input_p = byref(input_mem)
            input_p = cast(input_p, POINTER(map_kk_ct[node.dtype][0]))
            mem_params.append(input_p)
            inputs_mems[node] = input_mem

        POINTERs.append(POINTER(map_kk_ct[node.dtype][0]))

    for node in output_arr:
        if node in input_arr:
            ret_mems.append(np.ctypeslib.as_array(
                inputs_mems[node], shape=[node.size]))
            mem_params.append(inputs_mems[node])
        else:
            length = node.size
            mem_type = map_kk_ct[node.dtype][0] * length
            ret_mem = mem_type()
            ret_mems.append(np.ctypeslib.as_array(ret_mem, shape=[node.size]))
            ret_p = byref(ret_mem)
            ret_p = cast(ret_p, POINTER(map_kk_ct[node.dtype][0]))
            mem_params.append(ret_p)
        POINTERs.append(POINTER(map_kk_ct[node.dtype][0]))

    backing_mod = bd.parse_assembly(str(mod))
    backing_mod.verify()
    with bd.create_mcjit_compiler(backing_mod, tm) as ee:
        ee.finalize_object()
        main_func_ptr = ee.get_function_address(builder.block.function.name)
        cfunc = CFUNCTYPE(
            c_int,
            *POINTERs
        )(main_func_ptr)
        cfunc(*mem_params)
        ee.detach()
        return ret_mems
