import llvmlite.ir as ir
from ctypes import CFUNCTYPE, c_int, c_float, POINTER, c_double, py_object, byref, pointer, cast
import llvmlite.binding as bd
from . import global_records as gr
from .utils import LoopCtx
from .typing import DType

int_type = ir.IntType(32)
float_type = ir.FloatType()
double_type = ir.DoubleType()
void_type = ir.VoidType()
ptr_float = ir.PointerType(float_type)
ptr_double = ir.PointerType(double_type)
ptr_int = ir.PointerType(int_type)
type_dict = {int_type: c_int, float_type: c_float,
             ptr_float: POINTER(c_float), ptr_int: POINTER(c_int)}


bd.initialize()
bd.initialize_native_target()
bd.initialize_native_asmprinter()
target = bd.Target.from_triple(bd.get_default_triple())
tm = target.create_target_machine()


# def run_func(engine, fn, arglist):
#     func_ptr = engine.get_function_address(fn.name)
#     rettype = type_dict[fn.return_value.type]
#     argtype = []
#     for arg in fn.args:
#         argtype.append(type_dict[arg.type])
#     cfunc = CFUNCTYPE(rettype, *argtype)(func_ptr)
#     res = cfunc(*arglist)
#     return res


def func(name, module, rettype, argtypes):
    func_type = ir.FunctionType(rettype, argtypes, False)
    lfunc = ir.Function(module, func_type, name)
    entry_blk = lfunc.append_basic_block("entry")
    builder = ir.IRBuilder(entry_blk)
    return (lfunc, builder)


def initialize():
    gr._init()
    mod = ir.Module("mymodule")
    # mainfn, builder = func("main", mod, int_type, [ptr_float])
    mainfn, builder = func("main", mod, int_type, [ptr_int, ptr_float])
    return mainfn, builder


def finalize(builder, array=None):
    if not array:
        res = ir.Constant(int_type, 0)
        builder.ret(res)
        return
    else:
        fn = builder.block.function
        with LoopCtx("ret", builder, array.size) as loop:
            index = builder.load(loop.inc)
            value = array.get_ele(index, builder)[0]
            if array.dtype == DType.Int:
                dest_ptr = builder.gep(fn.args[0], [index])
            else:
                dest_ptr = builder.gep(fn.args[1], [index])
            builder.store(value, dest_ptr)
        builder.ret(ir.Constant(int_type, 0))
        return


def finalize_and_return(builder, array=None):
    finalize(builder, array=array)
    mod = builder.block.module
    length = array.size
    if array.dtype == DType.Float:
        ret_int_p = POINTER(c_int)()
        mem_type = c_float * length
        ret_mem = mem_type()
        ret_float_p = byref(ret_mem)
        ret_float_p = cast(ret_float_p, POINTER(c_float))
    elif array.dtype == DType.Int:
        ret_float_p = POINTER(c_float)()
        mem_type = c_int * length
        ret_mem = mem_type()
        ret_int_p = byref(ret_mem)
        ret_int_p = cast(ret_int_p, POINTER(c_int))

    backing_mod = bd.parse_assembly(str(mod))
    backing_mod.verify()
    with bd.create_mcjit_compiler(backing_mod, tm) as ee:
        ee.finalize_object()
        main_func_ptr = ee.get_function_address(builder.block.function.name)
        cfunc = CFUNCTYPE(c_int, POINTER(c_int),
                          POINTER(c_float))(main_func_ptr)
        cfunc(ret_int_p, ret_float_p)
        ee.detach()
        return ret_mem


# if __name__ == "__main__":
#     print("===Codegen & RUN test===")
#     mainfn, builder = initialize()
#     gg = Graph()
#     z = gg.new("z", 4)
#     x = gg.add("x", z + 1)
#     z[2] = 1.2
#     y = gg.add("y",nabla(z))
#     z.set_input(2.5)
#     gg.code_gen(builder)
#     ret = finalize_and_return(builder, y)
#     print(ret[:])
