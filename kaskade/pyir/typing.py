from enum import Enum
import llvmlite.ir as ir
from ctypes import CFUNCTYPE, c_int, c_float, POINTER, c_double, py_object, byref, pointer, cast


class DType(Enum):
    Int = 1
    Float = 2
    Double = 3
    Complx = 4
    DComplx = 5


type_map_llvm = {
    DType.Int: ir.IntType(32),
    DType.Float: ir.FloatType(),
    DType.Double: ir.DoubleType(),
    DType.Complx: ir.FloatType(),
    DType.DComplx: ir.DoubleType()
}

int_type = ir.IntType(32)
float_type = ir.FloatType()
double_type = ir.DoubleType()
void_type = ir.VoidType()
ll_ptr_float = ir.PointerType(float_type)
ll_ptr_double = ir.PointerType(double_type)
ll_ptr_int = ir.PointerType(int_type)

map_kk_ct = {
    DType.Int: (c_int, ll_ptr_int),
    DType.Float: (c_float, ll_ptr_float),
    DType.Double: (c_double, ll_ptr_double)
}

map_kk_np = {
    DType.Int: "int32",
    DType.Float: "float32",
    DType.Double: "float64"
}

type_cast_llvm = {
    # Type cast function and parameter in llvmlite
    (DType.Int, DType.Float): ("sitofp", ir.FloatType()),
    (DType.Float, DType.Int): ("fptosi", ir.IntType(32)),
    (DType.Int, DType.Double): ("sitofp", ir.DoubleType()),
    (DType.Double, DType.Int): ("fptosi", ir.IntType(32)),
    (DType.Float, DType.Double): ("fpext", ir.DoubleType()),
    (DType.Double, DType.Float): ("fptrunc", ir.FloatType()),


    (DType.Int, DType.Complx): ("sitofp", ir.FloatType()),
    (DType.Int, DType.DComplx): ("sitofp", ir.DoubleType()),
    (DType.Float, DType.Complx): None,
    (DType.Float, DType.DComplx): ("fpext", ir.DoubleType()),
    (DType.Double, DType.Complx): ("fptrunc", ir.FloatType()),
    (DType.Double, DType.DComplx): None,
    (DType.Complx, DType.DComplx): ("fpext", ir.DoubleType()),
    (DType.DComplx, DType.Complx): ("fptrunc", ir.FloatType())
}

if __name__ == "__main__":
    print(isinstance(DType.Int, DType))
