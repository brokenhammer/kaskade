from enum import Enum
import llvmlite.ir as ir

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
    

