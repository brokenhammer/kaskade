from enum import Enum
from llvmlite import ir
from .typing import DType

global_node_id = -1
int_type = ir.IntType(32)


class LRValue(Enum):
    """Enum class for the node,
    whether it is a left value or a right value
    """

    LEFT = 1
    RIGHT = 2


class BinaryOpr(Enum):
    """Legal Binay operator
    """
    ADD = 1
    SUB = 2
    MUL = 3
    SDIV = 4
    FDIV = 5


class UnaryOpr(Enum):
    """Legal Unary operator
    """
    NEG = 1


def uuname(raw_name):
    """Get a unique name for every node
    """

    global global_node_id
    global_node_id += 1
    return raw_name + "_" + str(global_node_id)


biopr_map = {
    BinaryOpr.ADD: {
        DType.Int: "add",
        DType.Float: "fadd",
        DType.Double: "fadd",
        DType.Complx: "fadd",
        DType.DComplx: "fadd"
    },
    BinaryOpr.SUB: {
        DType.Int: "sub",
        DType.Float: "fsub",
        DType.Double: "fsub",
        DType.Complx: "fsub",
        DType.DComplx: "fsub"
    },
    BinaryOpr.MUL: {
        DType.Int: "mul",
        DType.Float: "fmul",
        DType.Double: "fmul",
        DType.Complx: "fmul",
        DType.DComplx: "fmul"
    },
    BinaryOpr.SDIV: {
        DType.Int: "sdiv",
    },
    BinaryOpr.FDIV: {
        DType.Float: "fdiv",
        DType.Double: "fdiv",
        DType.Complx: "fdiv",
        DType.DComplx: "fdiv"
    }
}
unopr_map = {
    UnaryOpr.NEG: {
        DType.Int: "neg",
        DType.Float: "neg",
        DType.Double: "neg",
        DType.Complx: "neg",
        DType.DComplx: "neg"
    }
}


class LoopCtx():
    """Context for loop ir in llvmlite
    """

    def __init__(
        self,
        prefix: str,
        builder: ir.IRBuilder,
        stop: (int_type, ir.Constant),
        start: (int_type, ir.Constant) = 0,
        step: (int_type, ir.Constant) = 1
    ) -> None:
        self.builder = builder
        self.inc = builder.alloca(int_type, name='i')
        self.stop = ir.Constant(
            int_type, stop) if isinstance(stop, int) else stop
        self.start = ir.Constant(int_type, start) if isinstance(
            start, int) else start
        self.step = ir.Constant(
            int_type, step) if isinstance(step, int) else step
        assert(self.stop.type == ir.IntType(32))
        assert(self.start.type == ir.IntType(32))
        assert(self.step.type == ir.IntType(32))
        fn = self.builder.block.function
        self.init_block = fn.append_basic_block(prefix + '_for.init')
        self.body_block = fn.append_basic_block(prefix + '_for.body')
        # self.cond_block = fn.append_basic_block(prefix + '_for.cond')
        self.end_block = fn.append_basic_block(prefix + '_for.end')

    def __enter__(self):
        self.builder.branch(self.init_block)
        self.builder.position_at_end(self.init_block)
        self.builder.store(self.start, self.inc)
        self.builder.branch(self.body_block)
        self.builder.position_at_end(self.body_block)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # self.builder.branch(self.cond_block)
        # self.builder.position_at_end(self.cond_block)
        crt_inc = self.builder.load(self.inc)
        const_step = self.step
        self.builder.store(self.builder.add(crt_inc, const_step), self.inc)
        cond = self.builder.icmp_signed('<', self.builder.load(
            self.inc), self.stop)
        self.builder.cbranch(cond, self.body_block, self.end_block)
        self.builder.position_at_end(self.end_block)
