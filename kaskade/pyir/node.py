# -*- coding: utf-8 -*-

from __future__ import annotations
import llvmlite.ir as ir
import numpy as np
from ctypes import c_float, POINTER, pointer, byref, addressof, c_double, c_int32
from .utils import LRValue, uuname, BinaryOpr, UnaryOpr, biopr_map, LoopCtx, unopr_map
from . import global_records as gr
from .typing import DType, type_map_llvm, type_cast_llvm
from typing import Optional, Set
from math import ceil



float_type = ir.FloatType()
np_float = np.float32
int_type = ir.IntType(32)
ptr_float = ir.PointerType(float_type)


class Node():
    """Base class for all kinds of nodes
    """

    size: int
    name: str
    label: str
    dtype: DType
    vtype: LRValue
    dependence: Set[Node]
    iter_ind: int

    def __init__(
        self,
        size: int,
        name: str,
        dtype: DType,
        vtype: LRValue = LRValue.RIGHT
        ) -> None:
        super().__init__()
        self.size = size
        self.name = uuname(name)
        self.label = name
        self.dtype = dtype
        self.vtype = vtype
        self.gened = False
        self.dependence: Set[Node] = set()
        gr._append({"type":"make","target":self}) # leave a record of initilizing itself

 #-------------------------------------------
 # Basic operations
 #-------------------------------------------
    def allocate(self, builder: ir.IRBuilder) -> None:
        """Use IRBuilder to allocate memory of 'size' and use self.alloc as the pointer,
        Need twice of space for complex data.
        Should be called at the graph level.
        """
        if self.dtype in (DType.Int, DType.Float, DType.Double):
            self.alloc = builder.alloca(
                type_map_llvm[self.dtype],
                self.size,
                self.name+"-data"
            )
        else: #Complx or DComplx
            self.alloc = builder.alloca(
                type_map_llvm[self.dtype],
                self.size * 2,
                self.name+"-data"
            )

    def code_gen(self, bd: ir.IRBuilder) -> None:
        """Wrapper of the code_gen function,
        Rcursively generates it's dependences and call itselves _code_gen core
        Note that only Lvalue node need generate llvm ir
        """
        if self.gened:
            return
        for dep in self.dependence:
            dep.code_gen(bd)
        self.gened = True

        if self.vtype == LRValue.LEFT:
            self._code_gen(bd)
        
    def _code_gen(self, bd: ir.IRBuilder) -> None:
        raise NotImplementedError

    def get_ele(
        self,
        ind: (ir.Constant, ir.instructions.Instruction, int),
        builder: ir.IRBuilder
        ) -> [ir.instructions.Instruction]:
        """Generate the llvm ir to get cirtain element,
        ind can be a integer in python or int_type in llvm ir,
        but its type must be ir.IntType(32).
        It may contains two elements for complex number.
        """
        if type(ind) == int:
            ind = ir.Constant(int_type, ind)
        assert(ind.type == int_type)
        if not self.gened:
            print("Error: try to access ungenerated array")
            raise NameError
        if self.size == 1:
            ind = ir.Constant(int_type, 0)
        if self.vtype == LRValue.LEFT:
            return self._load_from_alloc(ind, builder)
        else:
            # print(self._load_from_src(ind, builder))
            return self._load_from_src(ind, builder)
        
    def _load_from_src(self, ind, builder) -> [ir.LoadInstr]:
        raise NotImplementedError

    def _store_to_alloc(
        self,
        index: (ir.Constant, ir.Instruction),
        src_nums: [ir.LoadInstr],
        builder: ir.IRBuilder
        ) -> None:
        if not self.dtype in (DType.Complx, DType.DComplx):
            dest_ptr = builder.gep(self.alloc, [index])
            builder.store(src_nums[0], dest_ptr)
        else:
            cmplx_dest_index = builder.mul(index, ir.Constant(int_type, 2))
            dest_ptr_r = builder.gep(self.alloc, [cmplx_dest_index])
            builder.store(src_nums[0], dest_ptr_r)
            cmplx_dest_index = builder.add(cmplx_dest_index, ir.Constant(int_type, 1))
            dest_ptr_i = builder.gep(self.alloc, [cmplx_dest_index])
            builder.store(src_nums[1], dest_ptr_i)

    def _load_from_alloc(
        self,
        index: (ir.Constant, ir.Instruction),
        builder: ir.IRBuilder
        ) -> [ir.LoadInstr]:
        if not self.dtype in (DType.Complx, DType.DComplx):
            self_ptr = builder.gep(self.alloc, [index])
            products = [builder.load(self_ptr)]
        else:
            cmplx_index = builder.mul(index, ir.Constant(int_type, 2))
            self_ptr_real = builder.gep(self.alloc, [cmplx_index])
            product_real = builder.load(self_ptr_real)
            cmplx_index = builder.add(cmplx_index, ir.Constant(int_type, 1))
            self_ptr_imag = builder.gep(self.alloc, [cmplx_index])
            product_imag = builder.load(self_ptr_imag)
            products = [product_real, product_imag]
        
        return products

 #-------------------------------------------
 # Mathematical operations
 #-------------------------------------------

    def __add__(
        self,
        other: (Node, int, float, complex)
        ) -> Node:
        if isinstance(other, (int, float, complex)):
            other = make_const_node(other)
        return BinOpNode(BinaryOpr.ADD, self, other)
    
    def __radd__(
        self,
        other: (Node, int, float, complex)
        ) -> Node:
        if isinstance(other, (int, float, complex)):
            other = make_const_node(other)
        return BinOpNode(BinaryOpr.ADD, self, other)
    
    def __sub__(
        self,
        other: (Node, int, float, complex)
        ) -> Node:
        if isinstance(other, (int, float, complex)):
            other = make_const_node(other)
        return BinOpNode(BinaryOpr.SUB, self, other)

    def __rsub__(
        self,
        other: (Node, int, float, complex)
        ) -> Node:
        if isinstance(other, (int, float, complex)):
            other = make_const_node(other)
        return BinOpNode(BinaryOpr.SUB, other, self)

    def __mul__(
        self,
        other: (Node, int, float, complex)
        ) -> Node:
        if isinstance(other, (int, float, complex)):
            other = make_const_node(other)
        return BinOpNode(BinaryOpr.MUL, self, other)

    def __rmul__(
        self,
        other: (Node, int, float, complex)
        ) -> Node:
        if isinstance(other, (int, float, complex)):
            other = make_const_node(other)
        return BinOpNode(BinaryOpr.MUL, self, other)

    def __floordiv__(
        self,
        other: (Node, int, float, complex)
        ) -> Node:
        if isinstance(other, (int, float, complex)):
            other = make_const_node(other)

        return BinOpNode(BinaryOpr.SDIV, self, other)

    def __rfloordiv__(
        self,
        other: (Node, int, float, complex)
        ) -> Node:
        if isinstance(other, (int, float, complex)):
            other = make_const_node(other)

        return BinOpNode(BinaryOpr.SDIV, other, self)

    def __truediv__(
        self,
        other: (Node, int, float, complex)
        ) -> Node:
        if isinstance(other, (int, float, complex)):
            other = make_const_node(other)
        return BinOpNode(BinaryOpr.FDIV, self, other)

    def __rtruediv__(
        self,
        other: (Node, int, float, complex)
        ) -> Node:
        if isinstance(other, (int, float, complex)):
            other = make_const_node(other)
        return BinOpNode(BinaryOpr.FDIV, other, self)

 #-------------------------------------------
 # Indexing operations
 #-------------------------------------------
    def __getitem__(
        self,
        index: (int, list, tuple, Node, slice, np.ndarray)
        ) -> Node:
        return GetSliceNode(index, self)

    def __setitem__(
        self,
        index: (int, Node, slice, list, tuple, np.ndarray),
        val: (float, int, complex, Node)
        ) -> None:
        """Setting index,
        Do not recommend to use numpy array for index
        Do not recommend to use numpy array, list or tuple for val
        """
        
        if isinstance(val, (float, int, complex)):
            val = make_const_node(val)
        length = compute_size(index, self.size)
        if isinstance(index, Node):
            assert(index.dtype == DType.Int)
        assert((length == val.size) or len(val) == 1)
        gr._append({"type": "set", "target": self, "src": (index, val)})

    def _gen_setitem(
        self,
        builder: ir.IRBuilder,
        index: (int, Node, slice, list, tuple, np.ndarray),
        val: Node) -> None:
        """When set index to one node, it must be LValue node,
        if not, the graph maintainer should modify its vtype to LEFT.
        Also, only when the arry is required, it can be generated.
        """
        if isinstance(index, int):
            const0 = ir.Constant(int_type, 0)
            src_nums = val.get_ele(const0, builder)
            index = ir.Constant(int_type, index)
            self._store_to_alloc(index, src_nums, builder)
        elif isinstance(index, slice):
            size = compute_size(index, self.size)
            start, _, step = index.indices(val.size)
            v_start = ir.Constant(int_type, start)
            v_step = ir.Constant(int_type, step)
            dest_index_ptr = builder.alloca(int_type, 1)
            builder.store(v_start, dest_index_ptr)
            with LoopCtx(self.name+"_set_slice", builder, size) as loop:
                loop_inc = builder.load(loop.inc)
                dest_index = builder.load(dest_index_ptr)
                src_nums = val.get_ele(loop_inc, builder)
                self._store_to_alloc(dest_index, src_nums, builder)
                builder.store(builder.add(
                    dest_index, v_step), dest_index_ptr)
        elif isinstance(index, Node):
            with LoopCtx(self.name+"_set_slice", builder, index.size) as loop:
                loop_inc = builder.load(loop.inc)
                dest_index = index.get_ele(loop_inc)[0]
                src_nums = val.get_ele(loop_inc, builder)
                self._store_to_alloc(dest_index, src_nums, builder)
        else:
            all_inds = builder.alloca(int_type, len(index))
            #TODO: change this to malloc function
            for i in range(len(index)):
                ind_ptr = builder.gep(all_inds, [ir.Constant(int_type, i)])
                builder.store(ir.Constant(int_type, index[i]), ind_ptr)
            with LoopCtx(self.name+"_set_slice", builder, len(index)) as loop:
                loop_inc = builder.load(loop.inc)
                dest_index_ptr = builder.gep(all_inds, [loop_inc])
                dest_index = builder.load(dest_index_ptr)
                src_nums = val.get_ele(loop_inc)
                self._store_to_alloc(dest_index, src_nums, builder)

 #-------------------------------------------
 # Iteration methods
 #-------------------------------------------
    def __iter__(self):
        self.iter_ind = 0
        return self

    def __next__(self) -> Optional[GetSliceNode]:
        self.iter_ind += 1
        if self.iter_ind >= self.size:
            raise StopIteration
        else:
            return self[self.iter_ind]

    def __len__(self) -> int:
        return self.size

 # #-------------------------------------------
 # # Moving methods
 # #-------------------------------------------
 #     def mov_from(self, other):
 #         pass
 #     def _gen_mov(self,builder: ir.IRBuilder):
 #         pass
 #
 #-------------------------------------------
 # Other methods
 #-------------------------------------------
    def rename(self, new_name: str) -> None:
        self.name = uuname(new_name)
        self.label = new_name


class InputNode(Node):
    """Input Node for the whole graph, cannot generated by other node.
    """

    def __init__(
        self,
        size: int,
        name: str,
        dtype: DType
        ):
        if not name:
            name = "input"
        super().__init__(size, name, dtype, LRValue.LEFT)

    def fill(self, builder: ir.IRBuilder) -> None:
        pass

    def _code_gen(self, builder: ir.IRBuilder) -> None:
        self.fill(builder)


class BinOpNode(Node):
    """Generated node from binary operator
    """

    def __init__(self, opr: BinaryOpr ,LHS: Node, RHS: Node) -> None:
        size = det_size(LHS, RHS)
        dtype = det_dtype(opr, LHS.dtype, RHS.dtype)
        super().__init__(size, biopr_map[opr][dtype], dtype)
        self.opr = opr
        self.dependence = {LHS, RHS}
        self.LHS = LHS
        self.RHS = RHS

    def _code_gen(self, builder: ir.IRBuilder) -> None:
        # instr = getattr(builder, biopr_map[self.opr][self.dtype])
        with LoopCtx(self.name, builder, self.size) as loop:
            loop_inc = builder.load(loop.inc)
            products = self._build_opr(loop_inc, builder)
            self._store_to_alloc(loop_inc, products, builder)

    def _build_opr(
        self,
        loop_inc: (ir.instructions.Instruction, ir.Constant),
        builder: ir.IRBuilder
        ) -> [ir.instructions.Instruction]:
        """Build llvm ir for binary operator
        If the type of any operand is different from the output type, a type cast from type_cast_llvm dict
        should be performed.
        Special calculations on Complx and DComplx output
        """
        assert(loop_inc.type == int_type)
        if not self.dtype in (DType.Complx, DType.DComplx):
            left_num = self.LHS.get_ele(loop_inc, builder)[0]
            right_num = self.RHS.get_ele(loop_inc, builder)[0]

            opr_instr = getattr(builder, biopr_map[self.opr][self.dtype])

            if self.LHS.dtype != self.dtype:
                cast_params = type_cast_llvm[(self.LHS.dtype, self.dtype)]
                if cast_params:
                    cast_instr = getattr(builder, cast_params[0])
                    left_num = cast_instr(left_num, cast_params[1])
            
            if self.RHS.dtype != self.dtype:
                cast_params = type_cast_llvm[(self.RHS.dtype, self.dtype)]
                if cast_params:
                    cast_instr = getattr(builder, cast_params[0])
                    right_num = cast_instr(right_num, cast_params[1])

            product = opr_instr(left_num, right_num)
            return [product]
        else:
            
            if self.LHS.dtype in (DType.Complx, DType.DComplx):
                left_num_real, left_num_imag = self.LHS.get_ele(loop_inc, builder)

            else:
                left_num_real = self.LHS.get_ele(loop_inc, builder)[0]
                left_num_imag = ir.Constant(type_map_llvm[self.LHS.dtype], 0)

            if self.LHS.dtype != self.dtype:
                cast_params = type_cast_llvm[(self.LHS.dtype, self.dtype)]
                if cast_params:
                    cast_instr = getattr(builder, cast_params[0])
                    left_num_real = cast_instr(left_num_real, cast_params[1])
                    left_num_imag = cast_instr(left_num_imag, cast_params[1])

            if self.RHS.dtype in (DType.Complx, DType.DComplx):
                right_num_real, right_num_imag = self.RHS.get_ele(loop_inc, builder)
            else:
                right_num_real = self.RHS.get_ele(loop_inc, builder)[0]
                right_num_imag = ir.Constant(type_map_llvm[self.RHS.dtype], 0)

            if self.RHS.dtype != self.dtype:
                cast_params = type_cast_llvm[(self.RHS.dtype, self.dtype)]
                if cast_params:
                    cast_instr = getattr(builder, cast_params[0])
                    right_num_real = cast_instr(right_num_real, cast_params[1])
                    right_num_imag = cast_instr(right_num_imag, cast_params[1])

            if self.opr == BinaryOpr.ADD or self.opr == BinaryOpr.SUB:
                opr_instr = getattr(builder, biopr_map[self.opr][self.dtype])
                product_real = opr_instr(left_num_real, right_num_real)
                product_imag = opr_instr(left_num_imag, right_num_imag)
            elif self.opr == BinaryOpr.MUL:
                rxr = builder.fmul(left_num_real, right_num_real)
                rxi = builder.fmul(left_num_real, right_num_imag)
                ixr = builder.fmul(left_num_imag, right_num_real)
                ixi = builder.fmul(left_num_imag, right_num_imag)

                product_real = builder.fsub(rxr, ixi)
                product_imag = builder.fadd(rxi, ixr)
            elif self.opr == BinaryOpr.FDIV:
                denom1 = builder.fmul(right_num_real, right_num_real)
                denom2 = builder.fmul(right_num_imag, right_num_imag)
                denom = builder.fadd(denom1, denom2)
                rxr = builder.fmul(left_num_real, right_num_real)
                rxi = builder.fmul(left_num_imag, right_num_imag)
                ixr = builder.fmul(left_num_imag, right_num_real)
                ixi = builder.fmul(left_num_imag, right_num_imag)
                product_real = builder.fadd(rxr, ixi)
                product_imag = builder.fsub(ixr, rxi)

                product_real = builder.fdiv(product_real, denom)
                product_imag = builder.fdiv(product_imag, denom)

            return [product_real, product_imag]

    def _load_from_src(
        self,
        ind: (ir.Constant, ir.Instruction),
        builder: ir.IRBuilder
        ) -> [ir.Instruction]:
        # If self is Rvalue type, get elements from its source
        # and calculate them inplace.
        # If self is Lvalue type, the value has been calculated and stored in self.alloc
        # just find that value and return it.

        return self._build_opr(ind, builder)


class UnaOpNode(Node):
    """Generated node from unary operator
    """
    def __init__(self, opr: UnaryOpr ,SRC: Node) -> None:
        super().__init__(SRC.size, unopr_map[opr][self.dtype], SRC.dtype)
        self.SRC = SRC
        self.opr = opr
        self.dependence = {SRC}

    def _code_gen(self, builder:ir.IRBuilder) -> None:
        with LoopCtx(self.name, builder, self.size) as loop:
            loop_inc = builder.load(loop.inc)
            products = self._build_opr(loop_inc, builder)
            self._store_to_alloc(loop_inc, products, builder)

    def _build_opr(
        self,
        loop_inc: (ir.Constant, ir.instructions.Instruction),
        builder: ir.IRBuilder) -> [ir.instructions.Instruction]:
        
        if not self.dtype in (DType.DComplx, DType.Complx):
            instr = getattr(builder, unopr_map[self.opr][self.dtype])
            src_num = self.SRC.get_ele(loop_inc, builder)[0]
            products = [instr(src_num)]

        else:
            if self.opr == UnaryOpr.NEG:
                src_real, src_imag = self.SRC.get_ele(loop_inc, builder)
                product_real = builder.neg(src_real)
                product_imag = builder.neg(src_imag)
                products = [product_real, product_imag]

            else:
                print("Unsupported unary operation: {}".format(self.opr))
                raise ArithmeticError
        return products

    def _load_from_src(
        self,
        ind: (ir.Constant, ir.Instruction),
        builder: ir.IRBuilder
        ) -> [ir.Instruction]:

        return self._build_opr(ind, builder)


class ConstNode(Node):
    """Node that stores a constant number
    Exactly one node allowed for each number
    """
    _existing_vals = {}
    def __new__(cls, val:(int, float, complex), dtype:DType, *args, **kw):
        if not val in cls._existing_vals:
            cls._existing_vals[val] = object.__new__(cls, *args, **kw)
        return cls._existing_vals[val]
    
    def __init__(self, val: (int, float, complex), dtype: DType) -> None:
        super().__init__(1, "const"+str(val), dtype)

        if not self.dtype in (DType.Complx, DType.DComplx):
            self.const = ir.Constant(type_map_llvm[self.dtype], val)
        else:
            self.const_real = ir.Constant(type_map_llvm[self.dtype], val.real)
            self.const_imag = ir.Constant(type_map_llvm[self.dtype], val.imag)
    
    def _load_from_src(
        self,
        ind:(ir.Constant, ir.Instruction),
        builder: ir.IRBuilder
        ) -> [ir.Instruction]:
        zero = ir.Constant(type_map_llvm[self.dtype], 0)
        if not self.dtype in (DType.Complx, DType.DComplx):
            if self.dtype == DType.Float:
                return [builder.fadd(zero, self.const)]
            elif self.dtype == DType.Int:
                return [builder.add(zero, self.const)]
        else:
            return [builder.fadd(zero, self.const_real), builder.fadd(zero, self.const_imag)]


class GetSliceNode(Node):
    """ Node generated from indexing,
    support int, slice, Node and numpy array index
    """
    def __init__(
        self,
        ind: (int, list, tuple, Node, slice, np.ndarray),
        src: Node
        ) -> None:
        size = compute_size(ind, src.size)
        super().__init__(size, "slice", src.dtype)
        self.dependence = {src}
        if isinstance(ind, Node):
            self.dependence.add(ind)
        self.ind = ind
        self.src = src

    def allocate(self, builder: ir.IRBuilder):
        super().allocate(builder)
        if isinstance(self.ind, (list, tuple, np.ndarray)):
            self.src_inds = builder.alloca(
                int_type,
                len(self.ind),
                name=self.name+"-inds")

    def code_gen(self, builder: ir.IRBuilder) -> None:
        """Wrapper of the code_gen function,
        Rcursively generates it's dependences and call itselves _code_gen core
        Note that only Lvalue node need generate llvm ir
        """
        if self.gened:
            return
        for dep in self.dependence:
            dep.code_gen(builder)
        self.gened = True

        if isinstance(self.ind, (list, tuple, np.ndarray)):
            for i in range(len(self.ind)):
                index = ir.Constant(int_type, self.ind[i])
                builder.store(index, builder.gep(
                    self.src_inds, [ir.Constant(int_type, i)]))
        if self.vtype == LRValue.LEFT:
            self._code_gen(builder)

    def _code_gen(self, builder: ir.IRBuilder) -> None:
        """Generating indexing llvm ir
        Note that indexing using numpy array is not recomended,
        because it generates static loops and will cause the generated
        llvm ir too large.
        """
        if isinstance(self.ind, slice):
            start, _, step = self.ind.indices(self.src.size)
            step_const = ir.Constant(int_type, step)
            src_index_ptr = builder.alloca(int_type, 1)
            builder.store(ir.Constant(int_type, start), src_index_ptr)

        with LoopCtx(self.name, builder, self.size) as loop:
            loop_inc = builder.load(loop.inc)
            if isinstance(self.ind, slice):
                src_index = builder.load(src_index_ptr)
            elif isinstance(self.ind, Node):
                src_index = self.ind.get_ele(loop_inc, builder)[0]
            else:
                src_index_ptr = builder.gep(self.src_inds, [loop_inc])
                src_index = builder.load(src_index_ptr)

            src_nums = self.src.get_ele(src_index, builder)
            self._store_to_alloc(loop_inc, src_nums, builder)

            if isinstance(self.ind, slice):
                builder.store(builder.add(
                    src_index, step_const), src_index_ptr)
                

    def _load_from_src(
        self,
        ind: (ir.Constant, ir.Instruction),
        builder: ir.IRBuilder
        ) -> [ir.Instruction]:

        if isinstance(self.ind, slice):
            start, _, step = self.ind.indices(self.src.size)
            muled = builder.mul(ind, ir.Constant(int_type, step))
            src_index = builder.add(muled, ir.Constant(int_type, start))
        elif isinstance(self.ind, Node):
            src_index = self.ind.get_ele(ind, builder)[0]
        else:
            src_index_ptr = builder.gep(self.src_inds, [ind])
            src_index = builder.load(src_index_ptr)

        return self.src.get_ele(src_index, builder)


class FuncNode(Node):
    # mapping scalar function to vector
    def __init__(self, size: int, func_name: str, rettype: DType, ftype: ir.FunctionType, SRC: list[Node] or []):
        super().__init__(size, func_name, rettype)
        self.func_name = func_name
        self.SRC = SRC
        self.ftype = ftype
        for n in SRC:
            assert(not n.dtype in (DType.Complx, DType.DComplx))
            self.dependence.add(n)

    def _code_gen(self, builder: ir.IRBuilder) -> None:
        mod = builder.block.module
        instr = ir.values.Function(mod, self.ftype, self.func_name)
        params = []
        with LoopCtx(self.name, builder, self.size) as loop:
            index = builder.load(loop.inc)
            data_ptr = builder.gep(self.alloc, [index])
            for n in self.SRC:
                params.append(n.get_ele(index, builder)[0])
            res = builder.call(instr, params)
            builder.store(res, data_ptr)

    def _load_from_src(
        self,
        ind: (ir.Constant, ir.Instruction),
        builder: ir.IRBuilder
        ) -> [ir.Instruction]:

        mod = builder.block.module
        instr = ir.values.Function(mod, self.ftype, self.func_name)
        params = []
        for n in self.SRC:
            params.append(n.get_ele(ind, builder))
        res = builder.call(instr, params)
        return [res]


def det_size(LHS: Node, RHS: Node) -> int:
    """Return the expected size after operation
    Either at least one of them is ScalarNode,
    or they must have the same size.
    """

    if LHS.size == 1:
        return RHS.size
    if RHS.size == 1:
        return LHS.size

    assert(LHS.size == RHS.size)
    return LHS.size

def det_dtype(opr:BinaryOpr, Ltype: DType, Rtype: DType) -> DType:
    """Determine the output data type,
    Implicit dtype conversion may applies.
    Note that type may varies if opr is SDIV or FDIV
    """

    if opr != BinaryOpr.SDIV:
        if Ltype == DType.Int and Rtype == DType.Int and opr == BinaryOpr.FDIV:
            return DType.Float
        if Ltype == DType.Int:
            return Rtype
        if Rtype == DType.Int:
            return Ltype
        if Ltype == DType.Float:
            return Rtype
        if Rtype == DType.Float:
            return Ltype
        if Ltype == DType.Double:
            if Rtype == DType.Double:
                return DType.Double
            else:
                return DType.DComplx
        if Rtype == DType.Double:
            if Ltype == DType.Double:
                return DType.Double
            else:
                return DType.DComplx
        if Ltype == DType.DComplx or Rtype == DType.DComplx:
            return DType.DComplx

        return DType.Complx

    if opr == BinaryOpr.SDIV:
        if Ltype in (DType.DComplx, DType.Complx) or Rtype in (DType.DComplx, DType.Complx):
            print("Cannot perform floor div between {} and {}".format(Ltype, Rtype))
            raise ArithmeticError
        else:
            return DType.Int

def compute_size(index_obj, max_len: int) -> int:
    if isinstance(index_obj, int):
        return 1
    elif isinstance(index_obj, (list, tuple)):
        return len(index_obj)
    elif isinstance(index_obj, slice):
        (start, stop, step) = index_obj.indices(max_len)
        # return (stop - start - 1) // step + 1
        size = ceil((stop-start)/step)
        assert(size <= max_len)
        return size
    elif isinstance(np.ndarray):
        assert(index_obj.dtype=="int")
        return len(index_obj)
    elif isinstance(Node):
        assert(index_obj.dtype==DType.Int)
        return Node.size
    else:
        print("Unsupported indexing type")
        raise IndexError

def make_const_node(raw: (int, float, complex)) -> ConstNode:
    if isinstance(raw, int):
        return ConstNode(raw, DType.Int)
    elif isinstance(raw, float):
        return ConstNode(raw, DType.Float)
    elif isinstance(raw, complex):
        return ConstNode(raw, DType.Complx)