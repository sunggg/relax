# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, missing-docstring
"""Unittests for tvm.script.ir_builder.tir"""
import pytest
import numpy as np
import tvm
from tvm import tir
from tvm.runtime import ndarray
from tvm.script.ir_builder import tir as T
from tvm.script.ir_builder import IRBuilder
from tvm.ir.base import assert_structural_equal


def test_ir_builder_tir_primfunc():
    with IRBuilder() as ib:
        with T.prim_func():
            T.arg("a", T.handle())
            T.arg("b", T.var("int64"))
            T.arg("c", T.buffer_decl((128, 128), "float32"))
            d = T.arg("d", T.handle())
            e = T.arg("e", T.buffer_decl((1024,), "int8"))
            T.func_attr({"key": "value"})
            T.func_ret(tvm.ir.PrimType("int64"))
            buffer_d = T.match_buffer(d, (64, 64), "int64")
            T.preflattened_buffer(e, (32, 32), "int8", data=e.data)
            T.evaluate(0)
    # the prim_func generated by IRBuilder
    prim_func_actual = ib.get()

    # the expected prim_func
    c_handle, c_buffer = tir.Var("c_handle", "handle"), tir.decl_buffer(
        (128, 128), "float32", name="c"
    )
    d_handle, d_buffer = tir.Var("d", "handle"), tir.decl_buffer((64, 64), "int64", name="d")
    e_handle, e_buffer = tir.Var("e_handle", "handle"), tir.decl_buffer((1024,), "int8", name="e")
    prim_func_expected = tir.PrimFunc(
        params=[
            tir.Var("a", "handle"),
            tir.Var("b", "int64"),
            c_handle,
            d_handle,
            e_handle,
        ],
        body=tir.Evaluate(0),
        ret_type=tvm.ir.PrimType("int64"),
        buffer_map={c_handle: c_buffer, d_handle: d_buffer, e_handle: e_buffer},
        preflattened_buffer_map={
            e_handle: tir.decl_buffer((32, 32), "int8", name="e_preflatten", data=e_buffer.data)
        },
        attrs=tvm.ir.make_node("DictAttrs", key="value"),
    )
    # Check if the generated ir is expected
    assert_structural_equal(prim_func_actual, prim_func_expected, map_free_vars=True)


def test_ir_builder_tir_block():
    with IRBuilder() as ib:
        a = T.var("int64", "a")
        b = T.buffer_decl((128, 128), "float32")
        c = T.buffer_decl((128, 128), "float32")
        d = T.var("int32", "d")
        e = T.buffer_decl((128, 128), "float32")
        f = T.var("int32", "f")
        with T.block("block"):
            T.where(a > 1)
            T.reads(b[0:16, 0:16])
            T.writes(c[d:128, d:128])
            T.block_attr({"key": "value"})
            T.alloc_buffer((128, 128), "float32")
            T.match_buffer(e[0:32, 0:32], (32, 32), "float32")
            T.axis.spatial(128, f)
            T.evaluate(0)
    # the block generated by IRBuilder
    block_realize_actual = ib.get()

    # the expected block
    var_a = tir.Var("a", "int64")
    buffer_b = tir.decl_buffer((128, 128), "float32", name="b")
    buffer_c = tir.decl_buffer((128, 128), "float32", name="c")
    var_d = tir.Var("d", "int32")
    buffer_e = tir.decl_buffer((128, 128), "float32", name="c")
    var_f = tir.Var("f", "int32")
    block_expected = tir.Block(
        iter_vars=[tir.IterVar((0, 128), tir.Var("", "int32"), iter_type=tir.IterVar.DataPar)],
        reads=[buffer_b[0:16, 0:16]],
        writes=[buffer_c[var_d:128, var_d:128]],
        name_hint="block",
        body=tir.Evaluate(0),
        alloc_buffers=[tir.decl_buffer((128, 128), "float32")],
        match_buffers=[
            tir.MatchBufferRegion(tir.decl_buffer((32, 32), "float32"), buffer_e[0:32, 0:32])
        ],
        annotations={"key": "value"},
    )
    block_realize_expected = tir.BlockRealize(
        iter_values=[var_f],
        predicate=var_a > 1,
        block=block_expected,
    )
    # Check if the generated ir is expected
    assert_structural_equal(block_realize_actual, block_realize_expected, map_free_vars=True)


def test_ir_builder_tir_axis():
    with IRBuilder() as ib:
        a = T.var("int32", "a")
        b = T.var("int32", "b")
        c = T.var("int32", "c")
        d = T.var("int32", "d")
        with T.block("block"):
            T.axis.spatial(8, a)
            T.axis.reduce(16, b)
            T.axis.scan(32, c)
            T.axis.opaque(64, d)
            T.evaluate(0)

    # the block generated by IRBuilder
    block_realize_actual = ib.get()

    # the expected block
    var_a = tir.Var("a", "int32")
    var_b = tir.Var("b", "int32")
    var_c = tir.Var("c", "int32")
    var_d = tir.Var("d", "int32")
    block_expected = tir.Block(
        iter_vars=[
            tir.IterVar((0, 8), tir.Var("", "int32"), iter_type=tir.IterVar.DataPar),
            tir.IterVar((0, 16), tir.Var("", "int32"), iter_type=tir.IterVar.CommReduce),
            tir.IterVar((0, 32), tir.Var("", "int32"), iter_type=tir.IterVar.Ordered),
            tir.IterVar((0, 64), tir.Var("", "int32"), iter_type=tir.IterVar.DimInfo),
        ],
        reads=[],
        writes=[],
        name_hint="block",
        body=tir.Evaluate(0),
        annotations={"tir.script_parsing_detect_access": tir.IntImm("int64", 3)},
    )
    block_realize_expected = tir.BlockRealize(
        iter_values=[var_a, var_b, var_c, var_d],
        predicate=True,
        block=block_expected,
    )
    # Check if the generated ir is expected
    assert_structural_equal(block_realize_actual, block_realize_expected, map_free_vars=True)


def test_ir_builder_tir_for():
    with IRBuilder() as ib:
        with T.serial(128) as a:
            with T.parallel(64) as b:
                with T.vectorized(32) as c:
                    with T.unroll(16) as d:
                        with T.thread_binding(8, thread="threadIdx.x") as e:
                            T.evaluate(0)

    # the for generated by IRBuilder
    for_actual = ib.get()

    # the expected for
    thread_binding_expected = tir.For(
        loop_var=tir.Var("", "int32"),
        min_val=0,
        extent=8,
        kind=tir.ForKind.THREAD_BINDING,
        body=tir.Evaluate(0),
        thread_binding=tir.IterVar(
            None, tir.Var("", "int32"), tir.IterVar.ThreadIndex, "threadIdx.x"
        ),
    )
    unroll_expected = tir.For(
        loop_var=tir.Var("", "int32"),
        min_val=0,
        extent=16,
        kind=tir.ForKind.UNROLLED,
        body=thread_binding_expected,
    )
    vectorized_expected = tir.For(
        loop_var=tir.Var("", "int32"),
        min_val=0,
        extent=32,
        kind=tir.ForKind.VECTORIZED,
        body=unroll_expected,
    )
    parallel_expected = tir.For(
        loop_var=tir.Var("", "int32"),
        min_val=0,
        extent=64,
        kind=tir.ForKind.PARALLEL,
        body=vectorized_expected,
    )
    for_expected = tir.For(
        loop_var=tir.Var("", "int32"),
        min_val=0,
        extent=128,
        kind=tir.ForKind.SERIAL,
        body=parallel_expected,
    )
    # Check if the generated ir is expected
    assert_structural_equal(for_actual, for_expected, map_free_vars=True)


def test_ir_builder_tir_assert():
    with IRBuilder() as ib:
        with T.Assert(T.var("int32", name="a") == 0, message="a is 0"):
            T.evaluate(0)
    # the assert generated by IRBuilder
    assert_actual = ib.get()

    # the expected assert statement
    assert_expected = tir.AssertStmt(
        T.var("int32", name="a") == 0, tir.StringImm("a is 0"), tir.Evaluate(0)
    )
    # Check if the generated ir is expected
    assert_structural_equal(assert_actual, assert_expected, map_free_vars=True)


def test_ir_builder_tir_evaluate():
    with IRBuilder() as ib:
        T.evaluate(0)
    # the evaluate generated by IRBuilder
    eval_actual = ib.get()

    # the expected evaluate
    eval_expected = tir.Evaluate(0)
    # Check if the generated ir is expected
    assert_structural_equal(eval_actual, eval_expected, map_free_vars=True)


def test_ir_builder_tir_let():
    with IRBuilder() as ib:
        with T.let(T.var("int32", name="a"), tir.IntImm("int32", 2)):
            T.evaluate(0)
    # the let binding generated by IRBuilder
    let_actual = ib.get()

    # the expected Let statement
    let_expected = tir.LetStmt(T.var("int32", name="a"), tir.IntImm("int32", 2), tir.Evaluate(0))
    assert_structural_equal(let_actual, let_expected, map_free_vars=True)


def test_ir_builder_tir_realize():
    buffer_a = T.buffer_decl((128, 128), "float32")
    with IRBuilder() as ib:
        with T.realize(buffer_a[0:128, 0:128], "test_storage_scope", True):
            T.evaluate(0)
    realize_actual = ib.get()

    # the expected buffer realization
    buffer_realize = tir.BufferRealize(
        buffer_a, [tvm.ir.Range(0, 128), tvm.ir.Range(0, 128)], True, tir.Evaluate(0)
    )
    expected_realize = tir.AttrStmt(
        buffer_a, "realize_scope", tir.StringImm("test_storage_scope"), buffer_realize
    )
    assert_structural_equal(realize_actual, expected_realize, map_free_vars=True)


def test_ir_builder_tir_thread():
    with IRBuilder() as ib:
        with T.prim_func():
            brow = T.env_thread("blockIdx.y")
            with T.launch_thread(brow, 1):
                T.evaluate(0)
    ir_actual = ib.get()
    iter_var = tir.IterVar((0, 1), "v", iter_type=1, thread_tag="blockIdx.y")
    attr_stmt = tir.AttrStmt(iter_var, "thread_extent", 1, tir.Evaluate(0))
    func = tir.PrimFunc([], attr_stmt)
    assert_structural_equal(ir_actual, func, map_free_vars=True)


def test_ir_builder_tir_allocate():
    with IRBuilder() as ib:
        with T.allocate([10], "float32", scope="local"):
            T.evaluate(1)
    ir_actual = ib.get()
    buffer_var = tir.Var("v", tvm.ir.PointerType(tvm.ir.PrimType("float32"), "local"))
    ir_expected = tir.Allocate(
        buffer_var, "float32", [10], tvm.tir.const(1, "uint1"), tir.Evaluate(1)
    )
    assert_structural_equal(ir_actual, ir_expected, map_free_vars=True)


def test_ir_builder_tir_allocate_const():
    data = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    with IRBuilder() as ib:
        with T.allocate_const(data, "int32", [10]):
            T.evaluate(1)
    ir_actual = ib.get()
    buffer_var = tir.Var("v", tvm.ir.PointerType(tvm.ir.PrimType("int32")))
    ir_expected = tir.AllocateConst(
        buffer_var, "int32", [10], ndarray.array(np.asarray(data, "int32")), tir.Evaluate(1)
    )
    assert_structural_equal(ir_actual, ir_expected, map_free_vars=True)


def test_ir_builder_tir_prefetch():
    with IRBuilder() as ib:
        buffer_a = T.buffer_decl((128, 128), "float32")
        T.prefetch(buffer_a, [])
    ir_actual = ib.get()
    ir_expected = tir.Prefetch(buffer_a, [])
    assert_structural_equal(ir_actual, ir_expected, map_free_vars=True)


def test_ir_builder_tir_buffer_store():
    buffer_a = T.buffer_decl((10, 10), "float32")
    i = T.var("int32", "x")
    with IRBuilder() as ib:
        T.buffer_store(buffer_a, 0.1, [0, i])
    ir_actual = ib.get()
    ir_expected = tir.BufferStore(buffer_a, 0.1, [0, i])
    assert_structural_equal(ir_actual, ir_expected, map_free_vars=True)


def test_ir_builder_tir_while():
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with T.While(T.var("int32", "x") > 0):
            T.evaluate(0)
    ir_actual = ib.get()
    ir_expected = tir.While(tir.Var("x", "int32") > 0, tir.Evaluate(0))
    assert_structural_equal(ir_actual, ir_expected, map_free_vars=True)


def test_ir_builder_tir_if_then_else():
    with IRBuilder() as ib:
        with T.If(T.var("int32", "c") < 12):
            with T.Then():
                T.evaluate(T.int32(0))
            with T.Else():
                T.evaluate(T.int32(1))
    ir_actual = ib.get()
    ir_expected = tir.IfThenElse(
        tir.Var("c", "int32") < 12,
        tir.Evaluate(tir.IntImm("int32", 0)),
        tir.Evaluate(tir.IntImm("int32", 1)),
    )
    assert_structural_equal(ir_actual, ir_expected, map_free_vars=True)


if __name__ == "__main__":
    pytest.main([__file__])
