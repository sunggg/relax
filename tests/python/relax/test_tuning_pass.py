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

from __future__ import annotations  # must import to defer parsing of annotations
import pytest
from typing import Callable, Union, Dict, List, Optional
from tvm.relax.transform.tuning import (
    default_generate_candidate,
    default_consider_eval_passes,
    default_evaluate,
    select_best_candidate,
    get_trace,
)
import tvm
from tvm import ir
from tvm.ir import transform
from tvm.ir.transform import PassContext
from tvm.ir.module import IRModule
from tvm.script import tir as T, relax as R
from tvm.relax.transform.tuning import Choice, Knob, Trace
from tvm import relax
from tvm.relax.expr import Expr, DataflowBlock, Function
import numpy as np


@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def addone(A: T.Buffer[(16, 16), "int32"], B: T.Buffer[(16, 16), "int32"]) -> None:
        for i, j in T.grid(16, 16):
            with T.block("addone"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj] + T.int32(1)

    # Input IRModule
    @R.function
    def main(c0: Tensor((16, 16), "int32")):
        lv0 = relax.call_tir(addone, (c0,), (16, 16), dtype="int32")
        return lv0


def setup_test_const_folding():
    @tvm.script.ir_module
    class TestModule:
        @T.prim_func
        def addone(A: T.Buffer[(16, 16), "int32"], B: T.Buffer[(16, 16), "int32"]) -> None:
            for i, j in T.grid(16, 16):
                with T.block("addone"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi, vj] + T.int32(1)

        # Input IRModule
        @R.function
        def before(c0: Tensor((16, 16), "int32")):
            lv0 = relax.call_tir(addone, (c0,), (16, 16), dtype="int32")
            return lv0

        # Expected IRModule after transformation
        @R.function
        def expected(c1: Tensor((16, 16), "int32")):
            lv0 = c1
            return c1

    def gen_mod(mod, name, binding):
        """Select relax function with name, rename to main and and bind constant.

        Parameters
        ----------
        mod: IRModule
            The input module

        name: str
            The name of relax function to preserve and rename to main

        binding: Dict[str, array]
            The const parameter bindings
        """
        funcs = {}
        binding = {k: tvm.nd.array(v) for k, v in binding.items()}

        for k, v in mod.functions.items():
            if isinstance(v, tvm.relax.Function):
                if k.name_hint == name:
                    # rename to main
                    gv = tvm.ir.GlobalVar("main")
                    funcs[gv] = tvm.relax.Function(v.params, v.body, v.ret_type, gv)
            else:
                funcs[k] = v
        mod = tvm.IRModule(funcs)
        return relax.transform.BindParams("main", binding)(mod)

    mod = TestModule
    assert isinstance(mod, tvm.IRModule)
    # Test setup
    c0_np = np.arange((16 * 16)).astype("int32").reshape(16, 16)
    c1_np = c0_np + 1
    before = gen_mod(mod, "before", {"c0": c0_np})
    expected = gen_mod(mod, "expected", {"c1": c1_np})

    return before, expected


# Mock evaluation pass for testing
# Assigns arbitrary performance number to each candidate
def mock_evaluate(candidates: List[Trace], target_str: str, ctx: PassContext):
    num_evals = 0
    # Evaluation
    for candidate in candidates:
        # If this candidate is already evaluated, skip the measurement
        if candidate.perf != -1:
            continue

        num_evals += 1
        mock_perf = 100 - (ctx.num_evals + num_evals)
        candidate.set_perf(mock_perf)
    ctx.inc_num_evals(num_evals)


@ir.transform.module_pass(opt_level=0, traceable=True)
class MockConstFoldingTuningPass(transform.Pass):
    def __init__(
        self,
        eval_passes: List[transform.Pass] = None,
        required: List[transform.Pass] = [],
        database=None,
    ):
        self.required = required
        self.eval_passes = eval_passes
        self.database = database

    def transform_module(self, mod: IRModule, ctx: PassContext) -> IRModule:
        trace = ctx.trace

        def apply(mod):
            return relax.transform.FoldConstant()(mod)

        def noapply(mod):
            return mod

        # Create mock choices for testing
        choices = {"apply": Choice(apply), "noapply": Choice(noapply)}
        # Tuning pass manages a set of transformation functions registered via knob.
        knob = Knob("MockTuningKnob", choices)

        candidates = default_generate_candidate(knob, trace, self.eval_passes)
        mock_evaluate(candidates, "llvm", ctx)
        best_trace = select_best_candidate(candidates)

        ctx.set_trace(best_trace)
        return best_trace.out_mod


def test_choice():
    # Test setup
    (
        before,
        expected,
    ) = setup_test_const_folding()

    # Define a choice by using FoldConstant pass
    # TODO(sunggg): This wrapper seems necessary. Figure out why.
    def apply(mod):
        return relax.transform.FoldConstant()(mod)

    choice = Choice(apply)

    # Load transformation function from the choice and apply it
    after = choice.get_transform_func()(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_knob():
    # Test setup
    before, expected = setup_test_const_folding()

    def apply(mod):
        return relax.transform.FoldConstant()(mod)

    def noapply(mod):
        return mod

    choices = [Choice(apply), Choice(noapply)]

    knob = Knob("TestKnob", choices)
    assert knob.verify(0)
    assert knob.verify(1)
    assert not knob.verify(3)
    after_apply = knob.apply(before, 0)
    after_noapply = knob.apply(before, 1)
    tvm.ir.assert_structural_equal(after_apply, expected)
    tvm.ir.assert_structural_equal(after_noapply, before)

    choices = {"apply": Choice(apply), "noapply": Choice(noapply)}
    knob = Knob("TestKnob", choices)
    assert knob.verify("apply")
    assert knob.verify("noapply")
    assert not knob.verify("INVLAID")

    after_apply = knob.apply(before, "apply")
    after_noapply = knob.apply(before, "noapply")
    tvm.ir.assert_structural_equal(after_apply, expected)
    tvm.ir.assert_structural_equal(after_noapply, before)


def test_trace():
    before, expected = setup_test_const_folding()

    def apply(mod):
        return relax.transform.FoldConstant()(mod)

    def noapply(mod):
        return mod

    choices = {"apply": Choice(apply), "noapply": Choice(noapply)}
    knob = Knob("TestKnob", choices)

    trace = Trace(before, [knob], ["noapply"])
    assert trace.size == 1
    out = trace.add(knob, "noapply")
    tvm.ir.assert_structural_equal(trace.in_mod, before)
    tvm.ir.assert_structural_equal(trace.out_mod, before)
    tvm.ir.assert_structural_equal(out, before)
    # Assume we assign random performance number
    trace.set_perf(100)
    assert trace.perf == 100

    assert trace.size == 2
    out = trace.add(knob, "apply")
    tvm.ir.assert_structural_equal(trace.in_mod, before)
    tvm.ir.assert_structural_equal(trace.out_mod, expected)
    tvm.ir.assert_structural_equal(out, expected)
    assert trace.size == 3
    # Should be initalized when new knob is applied
    assert trace.perf == -1


def test_trace_wrapper():
    mod = MyModule
    assert isinstance(mod, tvm.IRModule)
    assert isinstance(Trace(mod), Trace)
    assert isinstance(get_trace(mod), Trace)
    assert isinstance(get_trace(mod["main"]), Trace)
    assert isinstance(get_trace(mod["addone"]), Trace)


def test_default_functions():
    mod = MyModule
    assert isinstance(mod, tvm.IRModule)

    def apply(mod):
        return relax.transform.FoldConstant()(mod)

    def noapply(mod):
        return mod

    choices = {"apply": Choice(apply), "noapply": Choice(noapply)}
    knob = Knob("TestKnob", choices)
    trace = Trace(mod)

    with transform.PassContext():
        candidates = default_generate_candidate(knob, trace)
        assert len(candidates) == 2

        default_evaluate(candidates, "llvm")
        assert PassContext.current().num_evals == 2

        # Since these candidates are already evaluated, we should not evaluate them again
        default_evaluate(candidates, "llvm")
        assert PassContext.current().num_evals == 2


def test_pass_context():
    mod = MyModule
    assert isinstance(mod, tvm.IRModule)
    HeuristicPass = relax.transform.FoldConstant

    # Without binding
    seq = transform.Sequential([HeuristicPass()])
    with transform.PassContext(trace=Trace(mod)):
        _ = seq(mod)
        assert PassContext.current().trace.size == 1

    # Binding IRModule
    c0 = np.arange((16 * 16)).astype("int32").reshape(16, 16)
    mod = relax.transform.BindParams("main", {"c0": tvm.nd.array(c0)})(mod)

    # With binding, the heuristic pass implicitly performs TIR passes
    seq = transform.Sequential([HeuristicPass()])
    with transform.PassContext(trace=Trace(mod)):
        _ = seq(mod)
        assert PassContext.current().trace.size == 57

    # We can explicitly specify which pass we want to keep track of
    with transform.PassContext(trace=Trace(mod), make_traceable=["FoldConstant"]):
        _ = seq(mod)
        assert PassContext.current().trace.size == 1


def test_module_pass():
    mod = MyModule
    assert isinstance(mod, tvm.IRModule)
    # Test setup
    c0 = np.arange((16 * 16)).astype("int32").reshape(16, 16)
    mod = relax.transform.BindParams("main", {"c0": tvm.nd.array(c0)})(mod)
    HeuristicPass = relax.transform.FoldConstant

    mock_pass = MockConstFoldingTuningPass(eval_passes=[])
    with transform.PassContext(trace=Trace(mod), make_traceable=["FoldConstant"]):
        _ = mock_pass(mod)
        assert PassContext.current().num_evals == 2
        assert PassContext.current().trace.size == 1

    # Heuristic pass should not affect the number of candidates
    mock_pass = MockConstFoldingTuningPass(eval_passes=[HeuristicPass()])
    with transform.PassContext(trace=Trace(mod), make_traceable=["FoldConstant"]):
        _ = mock_pass(mod)
        assert PassContext.current().num_evals == 2
        assert PassContext.current().trace.size == 2

    # Joint-optimization will increase the search space in the combinatorial way
    mock_pass = MockConstFoldingTuningPass(eval_passes=[MockConstFoldingTuningPass(eval_passes=[])])
    with transform.PassContext(trace=Trace(mod), make_traceable=["FoldConstant"]):
        _ = mock_pass(mod)
        assert PassContext.current().num_evals == 2 * 2
        assert PassContext.current().trace.size == 2

    mock_pass = MockConstFoldingTuningPass(
        eval_passes=[
            MockConstFoldingTuningPass(eval_passes=[MockConstFoldingTuningPass(eval_passes=[])])
        ]
    )
    with transform.PassContext(trace=Trace(mod), make_traceable=["FoldConstant"]):
        _ = mock_pass(mod)
        assert PassContext.current().num_evals == 2 * 2 * 2
        assert PassContext.current().trace.size == 3

    mock_pass = MockConstFoldingTuningPass(
        eval_passes=[
            MockConstFoldingTuningPass(
                eval_passes=[
                    MockConstFoldingTuningPass(eval_passes=[HeuristicPass(), HeuristicPass()])
                ]
            )
        ]
    )
    with transform.PassContext(
        trace=Trace(mod), make_traceable=["MockConstFoldingTuningPass", "FoldConstant"]
    ):
        _ = mock_pass(mod)
        assert PassContext.current().num_evals == 2 * 2 * 2
        assert PassContext.current().trace.size == 5

    # Sequential passes will
    mock_pass = MockConstFoldingTuningPass(
        eval_passes=[
            MockConstFoldingTuningPass(eval_passes=[]),
            MockConstFoldingTuningPass(eval_passes=[]),
            MockConstFoldingTuningPass(eval_passes=[]),
        ]
    )
    with transform.PassContext(
        trace=Trace(mod), make_traceable=["MockConstFoldingTuningPass", "FoldConstant"]
    ):
        _ = mock_pass(mod)
        assert PassContext.current().num_evals == 2 * (2 + 2 + 2)
        assert PassContext.current().trace.size == 4


def test_sequential():
    mod = MyModule
    assert isinstance(mod, tvm.IRModule)
    # Test setup
    c0 = np.arange((16 * 16)).astype("int32").reshape(16, 16)
    mod = relax.transform.BindParams("main", {"c0": tvm.nd.array(c0)})(mod)
    HeuristicPass = relax.transform.FoldConstant

    # Sequential with a single tuning pass should behave same with a single pass
    seq = transform.Sequential([MockConstFoldingTuningPass(eval_passes=[])])
    with transform.PassContext(trace=Trace(mod), make_traceable=["FoldConstant"]):
        _ = seq(mod)
        assert PassContext.current().num_evals == 2
        assert PassContext.current().trace.size == 1

    # Sequential pass should increase search space (num_evals) in additive manner
    seq = transform.Sequential(
        [
            MockConstFoldingTuningPass(eval_passes=[]),
            MockConstFoldingTuningPass(eval_passes=[]),
            MockConstFoldingTuningPass(eval_passes=[]),
        ]
    )
    with transform.PassContext(trace=Trace(mod), make_traceable=["FoldConstant"]):
        _ = seq(mod)
        assert PassContext.current().num_evals == 2 + 2 + 2
        assert PassContext.current().trace.size == 3

    # Heuristic pass will not increase the search space. Just increase trace length
    seq = transform.Sequential(
        [
            MockConstFoldingTuningPass(eval_passes=[]),
            HeuristicPass(),
            MockConstFoldingTuningPass(eval_passes=[]),
            MockConstFoldingTuningPass(eval_passes=[]),
            HeuristicPass(),
        ]
    )

    with transform.PassContext(trace=Trace(mod), make_traceable=["FoldConstant"]):
        _ = seq(mod)
        assert PassContext.current().num_evals == 2 + 2 + 2
        assert PassContext.current().trace.size == 5

    seq = transform.Sequential(
        [
            HeuristicPass(),
            MockConstFoldingTuningPass(
                eval_passes=[
                    MockConstFoldingTuningPass(
                        eval_passes=[
                            MockConstFoldingTuningPass(
                                eval_passes=[
                                    HeuristicPass(),
                                ]
                            )
                        ]
                    ),
                ]
            ),
            MockConstFoldingTuningPass(eval_passes=[]),
            HeuristicPass(),
        ]
    )

    with transform.PassContext(trace=Trace(mod), make_traceable=["FoldConstant"]):
        _ = seq(mod)
        assert PassContext.current().num_evals == (2 * 2 * 2) + 2
        assert PassContext.current().trace.size == 7


def test_mixed_passes():
    @tvm.script.ir_module
    class MockModule:
        @R.function
        def f1(x: Tensor((m, n), "float32")):
            with relax.dataflow():
                lv0 = relax.multiply(x, x)
                gv0 = relax.add(x, x)
                relax.output(gv0)
            return gv0

        @R.function
        def main(x: Tensor((m, n), "float32"), y: Tensor((m, n), "float32")):
            with relax.dataflow():
                lv0 = relax.multiply(x, y)
                gv0 = relax.add(lv0, y)
                relax.output(gv0)
            gv1 = relax.multiply(x, y)
            gv2 = relax.add(gv1, y)
            return (gv0, gv1, gv2)

    mod = MockModule
    assert isinstance(mod, tvm.IRModule)

    def pass_func(
        mod: IRModule, ctx: PassContext, eval_passes: List[transform.Pass] = None
    ) -> IRModule:
        trace = ctx.trace

        def noapply(mod):
            return mod

        # Create mock choices for testing
        choices = [Choice(noapply), Choice(noapply), Choice(noapply)]
        # Tuning pass manages a set of transformation functions registered via knob.
        knob = Knob("MockTuningKnob", choices)

        candidates = default_generate_candidate(knob, trace, eval_passes)
        mock_evaluate(candidates, "llvm", ctx)
        best_trace = select_best_candidate(candidates)

        ctx.set_trace(best_trace)
        return best_trace.out_mod

    @ir.transform.module_pass(opt_level=0, traceable=True)
    def MockModulePass(mod: IRModule, ctx: PassContext) -> IRModule:
        return pass_func(mod, ctx)

    @relax.transform.function_pass(opt_level=0, traceable=True)
    def MockFunctionPass(expr: Expr, mod: IRModule, ctx: PassContext) -> Function:
        # Do something
        pass_func(mod, ctx)
        return expr

    @relax.transform.dataflowblock_pass(opt_level=0, traceable=True)
    def MockDataflowBlockPass(
        block: DataflowBlock, mod: IRModule, ctx: PassContext
    ) -> DataflowBlock:
        # Do something
        pass_func(mod, ctx)
        return block

    seq = transform.Sequential(
        [
            MockModulePass,
            MockFunctionPass,
            MockDataflowBlockPass,
        ]
    )

    with transform.PassContext(trace=Trace(mod), make_traceable=[]):
        _ = seq(mod)
        # Trace length and num eval can be different depending on how each function/dataflow block is treated


if __name__ == "__main__":
    test_choice()
    test_knob()
    test_trace()
    test_default_functions()
    test_trace_wrapper()
    test_pass_context()
    test_module_pass()
    test_sequential()
    test_mixed_passes()
