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

from __future__ import annotations
import pytest
import numpy as np
import tempfile
import tvm
import tvm.script
import tvm.testing
from tvm import relax
from tvm.target import Target
from tvm.relax.testing import transform
from tvm.script import relax as R
from tvm import meta_schedule as ms
from tvm.relax.transform.tuning_api import Trace


def build_and_run(mod, target, dev, np_inputs):
    inputs = [tvm.nd.array(np_input, dev) for np_input in np_inputs]
    with tempfile.TemporaryDirectory() as work_dir:
        ex = ms.tune_relax(
            mod=mod,
            target=target,
            config=ms.TuneConfig(
                strategy="evolutionary",
                task_scheduler="round_robin",
                num_trials_per_iter=20,
                max_trials_per_task=20,
                max_trials_global=20,
            ),
            work_dir=work_dir,
        )
    vm = relax.VirtualMachine(ex, dev)
    vm["main"](*inputs)


def _test_lowering(target, dev):
    @tvm.script.ir_module
    class InputModule:
        @R.function
        def main(
            x: Tensor((16, 16), "float32"), w: Tensor((16, 16), "float32")
        ) -> Tensor((16, 16), "float32"):
            gv0 = R.multiply(x, w)
            gv1 = R.add(x, gv0)
            return gv1

    mod = InputModule
    assert mod
    with tvm.transform.PassContext(opt_level=3):
        out_mod = transform.LowerWithRelayOpStrategyPass(target, target_attrs={})(mod)

    input_shape = (16, 16)
    np_inputs = [
        np.random.rand(*input_shape).astype(np.float32),
        np.random.rand(*input_shape).astype(np.float32),
    ]
    build_and_run(out_mod, target, dev, np_inputs)


def test_lowering_cpu(target_str="llvm --num-cores=16"):
    _test_lowering(Target(target_str), tvm.cpu())


@tvm.testing.requires_gpu
def test_lowering_gpu(target_str="nvidia/nvidia-t4"):
    _test_lowering(Target(target_str), tvm.cuda())


def _test_partial_lowering(target, dev):
    @tvm.script.ir_module
    class InputModule:
        @R.function
        def should_not_lower(
            x: Tensor((16, 16), "float32"), w: Tensor((16, 16), "float32")
        ) -> Tensor((16, 16), "float32"):
            gv0 = R.multiply(x, w)
            gv1 = R.add(x, gv0)
            return gv1

        @R.function
        def should_lower(
            x: Tensor((16, 16), "float32"), w: Tensor((16, 16), "float32")
        ) -> Tensor((16, 16), "float32"):
            gv0 = R.multiply(x, w)
            gv1 = R.add(x, gv0)
            return gv1

    mod = InputModule
    assert mod
    # TODO(@sunggg): Revisit when TVMScript supports annotation.
    # Annotate target function.
    mod["should_lower"] = mod["should_lower"].with_attr("do_lower", "True")

    with tvm.transform.PassContext(opt_level=3):
        out_mod = transform.LowerWithRelayOpStrategyPass(target, target_attrs={"do_lower": "True"})(
            mod
        )
    print(out_mod)


def _test_incremental_lowering(target, dev):
    @tvm.script.ir_module
    class InputModule:
        @R.function
        def lower_first(
            x: Tensor((16, 16), "float32"), w: Tensor((16, 16), "float32")
        ) -> Tensor((16, 16), "float32"):
            gv0 = R.multiply(x, w)
            gv1 = R.add(x, gv0)
            return gv1

        @R.function
        def lower_second_0(
            x: Tensor((16, 16), "float32"), w: Tensor((16, 16), "float32")
        ) -> Tensor((16, 16), "float32"):
            gv0 = R.multiply(x, w)
            gv1 = R.add(x, gv0)
            return gv1

        @R.function
        def lower_second_1(
            x: Tensor((16, 16), "float32"), w: Tensor((16, 16), "float32")
        ) -> Tensor((16, 16), "float32"):
            gv0 = R.multiply(x, w)
            gv1 = R.add(x, gv0)
            return gv1

        @R.function
        def lower_third(
            x: Tensor((16, 16), "float32"), w: Tensor((16, 16), "float32")
        ) -> Tensor((16, 16), "float32"):
            gv0 = R.multiply(x, w)
            gv1 = R.add(x, gv0)
            return gv1

    mod = InputModule
    assert mod
    # TODO(@sunggg): Revisit when TVMScript supports annotation.
    # Annotate target function.
    mod["lower_first"] = mod["lower_first"].with_attr("lowering_priority", "high")
    mod["lower_second_0"] = mod["lower_second_0"].with_attr("lowering_priority", "mid")
    mod["lower_second_1"] = mod["lower_second_1"].with_attr("lowering_priority", "mid")
    mod["lower_third"] = mod["lower_third"].with_attr("lowering_priority", "low")

    with tvm.transform.PassContext(opt_level=3):
        out_mod1 = transform.LowerWithRelayOpStrategyPass(
            target, target_attrs={"lowering_priority": "high"}
        )(mod)

    print("[ Incremental Lowering (1/3) ]")
    print(out_mod1)
    print("\n")

    with tvm.transform.PassContext(opt_level=3):
        out_mod2 = transform.LowerWithRelayOpStrategyPass(
            target, target_attrs={"lowering_priority": "mid"}
        )(out_mod1)
    print("[ Incremental Lowering (2/3) ]")
    print(out_mod2)
    print("\n")

    # Without annotation, pass becomes target-agnostic.
    # Should perform lowering for the rest of relax functions regardless of annotation.
    with tvm.transform.PassContext(opt_level=3):
        out_mod3 = transform.LowerWithRelayOpStrategyPass(target)(out_mod2)

    print("[ Incremental Lowering (3/3) ]")
    print(out_mod3)
    print("\n")


def _test_incremental_lowering_with_byoc(target, dev):
    @tvm.script.ir_module
    class InputModule:
        @R.function
        def byoc_func(
            x: Tensor((16, 16), "float32"), w: Tensor((16, 16), "float32")
        ) -> Tensor((16, 16), "float32"):
            gv0 = R.multiply(x, w)
            gv1 = R.add(x, gv0)
            return gv1

        @R.function
        def tvm_func0(
            x: Tensor((16, 16), "float32"), w: Tensor((16, 16), "float32")
        ) -> Tensor((16, 16), "float32"):
            gv0 = R.multiply(x, w)
            gv1 = R.add(x, gv0)
            return gv1

        @R.function
        def tvm_func1(
            x: Tensor((16, 16), "float32"), w: Tensor((16, 16), "float32")
        ) -> Tensor((16, 16), "float32"):
            gv0 = R.multiply(x, w)
            gv1 = R.add(x, gv0)
            return gv1

        @R.function
        def main(x: Tensor((16, 16), "float32"), y: Tensor((16, 16), "float32")) -> Tensor:
            lv0 = byoc_func(x, y)
            lv1 = tvm_func0(x, lv0)
            lv2 = tvm_func1(x, lv1)
            return lv2

    mod = InputModule
    assert mod
    # TODO(@sunggg): Revisit when TVMScript supports annotation.
    # Annotate target function.
    new_byoc_func = mod["byoc_func"].with_attr("Codegen", "tensorrt")
    new_byoc_func = new_byoc_func.with_attr("global_symbol", "trt_relax_func")
    mod["byoc_func"] = new_byoc_func

    with tvm.transform.PassContext(opt_level=3):
        # Run Codegen pass
        seq = tvm.transform.Sequential(
            [relax.transform.RunCodegen(), relax.transform.RemoveUnusedFunctions()]
        )
        out_mod1 = seq(mod)
        print("[ Incremental Lowering (1/2) ]")
        print(out_mod1)
        print("\n")

        out_mod2 = transform.LowerWithRelayOpStrategyPass(target)(out_mod1)
        print("[ Incremental Lowering (2/2) ]")
        print(out_mod2)
        print("\n")

    with tvm.transform.PassContext(trace=Trace(out_mod2), opt_level=3):
        # To run TIR func, MS tuning is mendatory for now.
        with tempfile.TemporaryDirectory() as work_dir:
            config = ms.TuneConfig(
                strategy="evolutionary",
                num_trials_per_iter=2,
                max_trials_per_task=4,
                max_trials_global=4,
            )
            seq = tvm.transform.Sequential(
                [relax.transform.MetaScheduleTuneTIR(target, config, work_dir)]
            )
            out_mod3 = seq(out_mod2)

    """
    # TODO: There is a bug in running tvm-gen and byoc-gen codes together 
    with transform.PassContext(opt_level=0):
        ex = relax.vm.build(out_mod3, target, params={})
    np0 = np.random.rand(2, 3).astype(np.float32)
    np1 = np.random.rand(2, 3).astype(np.float32)
    data0 = tvm.nd.array(np0, tvm.cpu())
    data1 = tvm.nd.array(np1, tvm.cpu())

    vm = relax.VirtualMachine(ex, dev)
    out = vm["main"](data0, data1)
    """


def test_partial_lowering_cpu(target_str="llvm --num-cores=16"):
    _test_partial_lowering(Target(target_str), tvm.cpu())


def test_incremental_lowering_cpu(target_str="llvm --num-cores=16"):
    _test_incremental_lowering(Target(target_str), tvm.cpu())


# def test_incremental_lowering__with_byoc_gpu(target_str="nvidia/nvidia-t4"):
def test_incremental_lowering__with_byoc_gpu(target_str="nvidia/geforce-rtx-3070"):
    _test_incremental_lowering_with_byoc(Target(target_str), tvm.cuda())


if __name__ == "__main__":
    # test_partial_lowering_cpu()
    # test_incremental_lowering_cpu()
    test_incremental_lowering__with_byoc_gpu()
    # pytest.main([__file__])
