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
import tvm
import tvm.testing

from tvm import relax
import numpy as np
from tvm.script import tir as T, relax as R
import time
from tvm import transform
from tvm.relax import ExprMutator
from tvm.relax.expr import Call, Function


@tvm.script.ir_module
class InputModule:
    @R.function
    def relax_add(x: Tensor((2, 3), "float32"), y: Tensor((2, 3), "float32")) -> Tensor:
        z1 = relax.add(x, y)
        z2 = relax.add(z1, z1)
        return z2

    @R.function
    def main(x: Tensor((2, 3), "float32"), y: Tensor((2, 3), "float32")) -> Tensor:
        lv0 = relax_add(x, y)
        return lv0


def test_codegen_pass():
    @tvm.ir.transform.module_pass(opt_level=0)
    class CodegenPass(transform.Pass):
        def transform_module(self, mod: IRModule, ctx: PassContext) -> IRModule:
            class MyMutator(ExprMutator):
                def __init__(self):
                    ExprMutator.__init__(self)

                def visit_call(self, call: Call):
                    print(call)

                def visit_function(self, func: Function):
                    print(func)

    mod = InputModule
    assert isinstance(mod, tvm.IRModule)
    new_relax_mod = mod["relax_add"].with_attr("Codegen", "tensorrt")
    new_relax_mod = new_relax_mod.with_attr("global_symbol", "trt_relax_add")

    new_mod = CodegenPass()(mod)


def test_extern_trt_relax():
    # TODO: handle const bindings
    # y1 = relax.const([[0.1, 1.1, 2.1], [3.1, 4.1, 5.1]])
    # y2 = relax.const(2.1, dtype="float32")
    # y3 = relax.const([[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]])
    mod = InputModule
    assert isinstance(mod, tvm.IRModule)
    new_relax_mod = mod["relax_add"].with_attr("Codegen", "tensorrt")
    new_relax_mod = new_relax_mod.with_attr("global_symbol", "trt_relax_add")
    mod["relax_add"] = new_relax_mod
    target = "cuda"
    dev = tvm.device(target, 0)
    with transform.PassContext(opt_level=0):
        ex0 = relax.vm.build(mod, target, params={})

    print("\n")
    print(ex0.as_python())
    print("\n")

    vm0 = relax.VirtualMachine(ex0, dev)
    data0 = tvm.nd.array(np.random.rand(2, 3).astype(np.float32), tvm.cpu())
    data1 = tvm.nd.array(np.random.rand(2, 3).astype(np.float32), tvm.cpu())
    # weight = tvm.nd.array(np.random.rand(32, 32).astype(np.float32), dev)

    # Measure the performance w/o tuning log
    tic = time.time()
    vm0["main"](data0, data1)
    toc = time.time()
    e0 = toc - tic
    print(f"w/o tuning: {e0}")


if __name__ == "__main__":
    test_codegen_pass()
    # test_extern_trt_hybrid()
    # test_extern_trt_relax()
