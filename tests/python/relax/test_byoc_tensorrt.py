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
from tvm.relax.expr import Call, Function, GlobalVar, ExternFunc


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
    # TODO
    # 1. we has Relax func, tir func, packed func, te func. How do we lower/raise to each other?
    #    1) Convert function
    #    2) Rewrite the caller
    # 2. How do we pass the generated runtime module?
    #    O1) Embed in IRModule (e.g., metadata?) -> requires serialization
    #    O2) Save in PassContext
    @tvm.ir.transform.module_pass(opt_level=0)
    class CodegenPass(transform.Pass):
        def transform_module(self, mod: IRModule, ctx: PassContext) -> IRModule:
            class MyMutator(ExprMutator):
                def __init__(self):
                    ExprMutator.__init__(self)

                def visit_call(self, call: Call):
                    print(f"Call: {call.op}, {type(call.op)}")

                    # TODO(sunggg): Currently, this seems necessary. Bug?
                    if isinstance(call.op, GlobalVar):
                        new_op = self.visit(mod.functions[call.op])
                        new_args = [self.visit(arg) for arg in call.args]
                        if isinstance(new_op, ExternFunc):
                            print(f"Call: {call.op}, {type(call.op)}")
                            print(f"{new_op} --> rewrite")
                        return Call(new_op, new_args)
                    else:
                        new_op = self.visit(call.op)
                        new_args = [self.visit(arg) for arg in call.args]

                        return Call(new_op, new_args, call.attrs, call.type_args, call.span)

                def visit_function(self, func: Function):
                    print(f"Function: {func}, {func.attrs}")
                    if "Codegen" in func.attrs:
                        codegen_name, new_sym = func.attrs["Codegen"], func.attrs["global_symbol"]
                        codegen = tvm.get_global_func(f"relax.ext.{codegen_name}", True)
                        assert codegen

                        # TODO(sunggg): How do we serialize this?
                        ext_lib = codegen(func)  # returns runtime::Module
                        print(ext_lib)
                        # Write an external function?

                        return ExternFunc(new_sym)

                    else:
                        new_params = [self.visit(param) for param in func.params]
                        new_body = self.visit(func.body)

                        return Function(new_params, new_body, func.ret_type, func.attrs, func.span)

            MyMutator().visit(mod["main"])
            return mod

    mod = InputModule
    assert isinstance(mod, tvm.IRModule)
    new_relax_mod = mod["relax_add"].with_attr("Codegen", "tensorrt")
    new_relax_mod = new_relax_mod.with_attr("global_symbol", "trt_relax_add")
    mod["relax_add"] = new_relax_mod
    mypass = CodegenPass()
    new_mod = mypass(mod)

    print(new_mod)

    assert 0
    target = "cuda"
    dev = tvm.device(target, 0)

    with transform.PassContext(opt_level=0):
        ex0 = relax.vm.build(new_mod, target, params={})

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
