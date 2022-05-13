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
from tvm.ir import IRModule
from tvm.relax import ExprMutator
from tvm.relax.expr import Call, Function, GlobalVar, ExternFunc, Tuple

# TODO: Implement `RemoveUnused func`
# TODO: Test with other functions, ops


@tvm.ir.transform.module_pass(opt_level=0)
class CodegenPass(transform.Pass):
    def transform_module(self, mod: IRModule, ctx: PassContext) -> IRModule:
        ext_mods = []
        cached = dict()

        class MyMutator(ExprMutator):
            def __init__(self):
                ExprMutator.__init__(self)

            def visit_call(self, call: Call):
                if isinstance(call.op, GlobalVar):
                    # TODO: wait for the fix
                    func = mod.functions[call.op]
                    new_op = self.visit(func)
                    if isinstance(new_op, ExternFunc):
                        new_args = [self.visit(arg) for arg in call.args]
                        call_tir = tvm.relay.op.op.get("relax.call_tir")
                        return Call(
                            op=call_tir,
                            args=[new_op, Tuple(new_args), func.body.shape_],
                            attrs=None,
                            type_args=[func.ret_type],
                        )

                return call

            def visit_function(self, func: Function):
                if "Codegen" in func.attrs:
                    codegen_name, new_sym = func.attrs["Codegen"], func.attrs["global_symbol"]
                    if new_sym in cached:
                        return cached[new_sym]

                    codegen = tvm.get_global_func(f"relax.ext.{codegen_name}", True)
                    assert codegen
                    ext_mods.append(codegen(func))
                    extern_func = ExternFunc(new_sym)
                    cached[new_sym] = extern_func
                    return extern_func

                else:
                    new_params = [self.visit(param) for param in func.params]
                    new_body = self.visit(func.body)

                    return Function(new_params, new_body, func.ret_type, func.attrs, func.span)

        mod["main"] = MyMutator().visit(mod["main"])
        mod = mod.with_attr("external_mods", ext_mods)
        return mod


def test_single_annot_func():
    @tvm.script.ir_module
    class InputModule:
        @R.function
        def relax_func(x: Tensor((2, 3), "float32"), y: Tensor((2, 3), "float32")) -> Tensor:
            z1 = relax.add(x, y)
            z2 = relax.add(z1, z1)
            z3 = relax.add(z1, z2)
            return z3

        @R.function
        def main(x: Tensor((2, 3), "float32"), y: Tensor((2, 3), "float32")) -> Tensor:
            lv0 = relax_func(x, y)
            return lv0

    mod = InputModule
    assert isinstance(mod, tvm.IRModule)
    new_relax_mod = mod["relax_func"].with_attr("Codegen", "tensorrt")
    new_relax_mod = new_relax_mod.with_attr("global_symbol", "trt_relax_func")
    mod["relax_func"] = new_relax_mod
    mypass = CodegenPass()

    new_mod = mypass(mod)

    target = "cuda"
    dev = tvm.device(target, 0)
    print(mod)
    print(new_mod)

    with transform.PassContext(opt_level=0):
        ex0 = relax.vm.build(new_mod, target, params={})

    vm0 = relax.VirtualMachine(ex0, dev)
    np0 = np.random.rand(2, 3).astype(np.float32)
    np1 = np.random.rand(2, 3).astype(np.float32)
    data0 = tvm.nd.array(np0, tvm.cpu())
    data1 = tvm.nd.array(np1, tvm.cpu())

    # Measure the performance w/o tuning log
    out0 = vm0["main"](data0, data1)

    # Correct output: Current relax cannot lower relax.add
    # numpy baseline
    tmp = np0 + np1
    out1 = tmp + tmp
    out1 = out1 + tmp
    # out1 = out1 + np1
    tvm.testing.assert_allclose(out0.numpy(), out1)


def test_multiple_annot_funcs():
    @tvm.script.ir_module
    class InputModule:
        @R.function
        def relax_func1(x: Tensor((2, 3), "float32"), y: Tensor((2, 3), "float32")) -> Tensor:
            z1 = relax.add(x, y)
            z2 = relax.add(z1, z1)
            z3 = relax.add(z1, z2)
            return z3

        @R.function
        def relax_func2(x: Tensor((2, 3), "float32"), y: Tensor((2, 3), "float32")) -> Tensor:
            z1 = relax.add(x, y)
            z2 = relax.add(z1, z1)
            return z2

        @R.function
        def main(x: Tensor((2, 3), "float32"), y: Tensor((2, 3), "float32")) -> Tensor:
            lv0 = relax_func1(x, y)
            lv1 = relax_func2(lv0, y)
            return lv1

    mod = InputModule
    assert isinstance(mod, tvm.IRModule)
    new_relax_mod = mod["relax_func1"].with_attr("Codegen", "tensorrt")
    new_relax_mod = new_relax_mod.with_attr("global_symbol", "trt_relax_func1")
    new_relax_mod = mod["relax_func2"].with_attr("Codegen", "tensorrt")
    new_relax_mod = new_relax_mod.with_attr("global_symbol", "trt_relax_func2")
    mod["relax_func"] = new_relax_mod
    mypass = CodegenPass()

    new_mod = mypass(mod)

    target = "cuda"
    dev = tvm.device(target, 0)
    print(mod)
    print("====after====")
    print(new_mod)

    with transform.PassContext(opt_level=0):
        ex0 = relax.vm.build(new_mod, target, params={})

    vm0 = relax.VirtualMachine(ex0, dev)
    np0 = np.random.rand(2, 3).astype(np.float32)
    np1 = np.random.rand(2, 3).astype(np.float32)
    data0 = tvm.nd.array(np0, tvm.cpu())
    data1 = tvm.nd.array(np1, tvm.cpu())

    # Measure the performance w/o tuning log
    out0 = vm0["main"](data0, data1)

    # Correct output: Current relax cannot lower relax.add
    # numpy baseline
    tmp = np0 + np1
    out1 = tmp + tmp
    out1 = out1 + tmp
    # out1 = out1 + np1
    tvm.testing.assert_allclose(out0.numpy(), out1)


if __name__ == "__main__":
    test_single_annot_func()
    test_multiple_annot_funcs()
