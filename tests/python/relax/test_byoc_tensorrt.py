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

from tvm import relax, relay
import numpy as np
from tvm.script import tir as T, relax as R
import time
from tvm import transform
from tvm.target.target import Target
from tvm import te


def test_extern_trt_hybrid():
    dtype = "float32"
    ishape = (32, 256, 56, 56)
    w1shape = (128, 256, 1, 1)
    outshape = (32, 128, 28, 28)

    data0 = relay.var("x", shape=(ishape), dtype=dtype)
    weight0 = relay.var("w", shape=(w1shape), dtype=dtype)
    data = tvm.nd.array(np.random.uniform(0, 10, ishape).astype(np.float32), tvm.cpu())
    weight = tvm.nd.array(np.random.uniform(0, 1, w1shape).astype(np.float32), tvm.cpu())
    z = tvm.nd.array(np.random.uniform(0, 1, outshape).astype(np.float32), tvm.cpu())
    out = relay.nn.conv2d(data0, weight0, strides=(2, 2), padding=(0, 0, 0, 0))

    f = relay.Function([data0, weight0], out)
    f = relay.build_module.bind_params_by_name(f, {"w": weight})

    ref_mod = tvm.IRModule.from_expr(f)
    f = f.with_attr("Compiler", "tensorrt")
    f = f.with_attr("global_symbol", "default")
    f = f.with_attr("Composite", "test-composite")
    trt_engine = tvm.get_global_func("relay.ext.tensorrt", True)

    mod = tvm.IRModule.from_expr(f)
    mod = relay.transform.InferType()(mod)
    trt_lib = trt_engine(mod["default"])

    @tvm.script.ir_module
    class InputModule:
        @R.function
        def main(
            x: Tensor((32, 256, 56, 56), "float32"),
            z: Tensor((32, 128, 28, 28), "float32"),
        ) -> Tensor:
            with R.dataflow():
                lv0 = R.call_packed("default", x, z)
                R.output(lv0)
            return lv0

    mod = InputModule
    assert isinstance(mod, tvm.IRModule)
    mod = relax.transform.BindParams("main", {"default_const_0": weight})(mod)

    target = "cuda"
    dev = tvm.device(target, 0)
    params = {"default_const_0": weight}
    with transform.PassContext(opt_level=3):
        ex0 = relax.vm.build(mod, target, [trt_lib], params=params)

    vm0 = relax.VirtualMachine(ex0, dev)

    # Measure the performance w/o tuning log
    tic = time.time()
    vm0["main"](data, z)
    toc = time.time()
    e0 = toc - tic
    print(f"w/o tuning: {e0}")

    lib = tvm.relay.backend.vm.compile(ref_mod, target=target, params=params)
    exe = tvm.runtime.vm.VirtualMachine(lib, dev)

    exe.set_input("main", data)
    output = exe.invoke("main")

    tvm.testing.assert_allclose(z.numpy(), output.numpy())


def test_extern_trt_relax():
    # TODO: handle const bindings
    # y1 = relax.const([[0.1, 1.1, 2.1], [3.1, 4.1, 5.1]])
    # y2 = relax.const(2.1, dtype="float32")
    # y3 = relax.const([[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]])
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
    # test_extern_trt_hybrid()
    test_extern_trt_relax()
