from __future__ import annotations
import tvm
from tvm import relax, relay
import numpy as np
from tvm.script import tir as T, relax as R
import time
from tvm import transform
from tvm.target.target import Target
from tvm import te


def test_extern_trt_relay():
    dtype = "float32"
    ishape = (1, 32, 14, 14)
    w1shape = (32, 1, 3, 3)
    data0 = relay.var("data0", shape=(ishape), dtype=dtype)
    weight0 = relay.var("weight0", shape=(w1shape), dtype=dtype)

    data1 = relay.var("data0", shape=(ishape), dtype=dtype)
    weight1 = relay.var("weight0", shape=(w1shape), dtype=dtype)
    weight2 = relay.var("weight1", shape=(w1shape), dtype=dtype)
    depthwise_conv2d_1 = relay.nn.conv2d(
        data1, weight1, kernel_size=(3, 3), padding=(1, 1), groups=32
    )
    depthwise_conv2d_2 = relay.nn.conv2d(
        depthwise_conv2d_1, weight2, kernel_size=(3, 3), padding=(1, 1), groups=32
    )
    out = relay.add(depthwise_conv2d_1, depthwise_conv2d_2)

    f = relay.Function([data1, weight1, weight2], out)

    # f = set_external_func_attr(f, "tenssorrt", "tensorrt_0")
    f = f.with_attr("Compiler", "tensorrt")
    f = f.with_attr("Composite", "test-composite")
    f = f.with_attr("PartitionedFromPattern", "test-composite")
    call = relay.Call(f, [data0, weight0, weight0])
    mod = tvm.IRModule.from_expr(f)
    mod = relay.transform.InferType()(mod)

    target = tvm.target.Target("cuda")
    dev = tvm.device("cuda", 0)

    """
    te_data = te.placeholder(ishape, name="data", dtype=dtype)
    output_shape = ishape
    cuDNN_OP = te.extern(
        output_shape,
        [te_data],
        lambda ins, outs: tvm.tir.call_packed(
            "tvm.contrib.cudnn.activation.forward",
            ins[0],  # x
            outs[0],  # y,
            1,
            0,  # alpha, beta
            1,
            0,
            1e100,
        ),
        name="y",
    )
    s = te.create_schedule(cuDNN_OP.op)
    lib = tvm.build(s, [te_data, cuDNN_OP], "cuda -libs=cudnn", target_host="llvm")
    """
    # with tvm.transform.PassContext(opt_level=0):
    #    rt_mod = relay.build(mod, target)
    # assert 0

    trt_engine = tvm.get_global_func("relay.ext.tensorrt", True)
    extern_lib = trt_engine(mod["main"])
    # Assume everything is passed to BYOC
    # Create a virtual main module to make sure a DSO module will be also available.
    lib = tvm.get_global_func("runtime.CSourceModuleCreate", True)(";", "", [], [])

    runtime = relay.backend.Runtime("cpp", {"system-lib": True})

    i_data = np.random.uniform(0, 1, ishape).astype(dtype)
    w_data = np.random.uniform(0, 1, w1shape).astype(dtype)
    # relay.create_executor("vm", device=dev, target=target, mod=lib).evaluate()(i_data, w_data)

    # create_metadata_module = tvm.get_global_func("runtime.CreateLLVMCrtMetadataModule", True)
    # lib = create_metadata_module([lib], target, runtime)
    # print(dir(lib))
    # print(lib.get_function("TVMSystemLibEntryPoint"))

    # graph_mod = tvm.contrib.graph_executor.GraphModule(lib["TVMSystemLibEntryPoint"](tvm.cuda()))
    # i_data = np.random.uniform(0, 1, ishape).astype(dtype)
    # w_data = np.random.uniform(0, 1, w1shape).astype(dtype)
    # rt_mod(i_data, w_data)
    assert 0

    # i_data = np.random.uniform(0, 1, ishape).astype(dtype)
    # w_data = np.random.uniform(0, 1, w1shape).astype(dtype)
    # rt_mod(i_data, w_data)


def test_extern_trt_hybrid():
    dtype = "float32"
    ishape = (32, 256, 56, 56)
    w1shape = (128, 256, 1, 1)
    outshape = (32, 128, 28, 28)
    data0 = relay.var("x", shape=(ishape), dtype=dtype)
    weight0 = relay.var("w", shape=(w1shape), dtype=dtype)

    out = relay.nn.conv2d(data0, weight0, strides=(2, 2), padding=(0, 0, 0, 0))

    f = relay.Function([data0, weight0], out)

    f = f.with_attr("Compiler", "tensorrt")
    f = f.with_attr("Composite", "test-composite")
    f = f.with_attr("PartitionedFromPattern", "test-composite")

    target = "cuda"
    dev = tvm.device(target, 0)
    data = tvm.nd.array(np.random.uniform(0, 1, ishape).astype(np.float32), dev)
    weight = tvm.nd.array(np.random.uniform(0, 1, w1shape).astype(np.float32), dev)
    z = tvm.nd.array(np.random.uniform(0, 1, outshape).astype(np.float32), dev)
    # mod = relax.transform.BindParams("main", {"x": data})(mod)
    # mod = relax.transform.BindParams("main", {"w": weight})(mod)

    trt_engine = tvm.get_global_func("relay.ext.tensorrt", True)

    f = relay.build_module.bind_params_by_name(f, {"w": weight})
    mod = tvm.IRModule.from_expr(f)
    print(mod)
    mod = relay.transform.InferType()(mod)
    print(mod)
    trt_lib = trt_engine(mod["main"])

    @tvm.script.ir_module
    class InputModule:
        @R.function
        def main(
            x: Tensor[(32, 256, 56, 56), "float32"],
            # w: Tensor[(128, 256, 1, 1), "float32"],
            z: Tensor[(32, 128, 28, 28), "float32"],
        ) -> Tensor:
            with R.dataflow():
                lv0 = R.call_packed("default", x, w, z)
                # lv0 = R.call_packed("default", x, w, z)
                R.output(lv0)
            return lv0

    mod = InputModule
    assert isinstance(mod, tvm.IRModule)

    with transform.PassContext(opt_level=3):
        ex0 = relax.vm.build(mod, target, [trt_lib])

    vm0 = relax.VirtualMachine(ex0, dev)

    # Measure the performance w/o tuning log
    tic = time.time()
    # vm0["main"](data, weight, z)
    vm0["main"](data, z)
    toc = time.time()
    e0 = toc - tic
    print(f"w/o tuning: {e0}")

    # graph_mod = tvm.contrib.graph_executor.GraphModule(lib["default"](tvm.cuda()))
    # i_data = np.random.uniform(0, 1, ishape).astype(dtype)
    # w_data = np.random.uniform(0, 1, w1shape).astype(dtype)
    # rt_mod(i_data, w_data)
    assert 0

    # i_data = np.random.uniform(0, 1, ishape).astype(dtype)
    # w_data = np.random.uniform(0, 1, w1shape).astype(dtype)
    # rt_mod(i_data, w_data)


def test_extern_trt_relax():
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def tir_matmul(x: T.handle, y: T.handle, z: T.handle) -> None:
            T.func_attr({"global_symbol": "tir_matmul"})
            m = T.var("int32")
            n = T.var("int32")
            k = T.var("int32")
            A = T.match_buffer(x, (32, 32))
            B = T.match_buffer(y, (32, 32))
            C = T.match_buffer(z, (32, 32))

            for (i0, j0, k0) in T.grid(32, 32, 32):
                with T.block():
                    i, j, k = T.axis.remap("SSR", [i0, j0, k0])
                    with T.init():
                        C[i, j] = 0.0
                    C[i, j] += A[i, k] * B[j, k]

        @R.function
        def relax_matmul(x: Tensor[(32, 32), "float32"], w: Tensor[(32, 32), "float32"]) -> Tensor:
            with R.dataflow():
                lv0 = R.call_tir(tir_matmul, (x, w), (32, 32), dtype="float32")
                R.output(lv0)
            return lv0

        @R.function
        def main(x: Tensor[(32, 32), "float32"], w: Tensor[(32, 32), "float32"]) -> Tensor:
            with R.dataflow():
                lv0 = relax_matmul(x, w)
                R.output(lv0)
            return lv0

    mod = InputModule
    assert isinstance(mod, tvm.IRModule)

    target = "llvm"
    dev = tvm.device(target, 0)

    with transform.PassContext(opt_level=0):
        ex0 = relax.vm.build(mod, target)

    vm0 = relax.VirtualMachine(ex0, dev)
    data = tvm.nd.array(np.random.rand(32, 32).astype(np.float32), dev)
    weight = tvm.nd.array(np.random.rand(32, 32).astype(np.float32), dev)

    # Measure the performance w/o tuning log
    tic = time.time()
    vm0["main"](data, weight)
    toc = time.time()
    e0 = toc - tic
    print(f"w/o tuning: {e0}")


def test_extern_llvm():
    dtype = "float32"
    ishape = (1, 32, 14, 14)
    wshape = (32, 1, 3, 3)
    data = relay.var("data", shape=(ishape), dtype=dtype)
    weight = relay.var("weight", shape=(wshape), dtype=dtype)
    conv2d = relay.nn.conv2d(data, weight, kernel_size=(3, 3), padding=(1, 1), groups=32)
    mod = tvm.IRModule.from_expr(conv2d)

    dev = tvm.cpu()
    libname = "test.so"

    @tvm._ffi.register_func("random.codegen")
    def codegen(mod, libname, lv):
        target = f"llvm --opt-level={lv}"
        with tvm.transform.PassContext(opt_level=0):
            rt_mod = relay.build(mod, target)
        rt_mod.export_library(libname)

    # Imagine these three lines happen in build
    extern_engine = tvm.get_global_func("random.codegen")
    extern_engine(mod, libname, 3)
    lib = tvm.runtime.load_module(libname)

    rt_mod = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    i_data = np.random.uniform(0, 1, ishape).astype(dtype)
    w_data = np.random.uniform(0, 1, wshape).astype(dtype)
    rt_mod.set_input("data", tvm.nd.array(i_data))
    rt_mod.set_input("weight", tvm.nd.array(w_data))

    ftimer = rt_mod.module.time_evaluator("run", dev, number=20, repeat=20)
    perfs = np.array(ftimer().results) * 1000
    print(f"codegen: {np.mean(perfs):.5f}ms")


if __name__ == "__main__":
    # test_extern_llvm()
    # test_extern_trt_relay()
    test_extern_trt_hybrid()
    # test_extern_trt_relax()
