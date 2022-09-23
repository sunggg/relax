from __future__ import annotations
import numpy as np
import tvm
from tvm import relax
import tvm.relay.testing
import subprocess
from tvm.script import relax as R
import tempfile
from tvm.relax.testing import transform
from tvm.relax.testing.onnx_converter import RelaxToOnnxVisitor
from tvm.relax.transform.tuning_api import Trace
from utils import tune_config, gen_ground_truth, check_executable

# Target and device configs
target_str = "llvm --num-cores=16"
target = tvm.target.Target(target_str)
dev = tvm.device(target_str, 0)

######################################################################
# Register BYOC codegen & runtime by using Python interface
######################################################################
# Usecase 1: ONNX Runtime
# Codegen flow:
#    1. Relax -> ONNX: ONNX converter
#    2. build runtime around ONNX model: ONNXRuntime
#    3. Wrap BYOC runtime with TVM runtime module: relax.CreatePyExtRuntime
@tvm.register_func("relax.ext.onnx.runtime")
def onnx_runtime(onnx_path, num_args, *args):
    import onnxruntime

    session = onnxruntime.InferenceSession(
        onnx_path, providers=onnxruntime.get_available_providers()
    )
    input_names = session.get_inputs()
    inputs = {input_names[i].name: args[i].asnumpy() for i in range(num_args.value)}
    outs = session.run(None, inputs)[0]
    return tvm.nd.array(outs, dev)


@tvm.register_func("relax.ext.onnx")
def onnx_codegen(func: relax.Function):
    # Extract info from relax function
    assert func.attrs and "global_symbol" in func.attrs
    global_symbol = func.attrs["global_symbol"]
    num_inputs, num_outputs = len(func.params), 1

    # Convert relax to ONNX
    onnx_path = "relax.onnx"
    RelaxToOnnxVisitor().convert(func, onnx_path)

    # Build BYOC runtime and wrap with TVM runtime::Module
    gen_rt_mod = tvm.get_global_func("relax.CreatePyExtRuntime")
    assert gen_rt_mod
    return gen_rt_mod(global_symbol, onnx_path, "relax.ext.onnx.runtime", num_inputs, num_outputs)


# [Usecase 2] MLIR
# Codegen flow:
#    1. Relax -> ONNX: ONNX converter
#    2. ONNX -> MLIR ONNX dialect -> MLIR LLVM ONNX dialect -> shared library: ONNX-MLIR toolchain
#    3. build runtime around shared library: PyRuntime from ONNX-MLIR toolchain
#    4. Wrap BYOC runtime with TVM runtime module: relax.CreatePyExtRuntime
@tvm.register_func("relax.ext.mlir.runtime")
def mlir_runtime(shared_lib_path, num_args, *args):
    from PyRuntime import ExecutionSession

    inputs = [args[i].asnumpy() for i in range(num_args.value)]
    session = ExecutionSession(shared_lib_path=shared_lib_path)
    outs = session.run(inputs)
    return tvm.nd.array(outs, dev)


@tvm.register_func("relax.ext.mlir")
def mlir_codegen(func: relax.Function):
    # Extract info from relax function
    assert func.attrs and "global_symbol" in func.attrs
    global_symbol = func.attrs["global_symbol"]
    num_inputs, num_outputs = len(func.params), 1

    # Convert relax to ONNX
    onnx_path = "relax.onnx"
    RelaxToOnnxVisitor().convert(func, onnx_path)
    shared_lib_path = onnx_path[:-5] + ".so"

    # Generate shared library
    cmd = f"/home/spark/onnx-mlir/build/Debug/bin/onnx-mlir --EmitLib {onnx_path}".split(" ")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process.communicate()

    # Build BYOC runtime and wrap with TVM runtime::Module
    gen_rt_mod = tvm.get_global_func("relax.CreatePyExtRuntime")
    assert gen_rt_mod
    return gen_rt_mod(
        global_symbol, shared_lib_path, "relax.ext.mlir.runtime", num_inputs, num_outputs
    )


######################################################################
# Test cases
######################################################################


def test_single_annot_func(byoc_backend="mlir"):
    @tvm.script.ir_module
    class InputModule:
        @R.function
        def relax_func(
            x: Tensor((16, 16), "float32"), y: Tensor((16, 16), "float32")
        ) -> Tensor((16, 16), "float32"):
            z1 = relax.multiply(x, y)
            z2 = relax.add(z1, z1)
            z3 = relax.add(z1, z2)
            return z3

        @R.function
        def main(
            x: Tensor((16, 16), "float32"), y: Tensor((16, 16), "float32")
        ) -> Tensor((16, 16), "float32"):
            lv0: Tensor((16, 16), "float32") = relax_func(x, y)
            return lv0

    mod = InputModule
    assert isinstance(mod, tvm.IRModule)

    # Set up runtime inputs
    np0 = np.random.rand(16, 16).astype(np.float32)
    np1 = np.random.rand(16, 16).astype(np.float32)
    data0 = tvm.nd.array(np0, dev)
    data1 = tvm.nd.array(np1, dev)
    inputs = [data0, data1]

    # Generate expected results
    expected = gen_ground_truth(mod, target, dev, inputs)

    # Annotate target function.
    # This won't be necessary when TVMScript supports annotation.
    new_relax_func = mod["relax_func"].with_attr("Codegen", byoc_backend)
    mod["relax_func"] = new_relax_func

    # Run Codegen pass.
    seq = tvm.transform.Sequential(
        [relax.transform.RunCodegen(), relax.transform.RemoveUnusedFunctions()]
    )
    new_mod = seq(mod)
    assert relax.analysis.well_formed(new_mod)

    # Generate VM executable.
    with tvm.transform.PassContext(opt_level=0):
        ex0 = relax.vm.build(new_mod, target, params={})

    # Check the results with the exepcted answer.
    check_executable(ex0, dev, inputs, expected)


def test_mix_use_mlir_and_tvm(byoc_backend="mlir"):
    @tvm.script.ir_module
    class InputModule:
        @R.function
        def byoc_func(
            x: Tensor((16, 16), "float32"), y: Tensor((16, 16), "float32")
        ) -> Tensor((16, 16), "float32"):
            z1 = relax.multiply(x, y)
            z2 = relax.add(z1, z1)
            z3 = relax.add(z1, z2)
            return z3

        @R.function
        def tvm_func(
            x: Tensor((16, 16), "float32"), w: Tensor((16, 16), "float32")
        ) -> Tensor((16, 16), "float32"):
            gv0 = R.multiply(x, w)
            gv1 = R.add(x, gv0)
            return gv1

        @R.function
        def main(
            x: Tensor((16, 16), "float32"), y: Tensor((16, 16), "float32")
        ) -> Tensor((16, 16), "float32"):
            lv0 = byoc_func(x, y)
            lv1 = tvm_func(x, lv0)
            return lv1

    # Prepare IRModule and its inputs
    mod = InputModule
    assert isinstance(mod, tvm.IRModule)

    # Set up runtime inputs
    np0 = np.random.rand(16, 16).astype(np.float32)
    np1 = np.random.rand(16, 16).astype(np.float32)
    data0 = tvm.nd.array(np0, dev)
    data1 = tvm.nd.array(np1, dev)
    inputs = [data0, data1]

    # Generate expected results
    expected = gen_ground_truth(mod, target, dev, inputs)

    # Annotate target function.
    # This won't be necessary when TVMScript supports annotation.
    new_mlir_func = mod["byoc_func"].with_attr("Codegen", byoc_backend)
    mod["byoc_func"] = new_mlir_func

    # Run BYOC codegen for the annotated functions while run MetaSchedule tuning for the rest.
    with tempfile.TemporaryDirectory() as work_dir:
        with tvm.transform.PassContext(trace=Trace(mod), opt_level=3):
            seq = tvm.transform.Sequential(
                [
                    relax.transform.RunCodegen(),
                    relax.transform.RemoveUnusedFunctions(),
                    transform.LowerWithRelayOpStrategyPass(target),
                    relax.transform.MetaScheduleTuneIRMod(
                        target,
                        config=tune_config,
                        work_dir=work_dir,
                        database=None,
                    ),
                    relax.transform.MetaScheduleApplyHistoryBest(target, work_dir=work_dir),
                ]
            )
            new_mod = seq(mod)
            assert relax.analysis.well_formed(new_mod)

    # Generate VM executable.
    with tvm.transform.PassContext(opt_level=0):
        ex0 = relax.vm.build(new_mod, target, params={})

    # Check the results with the exepcted answer.
    check_executable(ex0, dev, inputs, expected)


if __name__ == "__main__":
    # test_single_annot_func(byoc_backend="onnx")
    # test_single_annot_func(byoc_backend="mlir")
    # test_mix_use_mlir_and_tvm(byoc_backend="onnx")
    test_mix_use_mlir_and_tvm(byoc_backend="mlir")
