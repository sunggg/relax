from __future__ import annotations
import numpy as np
import tvm
from tvm import relax
import tvm.relay.testing
import subprocess
from tvm.script import relax as R
import tempfile
from tvm import meta_schedule as ms
from tvm.relax.testing import transform
from tvm.relax.transform.tuning_api import Trace

# TODO:
# - relax op->onnx converter (func->func)

target_str = "llvm --num-cores=16"
target = tvm.target.Target(target_str)
dev = tvm.device(target_str, 0)

# MS tuning config
tune_config = ms.TuneConfig(
    strategy="evolutionary",
    num_trials_per_iter=4,
    max_trials_per_task=4,
    max_trials_global=8,
)


def gen_ground_truth(mod, target, dev, inputs):
    # Lower and run tuning
    # Since there is no default schedule for GPU in MS yet, this is necessary
    with tempfile.TemporaryDirectory() as work_dir:
        with tvm.transform.PassContext(trace=Trace(mod), opt_level=0):
            seq = tvm.transform.Sequential(
                [
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
    exec = relax.vm.build(new_mod, target, params={})
    vm = relax.VirtualMachine(exec, dev)
    return vm["main"](*inputs)


@tvm.register_func("relax.ext.mlir.runtime")
def mlir_runtime(shared_lib_path, num_args, *args):
    from PyRuntime import ExecutionSession

    inputs = [args[i].asnumpy() for i in range(num_args.value)]
    session = ExecutionSession(shared_lib_path=shared_lib_path)
    outs = session.run(inputs)
    return tvm.nd.array(outs[0], dev)


@tvm.register_func("relax.ext.mlir")
def mlir_codegen(func: relax.Function):  # func: relax.function):
    # TODO: convert relax function -> onnx and dump it to the file

    assert func.attrs and "global_symbol" in func.attrs
    global_symbol = func.attrs["global_symbol"]
    onnx_path = "bert_full.onnx"
    shared_lib_path = onnx_path[:-5] + ".so"
    num_inputs, num_outputs = len(func.params), 1

    # Codegen
    cmd = f"/home/spark/onnx-mlir/build/Debug/bin/onnx-mlir --EmitLib {onnx_path}".split(" ")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    gen_rt_mod = tvm.get_global_func("relax.CreatePyExtRuntime")
    assert gen_rt_mod
    return gen_rt_mod(
        global_symbol, shared_lib_path, "relax.ext.mlir.runtime", num_inputs, num_outputs
    )


def check_executable(exec, dev, inputs, expected):
    vm = relax.VirtualMachine(exec, dev)
    # Measure the performance w/o tuning log
    out = vm["main"](*inputs)
    tvm.testing.assert_allclose(out.numpy(), expected)


def test_single_annot_func():
    @tvm.script.ir_module
    class InputModule:
        @R.function
        def relax_func(x: Tensor((1, 64, 256), "float32")) -> Tensor:
            z1 = relax.add(x, x)
            z2 = relax.add(z1, z1)
            z3 = relax.add(z1, z2)
            return z3

        @R.function
        def main(x: Tensor((1, 64, 256), "float32")) -> Tensor:
            lv0 = relax_func(x)
            return lv0

    mod = InputModule
    assert isinstance(mod, tvm.IRModule)
    # TODO(@sunggg): Revisit when TVMScript supports annotation.
    # Annotate target function.
    new_relax_func = mod["relax_func"].with_attr("Codegen", "mlir")
    mod["relax_func"] = new_relax_func

    # Run Codegen pass
    seq = tvm.transform.Sequential(
        [relax.transform.RunCodegen(), relax.transform.RemoveUnusedFunctions()]
    )
    new_mod = seq(mod)

    with tvm.transform.PassContext(opt_level=0):
        ex0 = relax.vm.build(new_mod, target, params={})

    mock_inputs = [tvm.nd.array(np.full([1, 64, 256], 1, np.dtype(np.float32)), device=tvm.cpu(0))]
    vm = relax.VirtualMachine(ex0, dev)

    # Measure the performance w/o tuning log
    out = vm["main"](*mock_inputs)

    print(new_mod)
    print(f"exec: {out}")

    assert 0

    np0 = np.random.rand(2, 3).astype(np.float32)
    np1 = np.random.rand(2, 3).astype(np.float32)
    data0 = tvm.nd.array(np0, tvm.cpu())
    data1 = tvm.nd.array(np1, tvm.cpu())

    tmp = np0 + np1
    out1 = tmp + tmp
    expected = out1 + tmp
    check_executable(ex0, dev, [data0, data1], expected)


def test_mix_use_mlir_and_tvm():
    @tvm.script.ir_module
    class InputModule:
        @R.function
        def mlir_func(x: Tensor((1, 64, 256), "float32")) -> Tensor:
            z1 = relax.add(x, x)
            z2 = relax.add(z1, z1)
            z3 = relax.add(z1, z2)
            return z3

        @R.function
        def relax_func(x: Tensor((1, 64, 256), "float32")) -> Tensor:
            z1 = relax.add(x, x)
            z2 = relax.add(z1, z1)
            z3 = relax.add(z1, z2)
            return z3

        @R.function
        def main(x: Tensor((1, 64, 256), "float32")) -> Tensor:
            lv0 = mlir_func(x)
            lv1 = relax_func(lv0)
            return lv1

    # Prepare IRModule and its inputs
    mod = InputModule
    assert isinstance(mod, tvm.IRModule)
    # mock_inputs = [tvm.nd.array(np.full([1, 64, 256], 1, np.dtype(np.float32)), device=tvm.cpu(0))]
    new_mlir_func = mod["mlir_func"].with_attr("Codegen", "mlir")
    mod["mlir_func"] = new_mlir_func

    # Run Codegen pass
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
    with tvm.transform.PassContext(opt_level=0):
        ex0 = relax.vm.build(new_mod, target, params={})


if __name__ == "__main__":
    # test_single_annot_func()
    test_mix_use_mlir_and_tvm()
