from __future__ import annotations  # must import to defer parsing of annotations
import tvm
from tvm import relax
from tvm.script import relax as R
from tvm import ir
from tvm.ir import transform
from tvm.relax.expr import Call, Function, GlobalVar, ExternFunc, Tuple
from tvm.relax import PyExprMutator
import torch
import tvm.relay.testing

# Target and device configs
target_str = "llvm --num-cores=16"
target = tvm.target.Target(target_str)
dev = tvm.device(target_str, 0)


@tvm.register_func("relax.fallback.torch.runtime")
def torch_runtime(op_name, num_args, *args):
    tokens = op_name.split(".")
    torch_package = __import__(tokens[0])
    ops_package = getattr(torch_package, tokens[1])
    aten_package = getattr(ops_package, tokens[2])
    func = getattr(aten_package, tokens[3])

    inputs = [torch.tensor(args[i].asnumpy()) for i in range(num_args.value)]
    outs = func(*inputs)
    return tvm.nd.array(outs, dev)


@tvm.register_func("relax.fallback.torch.codegen")
def torch_codegen(global_symbol, op_name, num_inputs, num_outputs):
    # Build BYOC runtime and wrap with TVM runtime::Module
    gen_rt_mod = tvm.get_global_func("relax.CreatePyExtRuntime")
    assert gen_rt_mod
    return gen_rt_mod(
        global_symbol, op_name, "relax.fallback.torch.runtime", num_inputs, num_outputs
    )


@ir.transform.module_pass(opt_level=0)
class LowerExternOp(transform.Pass):
    """FILLME"""

    def __init__(self, target: Target):
        self.target = target

    def transform_module(self, mod: IRModule, ctx: PassContext) -> IRModule:
        """Implement lowering mechanism.

        Parameters
        ----------
        mod : IRModule
            Input IRModule with Relax ops

        ctx: PassContext
            Pass context

        Returns
        -------
        out_mod : IRModule
            Output IRModule with lowered TIR functions
        """
        target = self.target
        ext_mods = []

        @relax.expr_functor.mutator
        class Lowerer(PyExprMutator):
            """Mutator that performs lowering."""

            def visit_call_(self, call_node: Call):
                # Ignore function calls
                # We only target calls for operators
                if isinstance(call_node.op, (relax.GlobalVar, relax.expr.ExternFunc)):
                    return call_node

                op = call_node.op
                if op.name == "relax.extern_op":
                    extern_kind = call_node.attrs.extern_kind
                    extern_op_name = call_node.attrs.op_name
                    codegen = tvm.get_global_func(f"relax.fallback.{extern_kind}.codegen", True)
                    assert codegen
                    args = call_node.args
                    num_inputs, num_outputs = len(args), 1
                    global_symbol = f"relax.fallback.{extern_kind}.{extern_op_name}"
                    ext_mod = codegen(global_symbol, extern_op_name, num_inputs, num_outputs)
                    ext_mods.append(ext_mod)

                    new_args = [self.visit_expr(arg) for arg in args[:-1]]
                    shape_expr = args[-1]

                    call_tir = tvm.relay.op.op.get("relax.call_tir")
                    return Call(
                        op=call_tir,
                        args=[ExternFunc(global_symbol), new_args[0], shape_expr],
                        attrs=None,
                        type_args=[call_node.checked_type],
                    )
                else:
                    return call_node

            # TOOD(@team): transform() wapper is necessary to include TIR functions.
            # IMO, this is bit unintuitive. Can we improve this?
            def transform(self):
                for gv, func in mod.functions.items():
                    if isinstance(func, relax.Function):
                        updated_func = self.visit_expr(func)
                        self.builder_.update_func(gv, updated_func)
                new_mod = self.builder_.get()
                new_mod = new_mod.with_attrs(mod.attrs) if mod.attrs else new_mod
                new_mod = new_mod.with_attr("external_mods", ext_mods)
                return new_mod

        return Lowerer().transform()


def test_extern_op() -> None:
    shape_anno = [16, 16]
    type_anno = relax.DynTensorType(2, "float32")
    v0 = relax.Var("v0", shape_anno, type_anno)
    v1 = relax.Var("v1", shape_anno, type_anno)
    v2 = relax.op.extern_op("torch", "torch.ops.aten.add", [v0, v1], shape_anno, "float32")
    func = relax.Function([v0, v1], v2, ret_type=type_anno)
    func = func.with_attr("global_symbol", "main")

    mod = tvm.IRModule.from_expr(func)
    assert mod
    target = tvm.target.Target("llvm")
    mod = relax.transform.Normalize()(mod)
    assert relax.analysis.well_formed(mod)
    print(mod)
    new_mod = LowerExternOp(target)(mod)
    assert relax.analysis.well_formed(new_mod)
    print(new_mod)

    torch_input = [torch.rand(shape_anno), torch.rand(shape_anno)]
    inputs = [tvm.nd.array(inp.numpy(), dev) for inp in torch_input]

    # Generate VM executable.
    with tvm.transform.PassContext(opt_level=0):
        ex0 = relax.vm.build(new_mod, target, params={})

    # Check the results with the exepcted answer.
    vm = relax.VirtualMachine(ex0, dev)

    # Measure the performance w/o tuning log
    out = vm["main"](*inputs)

    # ground truth
    expected = torch.add(*torch_input)
    tvm.testing.assert_allclose(out.numpy(), expected.numpy(), atol=1e-5, rtol=1e-5)


def test_extern_op_with_tvm_script():
    @tvm.script.ir_module
    class InputModule:
        @R.function
        def main(x: Tensor((16, 16), "float32"), y: Tensor((16, 16), "float32")) -> Tensor:
            lv0 = relax.extern_op("torch", "torch.ops.aten.add", (x, y), (16, 16), dtype="float32")
            return lv0

    mod = InputModule
    assert mod
    assert relax.analysis.well_formed(mod)
    new_mod = LowerExternOp(target)(mod)
    assert relax.analysis.well_formed(new_mod)
    torch_input = [torch.rand(16, 16), torch.rand(16, 16)]
    inputs = [tvm.nd.array(inp.numpy(), dev) for inp in torch_input]

    # Generate VM executable.
    with tvm.transform.PassContext(opt_level=0):
        ex0 = relax.vm.build(new_mod, target, params={})

    # Check the results with the exepcted answer.
    vm = relax.VirtualMachine(ex0, dev)

    # Measure the performance w/o tuning log
    out = vm["main"](*inputs)

    # ground truth
    expected = torch.add(*torch_input)
    tvm.testing.assert_allclose(out.numpy(), expected.numpy(), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    # test_extern_op()
    test_extern_op_with_tvm_script()
