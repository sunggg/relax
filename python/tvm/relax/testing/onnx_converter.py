from __future__ import annotations  # must import to defer parsing of annotations
import tvm
from tvm import relax
import onnx
import numpy as np
from onnx import helper, TensorProto, OperatorSetIdProto
import tvm.script
from tvm.script import relax as R
from tvm.relax import PyExprVisitor


relax_to_onnx_op_mapping = {"relax.add": "Add", "relax.multiply": "Mul"}


@relax.expr_functor.visitor
class RelaxToOnnxVisitor(PyExprVisitor):
    def __init__(self):
        super().__init__()
        self._node_dict = {}  # relax tensor to onnx tensor map
        self._nodes = []  # all intermediate onnx nodes
        self._inputs = []  # function inputs
        self._outputs = []  # function outputs

    def _shape_expr_to_list(self, shape: relax.ShapeExpr) -> list[int]:
        return [int(dim) for dim in shape]

    def convert(self, func: relax.Function, file_path: str = None) -> onnx.ModelProto:
        self.visit_expr(func)
        graph_def = helper.make_graph(
            self._nodes,  # nodes
            "relax-onnx-model",  # name
            self._inputs,  # inputs
            self._outputs,  # outputs
        )

        def _get_opsets():
            opsets = []
            imp = OperatorSetIdProto()
            imp.version = 11
            opsets.append(imp)
            return opsets

        model_def = helper.make_model(graph_def, opset_imports=_get_opsets())
        if file_path:
            onnx.save(model_def, file_path)
        return model_def

    def visit_function_(self, func: relax.Function) -> None:
        for param in func.params:
            tensor = helper.make_tensor_value_info(
                param.name_hint,
                onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(param.checked_type.dtype)],
                self._shape_expr_to_list(param.shape),
            )
            self._node_dict[param] = tensor
            self._inputs.append(tensor)

        self.visit_expr(func.body)

    def visit_seq_expr_(self, seqexpr: relax.SeqExpr) -> None:
        blocks = []
        for block in seqexpr.blocks:
            self.visit_binding_block(block)
        self.visit_expr(seqexpr.body)
        self._outputs.append(self._node_dict[seqexpr.body])

    def visit_var_binding_(self, binding: relax.VarBinding) -> None:
        # visit var of binding to make onnx tensor
        tensor = helper.make_tensor_value_info(
            binding.var.name_hint,
            onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(binding.var.checked_type.dtype)],
            self._shape_expr_to_list(binding.var.shape),
        )
        self._node_dict[binding.var] = tensor

        # visit value of binding to make onnx node
        call = binding.value
        relax_op = call.op.name

        op_node = helper.make_node(
            relax_to_onnx_op_mapping[relax_op],  # name
            [arg.name_hint for arg in call.args],  # inputs
            [binding.var.name_hint],  # outputs
        )

        self._nodes.append(op_node)


def main():
    @tvm.script.ir_module
    class Module:
        @R.function
        def f(x: Tensor((2, 2), "float32")):
            y = R.add(x, x)
            z = R.multiply(y, x)
            return z

    relax_to_onnx = RelaxToOnnxVisitor()
    onnx_model = relax_to_onnx.convert(Module["f"], "./relax.onnx")
    print("The model is:\n{}".format(onnx_model))
    onnx.checker.check_model(onnx_model)


main()