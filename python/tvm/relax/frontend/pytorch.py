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
# pylint: disable=import-self, too-many-lines, len-as-condition, no-else-return, unused-variable, too-many-nested-blocks
# pylint: disable=consider-iterating-dictionary, invalid-name, unused-argument, unused-variable, broad-except
# pylint: disable=import-outside-toplevel, simplifiable-if-expression, cell-var-from-loop, unnecessary-lambda
# pylint: disable=missing-function-docstring, redefined-builtin
"""PT: PyTorch frontend."""

import numpy as np
import itertools
import tvm
from tvm.relay.prelude import Prelude
from tvm.ir import IRModule
from tvm.topi.utils import get_const_tuple
from tvm.relay.frontend.pytorch_utils import is_version_greater_than, getattr_attr_name
from tvm.relay.frontend.common import logger
from ..expr_functor import PyExprMutator, mutator
from .. import transform
from .. import expr as _expr
from ..ty import DynTensorType
from .. import analysis as _analysis

# from .. import function as _function
from .. import op as _op

__all__ = ["from_pytorch"]

# This returns a "subgraph" which puts variables whenever
# the type is known. It also records things to map the input
# nodes to the extracted graph's nodes.
# As Python objects are not round-trippable through C++, and
# our type annotations only live in Python, we need to map
# the we need to map the nodes we get in visiting to the nodes
# we used to construct the graph (they are the same in C++,
# match each other in dictionary lookups, but are not the same
# in Python) by using the hint dictionary filled as
# {node: node for node in nodes} to get the type annotations.
# https://discuss.tvm.apache.org/t/round-tripping-objects-through-the-ffi/8440
@mutator
class _TypeFinder(PyExprMutator):
    def __init__(self, types):
        super().__init__()
        self.counter = 0
        self.vars = {}
        self.types = types
        self.leave = set()  # some variables are not inputs

    # def visit_function_(self, fn):
    #    self.leave.update(fn.params)
    #    return super().visit_function_(fn)

    def visit_expr(self, expr):
        # if expr in self.leave:
        #    return super().visit_expr(expr)
        # if expr in self.vars:
        #    return self.vars[expr]
        # if isinstance(expr, tvm.relax.Var):
        #    self.vars[expr] = expr
        #    return expr
        # if expr in self.types:
        #    ty = self.types[expr]
        #    v = tvm.relax.Var(f"_{self.counter}", type_annotation=ty)
        #    self.counter += 1
        #    self.vars[expr] = v
        #    return v
        print(expr)
        return super().visit_expr(expr)


def _should_construct_dynamic_list(list_construct_node):
    # if this list is element-accessed or modified at runtime, generate List ADT
    def inplace_add_to_add(op_name):
        if op_name == "aten::add_":
            return "aten::add"
        else:
            return op_name

    uses = _get_uses(list_construct_node)

    for loop_use in filter(lambda use: use.user.kind() == "prim::Loop", uses):
        block_input_index = loop_use.offset - 1
        block = list(loop_use.user.blocks())[0]
        list_loop_var = list(block.inputs())[block_input_index]
        uses += _get_uses(list_loop_var.node())

    op_names = map(inplace_add_to_add, set(use.user.kind() for use in uses))

    list_ops = set(["aten::add", "aten::__getitem__"])
    intersect = list_ops.intersection(op_names)

    if len(intersect) > 0 and intersect != set(["aten::add"]):
        return True

    # if add op outputs list, it is dynamic so we need to construct List ADT
    for use in filter(lambda use: use.user.kind() in ["aten::add", "aten::add_"], uses):
        output_type = _get_node_type(use.user)
        if output_type == "ListType":
            return True

    return False


def _is_int_seq(seq):
    # TODO (t-vi): handle non-int constants? (like numpy.intXX)
    return len(seq) > 0 and all([isinstance(i, int) for i in seq])


# operator implementation
class PyTorchOpConverter:
    """A helper class for holding PyTorch op converters."""

    def __init__(self, prelude, default_dtype):
        self.prelude = prelude
        self.default_dtype = default_dtype
        self.create_convert_map()
        self.types = {}  # map from nodes to (Relay) type annotations

    # this incrementally infers the type, see the comments on the type visitor
    # above.
    def infer_type(self, node, mod=None):
        """An incremental method to infer the type of a node in the relax graph."""

        if node in self.types:
            return self.types[node]
        if isinstance(node, tvm.relax.Var):
            return node.type_annotation

        tf = _TypeFinder(types=self.types)
        new_node = tf.visit_expr(node)
        fn = _expr.Function(list(tf.vars.values()), new_node)
        new_mod = IRModule({"main": fn})
        if mod is not None:
            new_mod.update(mod)
        new_mod = transform.RemoveUnusedFunctions()(new_mod)
        new_mod = transform.Normalize()(new_mod)
        entry = new_mod["main"]
        ty = entry.body.checked_type
        self.types[node] = ty
        return self.types[node]

    def infer_type_with_prelude(self, val):
        body = self.infer_type(val, self.prelude.mod)
        return body

    def infer_shape(self, inputs, mod=None):
        """A method to get the output type of an intermediate node in the graph."""
        typ = self.infer_type(inputs, mod=mod)
        if hasattr(typ, "shape"):
            # Regular operator that outputs tensors
            return get_const_tuple(typ.shape)
        # The return type is not a tensor, for example List
        return typ

    def infer_shape_with_prelude(self, inputs):
        return self.infer_shape(inputs, mod=self.prelude.mod)

    def record_output_type(self, output):
        if isinstance(output, tuple):
            cleaned_output = [o for o in output if o is not None]
            types = self.infer_type_with_prelude(_expr.Tuple(cleaned_output))
            for o, t in zip(cleaned_output, types.fields):
                self.types[o] = t
        elif isinstance(output, _expr.Expr):
            self.infer_type_with_prelude(output)
        # it can also happen that the type is int or so

    def add(self, inputs, input_types):
        # add_ is overloaded for tensor add and list concat
        if input_types[0] == "ListType":
            return self.prelude.concat(inputs[0], inputs[1])
        return self.make_elemwise("add")(inputs, input_types)

    # Operator mappings
    def create_convert_map(self):
        self.convert_map = {
            "aten::add": self.add,
        }

    def report_missing_conversion(self, op_names):
        known_ops = [
            "prim::Constant",
            "prim::GetAttr",
            "prim::ListConstruct",
            "prim::ListUnpack",
            "prim::TupleConstruct",
            "prim::TupleUnpack",
            "prim::RaiseException",
            "prim::If",
            "prim::Loop",
        ]
        known_ops += list(self.convert_map.keys())
        missing = []

        for op_name in op_names:
            # Also take care of in-place variant ops like aten::relu_
            if op_name not in known_ops and not (
                op_name.endswith("_") and op_name[:-1] in known_ops
            ):
                missing.append(op_name)

        if missing:
            msg = "The following operators are not implemented: {}".format(missing)
            raise NotImplementedError(msg)

    def convert_block(self, block, outputs):
        """Translate Torch "Block", used for prim::If and prim::Loop"""
        ops = _get_operator_nodes(block.nodes())
        ret_names = _get_input_names(block.returnNode())
        return self.convert_operators(ops, outputs, ret_names)

    def convert_if(self, if_node, outputs):
        """Translate Torch prim::If to Relay If"""
        cond = outputs[if_node.inputsAt(0).debugName()]
        blocks = list(if_node.blocks())
        true_branch = self.convert_block(blocks[0], outputs)
        false_branch = self.convert_block(blocks[1], outputs)
        assert len(true_branch) == 1 and len(false_branch) == 1
        return _expr.If(cond, true_branch[0], false_branch[0])

    def convert_loop(self, loop_node, outputs):
        """Translate Torch prim::Loop to Relay while_loop"""

        def get_input(index):
            ivalue = loop_node.inputsAt(index)
            inode = ivalue.node()
            if inode.kind() == "prim::Constant":
                return _expr.const(_get_constant(inode))
            var_name = ivalue.debugName()
            assert var_name in outputs
            return _wrap_const(outputs[var_name])

        # Refer to the spec for prim::Loop below
        # https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/OVERVIEW.md#loops
        # The first input: %max_trip_count
        # The second input: %initial_condition
        # The rest of input: loop variables
        max_loop_count = get_input(0)
        init_cond = get_input(1)
        num_loop_var = len(list(loop_node.inputs())) - 2
        init_vals = [get_input(i + 2) for i in range(num_loop_var)]

        # while loop has always max_loop_count being int64 max
        # max_loop_count.data (tvm.runtime.NDArray) is -1, so _get_constant again
        is_while_loop = (
            isinstance(max_loop_count, _expr.Constant)
            and _get_constant(loop_node.inputsAt(0).node()) == sys.maxsize
        )

        if is_while_loop:
            loop_iter_dtype = "bool"
            # while loop with non input dependent condition such as while i < 10:
            # init_cond is int, need to cast to bool to type check
            if isinstance(init_cond, _expr.Constant):
                init_cond = _op.cast(init_cond, "bool")
            init_loop_iter_val = init_cond
        else:
            loop_iter_dtype = "int32"
            # always count from 0
            init_loop_iter_val = _expr.const(0, dtype="int32")

        body_block = list(loop_node.blocks())[0]
        block_input_names = _get_input_names(body_block)
        num_block_inputs = len(block_input_names)
        name_val_pairs = list(zip(block_input_names, [init_loop_iter_val] + init_vals))
        outputs.update(name_val_pairs)

        def get_var(name, val):
            if val:
                checked_type = self.infer_type_with_prelude(val)
                if hasattr(checked_type, "shape"):
                    shape = get_const_tuple(checked_type.shape)
                    actual_shape = []
                    for dim in shape:
                        if isinstance(dim, int) and dim == 0:
                            actual_shape.append(Any())
                        else:
                            actual_shape.append(dim)
                    return _expr.var(name, shape=actual_shape, dtype=checked_type.dtype)
                else:
                    return _expr.var(name, type_annotation=checked_type)
            return _expr.var(name)

        loop_iter_var = _expr.var(block_input_names[0], shape=(), dtype=loop_iter_dtype)
        loop_vars = [get_var(name, val) for name, val in name_val_pairs[1:]]

        # Add non constant free variables to loop variables to prevent code blow up
        # Without this, if there are two for loops in a row, which often happens
        # if the outer loop is unrolled, the computation corresponding to the first for loop
        # is inlined inside loop body, turning O(N) + O(N) computation into O(N^2).
        # This issue was found when converting from Stacked LSTM test. Torch does not add the
        # outputof the eariler loop into loop variables of the next loop.
        # So the variable corresponding to the first loop output appears free in the second
        # loop body.
        free_vars = [
            var
            for var in _get_free_vars_from_block(body_block)
            if var in outputs
            and not isinstance(outputs[var], (_expr.Constant, int, float, str))
            and outputs[var]
        ]

        prev_outputs = {}
        for name in free_vars:
            prev_output = outputs[name]
            new_loop_var = get_var(name, prev_output)
            prev_outputs[name] = prev_output
            outputs[name] = new_loop_var
            loop_vars.append(new_loop_var)
            init_vals.append(prev_output)

        def cond(*current_vals):
            i = current_vals[0]

            if is_while_loop:
                return _op.equal(i, _expr.const(True, "bool"))

            return _op.less(i, max_loop_count)

        def body(*current_vals):
            # Update loop variables using the prev iteration outputs
            assert len(current_vals) == num_block_inputs + len(free_vars)

            for (i, val) in enumerate(current_vals):
                if i < num_block_inputs:
                    outputs[block_input_names[i]] = val
                else:
                    outputs[free_vars[i - num_block_inputs]] = val

            block_outputs = self.convert_block(body_block, outputs)
            block_outputs += [outputs[name] for name in free_vars]

            if not is_while_loop:
                # iter var increment implicit in torch, so do it manually
                # for while loop, block_outputs[0] is already a boolean,
                # the result of termination check
                incr = _expr.const(1, dtype="int32")
                block_outputs[0] = current_vals[0] + incr

            return block_outputs

        loop = while_loop(cond, [loop_iter_var] + loop_vars, body)
        loop_val = loop(init_loop_iter_val, *init_vals)

        # restore original output values for free vars
        outputs.update(prev_outputs)

        # The first element is a loop counter or boolean condition, ignore it
        return [_expr.TupleGetItem(loop_val, i + 1) for i in range(num_loop_var)]

    def convert_operators(self, operators, outputs, ret_names):
        """Convert each Torch IR operators to Relay equivalent"""
        for node_name, op_node in operators:
            operator = op_node.kind()
            inputs = _get_op_inputs(op_node, outputs)

            if operator == "prim::Constant":
                outputs[node_name] = _get_constant(op_node)
            elif operator == "prim::ListConstruct" and _should_construct_dynamic_list(op_node):
                outputs[node_name] = self.convert_to_list_adt(inputs)
            elif operator == "prim::ListConstruct":
                # This assumes that no more elements will be appended to this list
                # In this case, we keep the Python list
                outputs[node_name] = inputs
            elif operator == "prim::TupleConstruct":

                def _handel_nested_input(inputs):
                    inputs_list = []
                    for i, _ in enumerate(inputs):
                        if isinstance(inputs[i], list):
                            inputs_list.append(_handel_nested_input(inputs[i]))
                        else:
                            assert isinstance(inputs[i], _expr.Expr)
                            inputs_list.append(inputs[i])
                    return _expr.Tuple(inputs_list)

                outputs[node_name] = _handel_nested_input(inputs)
            elif operator in ["prim::ListUnpack", "prim::TupleUnpack"]:
                assert len(inputs) == 1
                if isinstance(inputs[0], (list, _expr.TupleWrapper)):
                    unpacked = inputs[0]
                else:
                    unpacked = _unpack_tuple(inputs[0])
                outputs.update(zip(_get_output_names(op_node), unpacked))
            elif operator == "prim::prim::RaiseException":
                logger.warning("raising exceptions is ignored")
                outputs[node_name] = None
            elif operator == "prim::If":
                if_out = self.convert_if(op_node, outputs)
                outputs[node_name] = if_out
            elif operator == "prim::Loop":
                loop_out = self.convert_loop(op_node, outputs)
                unpacked_names = _get_output_names(op_node)
                assert len(loop_out) == len(unpacked_names)
                outputs.update(zip(unpacked_names, loop_out))
            else:
                if operator not in self.convert_map:
                    # Handle missing operators
                    relax_op = _op.extern_op
                    tensor_inputs = list()
                    attr_inputs = list()
                    for inp in inputs:
                        if isinstance(inp, tvm.relax.Var):
                            tensor_inputs.append(inp)
                        else:
                            attr_inputs.append(inp)

                    torch_op_name = "torch.ops.aten." + operator.split(":")[-1]
                    relax_out = relax_op("torch", torch_op_name, tuple(tensor_inputs), attr_inputs)

                    """
                    # At this point, the only possible ops that are not in convert_map are
                    # in-place variant of ops like aten::relu_
                    assert operator.endswith("_")
                    logger.warning(
                        "An in-place op %s found, the result will not be correct "
                        "if the model depends on side-effects by this op.",
                        operator,
                    )
                    relax_op = self.convert_map[operator[:-1]]
                    """
                else:
                    relax_op = self.convert_map[operator]
                    relax_out = relax_op(
                        inputs, _get_input_types(op_node, outputs, default_dtype=self.default_dtype)
                    )

                print(relax_out)
                self.record_output_type(relax_out)
                assert 0

                if isinstance(relax_out, tuple):
                    # This is for torch operators that return multiple outputs
                    # See _adaptive_max_2d above for example
                    out_names = _get_output_names(op_node)
                    outputs.update(zip(out_names, relax_out))
                else:
                    assert op_node.outputsSize() == 1
                    outputs[node_name] = relax_out

        return [_wrap_const(outputs[ret_name]) for ret_name in ret_names]


def _pytorch_result_type(dtypes, non_tensor_inputs):
    """This promotes TVM dtypes like PyTorch would"""
    import torch

    dtype_map = {
        "float64": torch.float64,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int64": torch.int64,
        "int32": torch.int32,
        "int16": torch.int16,
        "int8": torch.int8,
        "uint8": torch.uint8,
        "bool": torch.bool,
    }
    if len(dtypes) > 0:
        result_type = dtypes[0]
        for dt in dtypes[1:]:
            if dt != result_type:  # we don't want to work with same types as we
                # don't do quantized here (which cannot be promoted?)
                result_type = _convert_data_type(
                    str(
                        torch.result_type(
                            torch.zeros((), dtype=dtype_map[result_type]),
                            torch.zeros((), dtype=dtype_map[dt]),
                        )
                    )
                )
    else:
        result_type = "bool"  # this is the smallest type...
    for inp in non_tensor_inputs:
        result_type = _convert_data_type(
            str(torch.result_type(torch.zeros((), dtype=dtype_map[result_type]), inp))
        )
    return result_type


# Helper functions for operator implementation
def _convert_dtype_value(val):
    """converts a PyTorch the PyTorch numeric type id to a torch scalar type."""
    convert_torch_dtype_map = {
        11: "torch.bool",
        7: "torch.float64",
        6: "torch.float32",
        5: "torch.float16",
        4: "torch.int64",
        3: "torch.int32",
        2: "torch.int16",
        1: "torch.int8",
        0: "torch.uint8",
        None: "torch.int64",
    }  # Default is torch.int64
    if val in convert_torch_dtype_map:
        return _convert_data_type(convert_torch_dtype_map[val])
    else:
        msg = "Torch data type value %d is not handled yet." % (val)
        raise NotImplementedError(msg)


def _convert_data_type(input_type, default_dtype=None):
    """converts the PyTorch scalar type input_type to a TVM dtype.
    optionally, default_dtype can be a TVM dtype that is used
    if input_type is None (but not when it is unknown)"""
    if input_type is None and default_dtype is not None:
        return default_dtype

    input_type = input_type.lower()
    if input_type in ["double", "float64", "torch.float64"]:
        return "float64"
    elif input_type in ["float", "float32", "torch.float32"]:
        return "float32"
    elif input_type in ["half", "float16", "torch.float16"]:
        return "float16"
    elif input_type in ["long", "int64", "torch.int64"]:
        return "int64"
    elif input_type in ["int", "int32", "torch.int32"]:
        return "int32"
    elif input_type in ["short", "int16", "torch.int16"]:
        return "int16"
    elif input_type in ["char", "int8", "torch.int8"]:
        return "int8"
    elif input_type in ["byte", "uint8", "torch.uint8"]:
        return "uint8"
    elif input_type in ["quint8", "torch.quint8"]:
        return "quint8"
    elif input_type in ["qint8", "torch.qint8"]:
        return "qint8"
    elif input_type in ["qint32", "torch.qint32"]:
        return "qint32"
    elif input_type in ["bool", "torch.bool"]:
        return "bool"
    elif input_type in ["str"]:
        return "str"
    else:
        raise NotImplementedError("input_type {} is not handled yet".format(input_type))
    return "float32"  # Never reached


def _create_typed_const(data, dtype):
    """create a (scalar) constant of given value and dtype.
    dtype should be a TVM dtype"""

    if dtype == "float64":
        typed_data = _expr.const(np.float64(data), dtype=dtype)
    elif dtype == "float32":
        typed_data = _expr.const(np.float32(data), dtype=dtype)
    elif dtype == "float16":
        typed_data = _expr.const(np.float16(data), dtype=dtype)
    elif dtype == "int64":
        typed_data = _expr.const(np.int64(data), dtype=dtype)
    elif dtype == "int32":
        typed_data = _expr.const(np.int32(data), dtype=dtype)
    elif dtype == "int16":
        typed_data = _expr.const(np.int16(data), dtype=dtype)
    elif dtype == "int8":
        typed_data = _expr.const(np.int8(data), dtype=dtype)
    elif dtype == "uint8":
        typed_data = _expr.const(np.uint8(data), dtype=dtype)
    else:
        raise NotImplementedError("input_type {} is not handled yet".format(dtype))
    return typed_data


def _wrap_const(c):
    if not isinstance(c, (_expr.Expr, list, tvm.tir.expr.Any)):
        return _expr.const(c)
    return c


def _run_jit_passes(graph, enable_lower_all_tuples=True):
    """The inline pass is necessary to unwrap prim::CallMethod"""
    # pylint: disable=c-extension-no-member
    import torch

    if is_version_greater_than("1.5.1"):
        # This is required for torchvision detection models from 1.6 above
        # It is the same as _jit_pass_inline, except that it has some special
        # case behaviors for some ops such as aten::__interpolate()
        torch._C._jit_pass_onnx_function_substitution(graph)
    else:
        torch._C._jit_pass_inline(graph)

    if enable_lower_all_tuples:
        torch._C._jit_pass_lower_all_tuples(graph)


def _get_tensor_and_var(torch_tensor, name):
    tensor = tvm.nd.array(torch_tensor.cpu().numpy())
    var = _expr.Var(
        name,
        shape_annotation=_expr.ShapeExpr(tensor.shape),
        type_annotation=DynTensorType(len(tensor.shape), tensor.dtype),
    )
    return tensor, var


def _get_output_name(node):
    assert node.outputsSize() == 1
    return node.output().debugName()


def _get_output_names(node):
    return [output.debugName() for output in node.outputs()]


def _get_input_names(node_or_graph):
    return [inp.debugName() for inp in node_or_graph.inputs()]


def _get_op_inputs(op_node, outputs):
    return [outputs[name] for name in _get_input_names(op_node)]


def _get_node_type(node):
    assert node.outputsSize() == 1
    return node.output().type().kind()


def _get_uses(node):
    uses = []
    for output in node.outputs():
        uses += output.uses()
    return uses


def _get_users(node):
    return [use.user for use in _get_uses(node)]


def _getattr_full_name(getattrs, sep="."):
    return sep.join([getattr_attr_name(node) for node in getattrs])


def _get_pytorch_value_type(typ, default_dtype="float32"):
    kind = typ.kind()
    if kind == "TensorType":
        if typ.scalarType() is None:
            # Tensor's type can be unknown if we use torch.jit.script(...)
            # Defaults can be passed in, if not it is float32
            logger.warning("Untyped Tensor found, assume it is %s", default_dtype)
            return default_dtype
        else:
            return _convert_data_type(typ.scalarType())

    elif kind == "ListType":
        return "ListType"
    elif kind in ["IntType", "FloatType", "BoolType", "StringType", "OptionalType"]:
        pt_dtype = str(typ).lower()
        dtype = pt_dtype if pt_dtype == "OptionalType" else _convert_data_type(pt_dtype)
        return dtype
    else:
        return "UnsupportedType"


def _get_input_types(op_node, outputs, default_dtype="float32"):
    """Returns a TVM dtype for each input nodes derived from the torch type"""
    in_types = []
    for inp in op_node.inputs():
        if inp.node().kind() == "prim::GetAttr":
            # GetAttr nodes always return None when we call scalarType() on it
            name = inp.debugName()
            assert name in outputs
            if isinstance(outputs[name], _expr.Var):
                in_types.append(outputs[name].type_annotation.dtype)
            else:
                # For quantized modules with parameters, here we would get
                # "prim::GetAttr[name="_packed_params"]". Since the dtype corresponding to
                # _packed_params is not needed by quantized ops, we return an arbitrary type.
                in_types.append(default_dtype)
        else:
            in_types.append(_get_pytorch_value_type(inp.type(), default_dtype=default_dtype))

    return in_types


def _get_constant(node):
    """Retrieve a constant associated with this prim::Constant node"""
    attribute_names = node.attributeNames()
    num_attributes = len(attribute_names)

    if num_attributes == 1:
        attr_name = attribute_names[0]
        ty = node.output().type().kind()

        if ty == "IntType":
            return node.i(attr_name)
        elif ty == "BoolType":
            return bool(node.i(attr_name))
        elif ty in ["FloatType", "LongType"]:
            return node.f(attr_name)
        elif ty in ["TensorType", "CompleteTensorType"]:
            tensor = node.t(attr_name)
            if tensor.is_cuda:
                tensor = tensor.cpu()
            if len(tensor.shape) == 0:  # tensor(0.1)
                # TODO(t-vi): When is this needed?
                return tensor.item()
            return _wrap_const(tensor.numpy())
        elif ty in ["DeviceObjType", "StringType"]:
            return node.s(attr_name)
        elif ty == "FunctionType":
            return None
        else:
            raise NotImplementedError("Unsupported type: %s" % ty)
    else:
        assert num_attributes == 0
        return None


def _get_operator_nodes(nodes):
    """Returns torch IR nodes that need conversion to Relay"""
    ops = []
    # Traverse nodes and add to graph
    for node in nodes:
        if node.outputsSize() == 0:
            continue
        if node.outputsSize() > 1:
            node_name = "_".join(_get_output_names(node))
        else:
            node_name = _get_output_name(node)

        if node.kind() != "prim::GetAttr":
            ops.append((node_name, node))

    return ops


def _get_relax_input_vars(graph, input_infos, prelude, is_module=True, default_dtype="float32"):
    """
    Return Relay vars from input shapes and create entries based on
    expected graph inputs - to allow translation
    """

    graph_inputs = list(graph.inputs())
    if is_module:
        # a module has "self" as first input, which we do not need/want
        graph_inputs = graph_inputs[1:]

    if not isinstance(input_infos, list):
        msg = "Graph inputs input_infos should be a list"
        raise RuntimeError(msg)

    if len(graph_inputs) != len(input_infos):
        msg = "PyTorch has {} inputs and input_infos lists {}.".format(
            len(graph_inputs), len(input_infos)
        )
        raise RuntimeError(msg)

    def get_relax_ty(ishape, itype, pt_type):
        if pt_type.kind() == "TensorType":
            if not (_is_int_seq(ishape) or len(ishape) == 0):
                msg = "Shape for Tensors must be lists of ints"
                raise RuntimeError(msg)
            if (pt_type.dim() is not None and pt_type.dim() != len(ishape)) or (
                pt_type.sizes() is not None
                and any([s1 != s2 for s1, s2 in zip(pt_type.sizes(), ishape)])
            ):
                msg = "Shapes of input list and information in the graph do not match"
                raise RuntimeError(msg)
            pt_dtype = pt_type.scalarType()
            if not pt_dtype and itype:
                pt_dtype = itype
            dtype = _convert_data_type(pt_dtype, default_dtype=default_dtype)
            return DynTensorType(len(ishape), dtype)
        elif pt_type.kind() == "TupleType":
            if not isinstance(ishape, tuple):
                msg = "Shapes for tuples must be tuples"
                raise RuntimeError(msg)
            return TupleType(
                [get_relax_ty(elem, itype, pt_t) for elem, pt_t in zip(ishape, pt_type.elements())]
            )
        elif pt_type.kind() == "ListType":
            if not isinstance(ishape, list):
                msg = "Shapes for lists must be lists"
                raise RuntimeError(msg)
            pt_elemtype = pt_type.getElementType()
            elem_tys = [get_relax_ty(s, itype, pt_elemtype) for s in ishape]
            if len(elem_tys) > 0 and not all(map(lambda ty: ty == elem_tys[0], elem_tys)):
                msg = "List elements need have identical types"
                raise RuntimeError(msg)
            rlist, _, _ = prelude.mod.get_type("List")
            return rlist(elem_tys[0])
        elif pt_type.kind() == "OptionalType":
            # we do not support None yet, so we fill in the type
            return get_relax_ty(ishape, itype, pt_type.getElementType())
        # TODO: scalar inputs
        raise NotImplementedError("unsupported input type")

    input_vars = {}

    new_input_infos = []
    for num, inp in enumerate(input_infos):
        if not isinstance(inp, tuple):
            msg = "Graph input {} is not a tuple".format(num)
            raise RuntimeError(msg)
        if len(inp) != 2 or not isinstance(inp[0], str):
            msg = (
                "Graph input {} is not valid,"
                " expected ('name', shape) or ('name', (shape, dtype))".format(inp)
            )
            raise RuntimeError(msg)
        if not isinstance(inp[1], tuple) or len(inp[1]) == 0 or not isinstance(inp[1][-1], str):
            new_input_infos.append((inp[0], (inp[1], default_dtype)))
        else:
            new_input_infos.append(inp)

    input_types = [
        (name, get_relax_ty(info[0], info[1], gi.type()))
        for (name, info), gi in zip(new_input_infos, graph_inputs)
    ]

    ir_inputs = [i.debugName() for i in graph_inputs]
    for ir_input, (name, itype) in zip(ir_inputs, input_types):
        inp = _expr.Var(name, type_annotation=itype)
        # Translate from graph input to user input name
        input_vars[ir_input] = inp

    return input_vars


def _unpack_tuple(tup):
    def unpack(tup, num_fields):
        return [_expr.TupleGetItem(tup, i) for i in range(num_fields)]

    if isinstance(tup, _expr.Tuple):
        return unpack(tup, len(tup.fields))
    elif isinstance(tup.type_annotation, TupleType):
        return unpack(tup, len(tup.type_annotation.fields))
    # shouldn't happen
    assert False


def _get_free_vars_from_block(block):
    block_inp_names = _get_input_names(block)
    bound_names = block_inp_names
    free_vars = set()

    for node in block.nodes():
        inp_names = _get_input_names(node)
        list_diff = [name for name in inp_names if name not in bound_names]
        free_vars.update(list_diff)
        bound_names += _get_output_names(node)

    return free_vars


def get_use_chains(root_node, terminate=lambda _: False):
    """
    Track a chain of users of this node forward, returning a list of chains
    See get_attr_chains below for its usage
    """

    def concat_lists(lists):
        return itertools.chain.from_iterable(lists)

    def inner(current, accum):
        users = _get_users(current)

        if not users or terminate(users):
            return [accum]

        return concat_lists([inner(nxt, accum + [nxt]) for nxt in users])

    return inner(root_node, [root_node])


def get_attr_chains(root_getattr_node):
    """Returns chains of attribute access starting from root_getattr_node

    For example, given attribute "block", as in "self.block" when "self" points
    to the top level torch.nn.Module, it returns lists of attribute "chains",
    e.g. ['block', '2'], ['block', '1'], ['block', '0', '_packed_params']

    These sets of attributes form full attribute accessors. For example,
    "self.block.1", "self.block.2" will return the second and third submodule,
    and "self.block.0._packed_params" will return the parameters of the first
    submodule.
    """

    def terminate(users):
        next_attrs = [user for user in users if user.kind() == "prim::GetAttr"]
        return len(next_attrs) == 0

    return get_use_chains(root_getattr_node, terminate)


def convert_params(graph, state_dict, use_parser_friendly_name=False):
    """
    Return Relay vars and TVM NDArrays for input parameters
    A chain of prim::GetAttr nodes is processed one at a time
    """
    getattr_nodes = graph.findAllNodes("prim::GetAttr", recurse=True)
    params = {}
    param_tensors = {}
    packed_param_map = {}
    vars_by_name = {}
    seen = set()
    attr_name_sep = "_" if use_parser_friendly_name else "."

    for node in getattr_nodes:
        if _get_output_name(node) in seen:
            continue

        for getattrs in get_attr_chains(node):
            seen.update(map(_get_output_name, getattrs))

            full_attr = _getattr_full_name(getattrs, attr_name_sep)
            full_attr_node_name = _get_output_name(getattrs[-1])

            if full_attr.endswith("_packed_params"):  # for quantized models
                packed_param_map[full_attr_node_name] = full_attr
            elif full_attr in state_dict:
                if full_attr in vars_by_name:
                    var = vars_by_name[full_attr]
                else:
                    torch_tensor = state_dict[full_attr]
                    tensor, var = _get_tensor_and_var(torch_tensor, full_attr)
                    param_tensors[full_attr] = tensor
                    vars_by_name[full_attr] = var
                params[full_attr_node_name] = var

    return params, param_tensors, packed_param_map


def get_all_op_names(graph):
    """Return all operator names in the input graph"""
    nodes = list(graph.nodes())
    prim_with_blocks = ["prim::If", "prim::Loop"]
    for prim in prim_with_blocks:
        prim_nodes = graph.findAllNodes(prim, recurse=True)
        for prim_node in prim_nodes:
            for block in prim_node.blocks():
                nodes += block.nodes()
    return set(node.kind() for node in nodes)


def from_pytorch_traced_script(
    script_module,
    input_infos,
    custom_convert_map=None,
    default_dtype="float32",
    use_parser_friendly_name=False,
    keep_quantized_weight=False,
):
    """Load PyTorch model in the form of a scripted PyTorch model and convert into relax.
    The companion parameters will be handled automatically.

    Parameters
    ----------
    script_module : TopLevelTracedModule object
        TorchScripted PyTorch graph
        Note: We currently only support traces (ie: torch.jit.trace(model, input))

    input_infos : List of tuples
        Can be (input name, input shape) or (input name, (input shape, input types))
        Graph level input shape and type list
        The same input names need to be used for deployment, so choose easy to
        remember names (such as: input0, input1)
        e.g.
        [('input0', (1, 2)), ('input1', (3, 4))]
        or
        [('input0', ((1, 2), 'int')), ('input1', ((3, 4), 'float'))]

    custom_convert_map : Dictionary of str to Relay op
        A custom op conversion map in the same format as _convert_map above

    default_type : str
        The default dtype to use when type information is not provided by PyTorch.

    use_parser_friendly_name : bool
        When True, replace '.' with `_' in a original parameter name.
        The Relay text parser treats a variable name followed by a period as a tuple element access,
        so a variable name like "dense.weight" cannot be parsed correctly.
        Use this option when you want to run the AnnotateSpans pass on the imported module.

    keep_quantized_weight : bool
        Return quantized weights and bias, rather than float ones. PyTorch stores quantized weights
        in a custom format, so we cannot directly access 8 bit weights as Numpy arrays. We use
        a PyTorch function to unpack quantized weights into float32 arrays and quantization
        parameters. By default, we return float32 weights and rely on the QNN lowering and the
        Relay constant folding pass to quantize weights at compile time. In BYOC use cases, however,
        we cannot apply the constant folding pass on a QNN graph. If keep_quantized_weight is True,
        we quantize weights in the frontend using a function that is equivalent to
        qnn.op.quantize(...) operating on Numpy arrays.

    Returns
    -------
    mod : tvm.IRModule
        The module that optimizations will be performed on.

    params : dict of str to tvm.runtime.NDArray
        Dict of converted parameters stored in tvm.runtime.ndarray format
    """
    import torch

    mod = tvm.IRModule()
    prelude = Prelude(mod)
    enable_lower_all_tuples = True

    converter = PyTorchOpConverter(prelude, default_dtype)

    graph = script_module.graph.copy()

    # Check if lower_all_tuples pass can be enabled
    graph_inputs = list(graph.inputs())
    for inp in graph_inputs:
        if inp.type().kind() == "TupleType" or inp.type().kind() == "ListType":
            enable_lower_all_tuples = False
            break
    _run_jit_passes(graph, enable_lower_all_tuples)

    if custom_convert_map:
        converter.update_convert_map(custom_convert_map)

    op_names = get_all_op_names(graph)
    # converter.report_missing_conversion(op_names)

    is_module = isinstance(script_module, torch.jit.ScriptModule)
    params = script_module.state_dict() if is_module else {}
    outputs = _get_relax_input_vars(
        graph, input_infos, prelude, default_dtype=default_dtype, is_module=is_module
    )

    if use_parser_friendly_name:
        new_names = [key.replace(".", "_") for key in params.keys()]
        params = dict(zip(new_names, params.values()))

    param_vars, tensors, packed_param_map = convert_params(graph, params, use_parser_friendly_name)

    tvm_params = {k: tvm.nd.array(v) for k, v in tensors.items()}

    outputs.update(param_vars)
    ret_name = _get_input_names(graph.return_node())

    # TODO(@tvm-team): Skip handling quantization

    outputs = converter.convert_operators(_get_operator_nodes(graph.nodes()), outputs, ret_name)
    assert 0
    # ListConstruct kept original python list. Convert to tuple.
    outputs = [_expr.Tuple(output) if isinstance(output, list) else output for output in outputs]

    if len(outputs) > 1:
        ret = _expr.Tuple(outputs)
    else:
        ret = outputs[0]

    # Separate data inputs and parameters to make sure data inputs come first.
    func_args = []
    data_inputs = []
    for arg in _analysis.free_vars(ret):
        if arg.name_hint not in tvm_params.keys():
            data_inputs.append(arg)
        else:
            func_args.append(arg)
    func_args = data_inputs + func_args

    mod["main"] = tvm.relax.Function(func_args, ret)
    assert 0
