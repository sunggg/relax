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
from typing import List, Dict, Callable, Tuple
from numpy import iterable
import os

import torch
from torch import nn, fx
from torch.fx.passes.split_module import split_module

import tvm
from tvm import relax, topi
import numpy as np
import operator


class TorchFXTranslator:
    def __init__(self, module: fx.GraphModule) -> None:
        self.env = {}
        self.params = {}
        self.params_transpose = {}
        self.named_modules = dict(module.named_modules())
        self.bb = relax.BlockBuilder()
        self.create_convert_map()
        self.missing_info = dict()

    @staticmethod
    def _convert_data_type(input_type):
        """converts the PyTorch scalar type input_type to a TVM dtype."""

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

    @staticmethod
    def _fetch_attr(model, target: str):
        target_atoms = target.split(".")
        attr_itr = model
        for i, atom in enumerate(target_atoms):
            if not hasattr(attr_itr, atom):
                raise RuntimeError(
                    f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}"
                )
            attr_itr = getattr(attr_itr, atom)
        if isinstance(attr_itr, torch.Tensor):
            return TorchFXTranslator._convert_torch_tensor_to_relax(attr_itr)
        return attr_itr

    @staticmethod
    def _convert_torch_tensor_to_relax(tensor: torch.Tensor) -> relax.Var:
        ndim = len(tensor.data.shape)
        dtype = TorchFXTranslator._convert_data_type(str(tensor.data.dtype))
        return relax.const(tensor.data.cpu().numpy(), relax.DynTensorType(ndim, dtype))

    @staticmethod
    def shape_of(expr):
        if expr is None:
            # Subgraph for missing ops would not have shape info
            return relax.RuntimeDepShape()
        elif isinstance(expr, relax.Var):
            return expr.shape_
        elif isinstance(expr, torch.Tensor):
            return expr.shape
        elif isinstance(expr, relax.Constant):
            return expr.data.shape
        elif isinstance(expr, relax.Function):
            return expr.body.shape
        else:
            raise Exception(f"Please check your expr: {expr}")

    @staticmethod
    def type_of(expr):
        if expr is None:
            # Subgraph for missing ops would not have shape info
            return relax.DynTensorType(-1, None)
        elif isinstance(expr, relax.Constant):
            # TODO(@tvm-team): Since `relax.const` does not populate `checked_type` now,
            # we manually reconstruct type info as a temporary solution.
            ndim = len(expr.data.shape)
            dtype = TorchFXTranslator._convert_data_type(str(expr.data.dtype))
            return relax.DynTensorType(ndim, dtype)
        elif isinstance(expr, torch.Tensor) or isinstance(expr, relax.Var):
            return expr.checked_type
        elif isinstance(expr, relax.Function):
            return expr.body.checked_type
        else:
            raise Exception(f"Please check your expr: {expr}")

    def retrive_args(self, node):
        return self._retrive_args(node.args)

    def _retrive_args(self, node):
        if isinstance(node, fx.node.Node):
            return self.env[node]
        elif isinstance(node, tuple):
            return tuple(self._retrive_args(x) for x in node)
        elif isinstance(node, list):
            return [self._retrive_args(x) for x in node]
        elif isinstance(node, dict):
            return {self._retrive_args(k): self._retrive_args(v) for k, v in node.items()}
        else:
            return node

    @staticmethod
    def _promote_binary_op_args(lhs, rhs):
        if isinstance(lhs, relax.Expr) and isinstance(rhs, relax.Expr):
            return lhs, rhs
        elif isinstance(lhs, relax.Expr):
            assert isinstance(lhs.checked_type, relax.DynTensorType)
            return lhs, relax.const(rhs, lhs.checked_type.dtype)
        elif isinstance(rhs, relax.Expr):
            assert isinstance(rhs.checked_type, relax.DynTensorType)
            return relax.const(lhs, rhs.checked_type.dtype), rhs
        else:
            assert False

    def _call_binary_op(self, op, lhs, rhs):
        lhs, rhs = TorchFXTranslator._promote_binary_op_args(lhs, rhs)
        return self.bb.emit(op(lhs, rhs))

    def normalize_axes(self, axes, ndim):
        if not isinstance(axes, (tuple, list)):
            axes = [axes]
        axes = tuple(self._normalize_axis(axis, ndim) for axis in axes)
        return axes

    def _normalize_axis(self, axis, ndim):
        if axis < 0:
            axis += ndim
        if axis < 0 or axis >= ndim:
            raise ValueError("axis %d is out of bounds for array of dimension %d" % (axis, ndim))
        return axis

    def _add(self, node: fx.node.Node) -> relax.Var:
        lhs, rhs = self.retrive_args(node)
        if isinstance(lhs, relax.Var) or isinstance(rhs, relax.Var):
            return self._call_binary_op(relax.op.add, lhs, rhs)
        return lhs + rhs

    def _mul(self, node: fx.node.Node) -> relax.Var:
        lhs, rhs = self.retrive_args(node)
        if isinstance(lhs, relax.Var) or isinstance(rhs, relax.Var):
            return self._call_binary_op(relax.op.multiply, lhs, rhs)
        return lhs * rhs

    def _getitem(self, node: fx.node.Node) -> relax.Var:
        x = self.env[node.args[0]]

        if iterable(x):
            return x[node.args[1]]
        elif isinstance(x, relax.Var):
            if isinstance(x.shape, relax.Tuple):
                return self.bb.emit(relax.TupleGetItem(x, node.args[1]))
            assert 0
            """
            else:
                begin = []
            end = []
            stride = []
            axes = []
            expand_dim = []
            i = 0
            for index in node.args[1]:
                if isinstance(index, int):
                    begin.append(index)
                    end.append(index + 1)
                    stride.append(1)
                    axes.append(i)
                    i = i + 1
                elif isinstance(index, slice):
                    begin.append(0 if index.start is None else index.start)
                    end.append(x.shape_[i] if index.stop is None else index.stop)
                    stride.append(1 if index.step is None else index.step)
                    axes.append(i)
                    i = i + 1
                elif index is None:
                    expand_dim.append(i)
                else:
                    raise ValueError("Unsupported index type: " + str(type(index)))
            while i < len(x.shape_):
                begin.append(0)
                end.append(x.shape_[i])
                axes.append(i)
                i = i + 1
            sliced = self.bb.emit_te(topi.strided_slice, x, begin, end, stride, axes)
            sliced_shape = list(sliced.shape_)
            for i in expand_dim:
                sliced_shape.insert(i, 1)
            return self.bb.emit(relax.op.reshape(sliced, sliced_shape))
            """
        else:
            raise Exception(f"Pleaes check the tensor: {x}")

    # TODO(@tvm-team): More operators are implemented in https://github.com/mlc-ai/relax/pull/14
    # Migrate more as we introduce more ops in relax branch
    def create_convert_map(self):
        self.convert_map = {
            # Torch operators
            torch.add: self._add,
            torch.mul: self._mul,
            # Python builtin operators
            operator.add: self._add,
            operator.mul: self._mul,
            operator.getitem: self._getitem,
        }

    def fetch_attr(self, model, target: str):
        target_atoms = target.split(".")
        attr_itr = model
        for i, atom in enumerate(target_atoms):
            if not hasattr(attr_itr, atom):
                raise RuntimeError(
                    f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}"
                )
            attr_itr = getattr(attr_itr, atom)
        return attr_itr


def get_target_func(node: fx.node.Node, named_modules: Dict) -> Callable:
    if node.op == "call_method":
        return getattr(torch, node.target)
    elif node.op == "call_function":
        return node.target
    elif node.op == "call_module":
        return type(named_modules[node.target])
    else:
        raise RuntimeError(f"Invalid op type: {node}")


class FallbackManager:
    def __init__(self, supported_ops: List[Callable]) -> None:
        self.supported_ops = supported_ops

    def split(self, graph_module: fx.GraphModule, root_module: torch.nn.Module) -> fx.GraphModule:
        partition_id = 0
        was_missing = False

        def partition_callback(node: fx.node.Node):
            nonlocal partition_id, was_missing
            func = get_target_func(node, dict(graph_module.named_modules()))

            # if previous op was missing op,
            if was_missing:
                # put the current missing op into the same partition
                if func not in self.supported_ops:
                    was_missing = True
                # create the new partition for the supported op
                else:
                    was_missing = False
                    partition_id += 1
            # if previous op was supported,
            else:
                # create the new partition for the missing op
                if func not in self.supported_ops:
                    partition_id += 1
                    was_missing = True
                # keep the partition for the supported op
                else:
                    was_missing = False
            return partition_id

        return split_module(graph_module, root_module, partition_callback, keep_original_order=True)


def convert_submodule(module: fx.GraphModule, module_name: str, arg_info: List):
    graph = module.graph
    translator = TorchFXTranslator(module)
    inputs = {}
    arg_idx = 0
    for node in graph.nodes:
        if node.op == "placeholder":
            shape_, type_ = arg_info[arg_idx]
            inputs[node.name] = relax.Var(
                node.name,
                shape_annotation=shape_,
                type_annotation=type_,
            )
            arg_idx += 1
        elif node.op in ["call_function", "call_method", "call_module"]:
            func = get_target_func(node, translator.named_modules)
            if func not in translator.convert_map:
                return None

    # Translate model parameters.
    for _, param in module.named_parameters():
        ndim = len(param.data.shape)
        dtype = translator._convert_data_type(str(param.data.dtype))
        translator.params[param] = relax.const(
            param.data.cpu().numpy(), relax.DynTensorType(ndim, dtype)
        )

    translator.bb = relax.BlockBuilder()
    with translator.bb.function(name=module_name, params=list(inputs.values())):
        output = None
        with translator.bb.dataflow():
            for node in graph.nodes:
                if node.op == "placeholder":
                    assert (
                        node.name in inputs
                    ), "The function input {} is not found in the provided input information".format(
                        node.name
                    )
                    translator.env[node] = inputs[node.name]
                elif node.op == "output":
                    output = translator.bb.emit_output(translator.env[node.args[0]])
                    break
                elif node.op == "get_attr":
                    translator.env[node] = TorchFXTranslator._fetch_attr(module, node.target)
                elif node.op in ["call_function", "call_method", "call_module"]:
                    if node.op == "call_method":
                        func = getattr(torch, node.target)
                    elif node.op == "call_function":
                        func = node.target
                    else:
                        func = type(translator.named_modules[node.target])
                    assert func in translator.convert_map
                    translator.env[node] = translator.convert_map[func](node)
                else:
                    raise ValueError(f"Unsupported Fx op {node.op}")

        assert output is not None
        translator.bb.emit_func_output(output)

    relax_funcs = list(translator.bb.get().functions.values())
    assert len(relax_funcs) == 1
    return relax_funcs[0]


def extract_output_info(
    translator: TorchFXTranslator, node: fx.node.Node, submodule: fx.GraphModule
):
    test_inputs = []
    for arg_ in node.args:
        assert arg_ in translator.env
        relax_arg_ = translator.env[arg_]
        arg_shape = [int(i) for i in TorchFXTranslator.shape_of(relax_arg_)]
        np_input = np.random.rand(*arg_shape).astype(TorchFXTranslator.type_of(relax_arg_))
        test_inputs.append(torch.from_numpy(np_input))

    res = fx.Interpreter(submodule).run(*test_inputs)
    if isinstance(res, tuple):
        out_shapes, out_types = [], []
        for e in res:
            out_shape_ = tuple(e.shape)
            out_shapes.append(relax.ShapeExpr(out_shape_))
            dtype_ = TorchFXTranslator._convert_data_type(str(e.dtype))
            out_types.append(relax.DynTensorType(len(out_shape_), dtype_))
        out_shape = relax.Tuple(out_shapes)
        out_type = relax.TupleType(out_types)
    else:
        out_shape_ = tuple(res.shape)
        out_shape = relax.ShapeExpr(out_shape_)
        dtype_ = TorchFXTranslator._convert_data_type(str(res.dtype))
        out_type = relax.DynTensorType(len(out_shape_), dtype_)

    return (out_shape, out_type)


def from_torch_fx(model: torch.nn.Module, input_infos: Dict[str, Tuple]):
    symbolic_traced: fx.GraphModule = fx.symbolic_trace(model)
    translator = TorchFXTranslator(symbolic_traced)

    # FallbackManager will split a graph into two types of subgraphs
    # -> sugraph for supported ops, subgraph for missing ops
    graph_module: fx.GraphModule = FallbackManager(translator.convert_map.keys()).split(
        symbolic_traced, model
    )
    graph = graph_module.graph

    # Extract input names from the graph
    graph_input_names = [node.name for node in graph.nodes if node.op == "placeholder"]

    inputs = {}
    for graph_input_name, (user_assigned_name, (shape, dtype)) in zip(
        graph_input_names, input_infos.items()
    ):
        inputs[graph_input_name] = relax.Var(
            user_assigned_name, shape, relax.DynTensorType(len(shape), dtype)
        )

    # Translate model parameters.
    for _, param in model.named_parameters():
        ndim = len(param.data.shape)
        dtype = translator._convert_data_type(str(param.data.dtype))
        translator.params[param] = relax.const(
            param.data.cpu().numpy(), relax.DynTensorType(ndim, dtype)
        )

    # Since block builder does not allow the nestsed function generation,
    # we create a relax function for each submodule before creating the main function.
    for node in graph.nodes:
        if node.op == "placeholder":
            assert (
                node.name in inputs
            ), "The function input {} is not found in the provided input information".format(
                node.name
            )
            translator.env[node] = inputs[node.name]
        elif node.op == "get_attr":
            translator.env[node] = TorchFXTranslator._fetch_attr(model, node.target)
        elif node.op == "call_module":
            arg_info = []
            for arg in node.args:
                if arg in translator.env:
                    arg_info.append(
                        (
                            TorchFXTranslator.shape_of(translator.env[arg]),
                            TorchFXTranslator.type_of(translator.env[arg]),
                        )
                    )
                elif arg.op == "call_module":
                    # unknown_submodule = graph_module.get_submodule(arg.target)
                    # assert unknown_submodule in translator.missing_info
                    arg_info.append(translator.missing_info[arg])
                elif arg.op == "call_function":
                    arg_info.append(translator.missing_info[arg])
                else:
                    raise Exception("Unsupported arg type")

            submodule = graph_module.get_submodule(node.target)
            relax_func = convert_submodule(submodule, node.target, arg_info)
            if relax_func:
                translator.env[node] = relax_func
            else:
                translator.missing_info[node] = extract_output_info(translator, node, submodule)

        elif node.op == "call_function":
            func = node.target
            # Graph splitter often inserts getitem, so we need to handle it here
            assert func is operator.getitem

            if node.args[0] in translator.env:
                assert func in translator.convert_map
                translator.env[node] = translator.convert_map[func](node)
            else:
                tuple_shape_info, tuple_type_info = translator.missing_info[node.args[0]]
                idx = node.args[1]
                shape_info, type_info = tuple_shape_info[idx], tuple_type_info[idx]
                translator.missing_info[node] = (shape_info, type_info)

    # Initialize the block builder with a function and a dataflow block.
    # Construct the relax "main" function that calls each of submodule functions we created.
    bb = translator.bb
    ext_mods = list()
    with bb.function(name="main", params=list(inputs.values())):
        output = None
        with bb.dataflow():
            for node in graph.nodes:
                if node.op == "call_module":
                    submodule_name = node.target
                    submodule = graph_module.get_submodule(submodule_name)
                    relax_args = [translator.env[arg] for arg in node.args]

                    if node in translator.env:
                        gv = bb.add_func(translator.env[node], submodule_name)
                        caller = bb.emit(relax.Call(gv, relax_args))
                    else:
                        fallback_dir = "temp_fallback_submodules"
                        if not os.path.exists(fallback_dir):
                            os.mkdir(fallback_dir)

                        fallback_path = f"{fallback_dir}/{node.target}.pt"
                        # Serialize fallback modules
                        # NOTE: we use jit.script API only because it provides robust
                        #       serialization/deserialization. If we can find another API,
                        #       we can switch to the new one.
                        # torch.jit.script(submodule).save(f"{fallback_path}/{node.target}.pt")
                        torch.jit.script(submodule).save(fallback_path)

                        # Define global symbol
                        global_symbol = f"fallback_{node.target}"
                        num_inputs, num_outputs = len(node.args), 1

                        # Build BYOC runtime and wrap with TVM runtime::Module
                        gen_rt_mod = tvm.get_global_func("relax.CreateTorchFallbackRuntime")
                        assert gen_rt_mod
                        fallback_mod = gen_rt_mod(
                            global_symbol,
                            fallback_path,
                            num_inputs,
                            num_outputs,
                        )
                        ext_mods.append(fallback_mod)
                        # create external call with the global symbol
                        out_shape, out_type = translator.missing_info[node]
                        caller = bb.emit(
                            relax.Call(
                                op=tvm.ir.Op.get("relax.call_tir"),
                                args=[
                                    relax.ExternFunc(global_symbol),
                                    relax.Tuple(relax_args),
                                    out_shape,
                                ],
                                type_args=[out_type],
                            )
                        )

                    translator.env[node] = caller
                elif node.op == "output":
                    output = bb.emit_output(translator.env[node.args[0]])
                    break
                elif node.op == "call_function":
                    func = node.target
                    # Graph splitter often inserts getitem, so we need to handle it here
                    assert func is operator.getitem
                    assert node.args[0] in translator.env
                    assert func in translator.convert_map
                    translator.env[node] = translator.convert_map[func](node)

        assert output is not None
        bb.emit_func_output(output)
    out_mod = bb.get()
    return out_mod.with_attr("external_mods", ext_mods)
