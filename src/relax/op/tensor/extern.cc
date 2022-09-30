/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/relay/backend/contrib/libtorch/codegen.cc
 * \brief Implementation of libtorch codegen.
 */

//#include <ATen/DLConvertor.h>
//#include <dlpack/dlpack.h>
//#include <torch/csrc/jit/api/compilation_unit.h>
//#include <torch/csrc/jit/serialization/import.h>
//#include <torch/torch.h>
//#include <tvm/ir/node/reflection.h>

#include <tvm/relax/expr.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/relay/op.h>
/*
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/contrib/libtorch_runtime.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/op.h>

#include <fstream>
#include <numeric>
#include <sstream>
*/

//#include "../op_common.h"

//#include "../../utils.h"

namespace tvm {
namespace relax {

/*! \brief Attributes of a TorchFunction node */
struct ExternOpAttrs : public tvm::AttrsNode<ExternOpAttrs> {
  std::string extern_kind;
  std::string op_name;

  TVM_DECLARE_ATTRS(ExternOpAttrs, "relax.attrs.ExternOpAttrs") {
    TVM_ATTR_FIELD(extern_kind).set_default("torch").describe("External Execution Provider");
    TVM_ATTR_FIELD(op_name).set_default("").describe("Operator");
  }
};

TVM_REGISTER_NODE_TYPE(ExternOpAttrs);

Optional<Expr> InferShapeExternOp(const Call& call, DiagnosticContext diag_ctx) {
  Expr output_shape = call->args[2];
  return output_shape;
}

Type InferTypeExternOp(const Call& call, DiagnosticContext diag_ctx) {
  if (call->type_args.size() != 1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "type_args should have exact 1 output type.");
  }
  Type output_type = call->type_args[0];
  return output_type;
}

RELAY_REGISTER_OP("relax.extern_op")
    .set_attrs_type<ExternOpAttrs>()
    .set_num_inputs(5)
    .add_argument("args", "Tuple", "The input arguments.")
    .add_argument("output_shape", "Expr", "The output shape.")
    .set_attr<FInferType>("FInferType", InferTypeExternOp)
    .set_attr<FInferShape>("FInferShapeExternOp", InferShapeExternOp);

Expr MakeExternOp(std::string extern_kind, std::string op_name, Tuple args, Expr output_shape,
                  Type output_type) {
  auto attrs = make_object<ExternOpAttrs>();
  attrs->extern_kind = extern_kind;
  attrs->op_name = op_name;
  static const Op& op = Op::Get("relax.extern_op");
  return Call(op, {args, output_shape}, Attrs(attrs), {output_type});
}

TVM_REGISTER_GLOBAL("relax.op.extern_op").set_body_typed(MakeExternOp);

}  // namespace relax
}  // namespace tvm
