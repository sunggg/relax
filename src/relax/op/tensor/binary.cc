/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file binary.cc
 * \brief binary broadcast operators.
 */

#include "binary.h"

namespace tvm {
namespace relax {

RELAX_REGISTER_BINARY_BROADCAST_OP("add")
    .describe("Elementwise add with broadcasting")
    .set_support_level(1);

RELAX_REGISTER_BINARY_BROADCAST_OP("multiply")
    .describe("Elementwise multiply with broadcasting")
    .set_support_level(1);

StructInfo InferStructInfoBroadcast(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 2) {
    ctx->ReportFatal(Diagnostic::Error(call) << "Binary broadcast op should have 2 arguments");
  }
  auto* sinfo0 = GetStructInfoAs<TensorStructInfoNode>(call->args[0]);
  auto* sinfo1 = GetStructInfoAs<TensorStructInfoNode>(call->args[1]);
  if (sinfo0 || sinfo1) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Both lhs and rhs should be Tensor for broadcasting, but got "
                     << call->args[0]->struct_info_->GetTypeKey() << " and "
                     << call->args[1]->struct_info_->GetTypeKey());
  }

  // Type deduction
  // data type 
  DataType output_dtype;
  if (sinfo0->IsUnknownDtype() || sinfo1->IsUnknownDtype()) {
    output_dtype = DataType::Void();
  } else if (sinfo0->dtype != sinfo1->dtype) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Data types " << sinfo0->dtype << " and " << sinfo1->dtype
                     << " must be equal for broadcasting operators");
  } else {
    output_dtype = sinfo0->dtype;
  }

  // ndims
  int output_ndim;
  if (sinfo0->IsUnknownNdim() || sinfo1->IsUnknownNdim()) {
    output_ndim = kUnknownNDim;
  } else {
    output_ndim = std::max(sinfo0->ndim, sinfo1->ndim);
  }

  auto* lhs_shape = sinfo0->shape.as<ShapeExprNode>();
  auto* rhs_shape = sinfo1->shape.as<ShapeExprNode>();
  // Shapes and ndims
  if (lhs_shape && rhs_shape) {
    // If all inputs have shapes, directly infer shapes
    std::vector<PrimExpr> output_shape;

    size_t lhs_ndim = sinfo0->ndim;
    size_t rhs_ndim = sinfo1->ndim;
    size_t max_ndim = std::max(lhs_ndim, rhs_ndim);

    size_t i = 1;
    for (; i <= std::min(lhs_ndim, rhs_ndim); ++i) {
      const PrimExpr& dim0 = lhs_shape->values[lhs_ndim - i];
      const PrimExpr& dim1 = rhs_shape->values[rhs_ndim - i];
      if (EqualConstInt(dim0, 1)) {
        output_shape.push_back(dim1);
      } else if (EqualConstInt(dim1, 1)) {
        output_shape.push_back(dim0);
      } else if (EqualCheck(dim0, dim1)) {
        output_shape.push_back(dim0);
      } else {
        // Use simple fallback when shape mismatch.
        return TensorStructInfo(output_dtype, /*ndim=*/output_ndim);
      }
    }
    auto& longer_shape = (lhs_ndim > rhs_ndim) ? lhs_shape : rhs_shape;
    for (; i <= max_ndim; ++i) {
      output_shape.push_back(longer_shape->values[max_ndim - i]);
    }
    Expr shape = ShapeExpr(Array<PrimExpr>(output_shape.rbegin(), output_shape.rend()));
    return TensorStructInfo(shape, output_dtype);
  } else {
    return TensorStructInfo(output_dtype, /*ndim=*/output_ndim);
  }
}

}  // namespace relax
}  // namespace tvm
