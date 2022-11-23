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
 * \file src/relax/torch_fallback_runtime_module.cc
 * \brief Runtime module for torch fallback external module.
 */
#include <ATen/DLConvertor.h>
#include <ATen/dlpack.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <tvm/node/reflection.h>
#include <tvm/relax/tuning_api.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>

namespace tvm {
namespace relax {

static void monly_deleter(DLManagedTensor* self) { delete self; }

class TorchFallbackRuntimeNode : public tvm::runtime::ModuleNode {
 public:
  /*! \brief The symbol name */
  String symbol_name_;
  /*! \brief The path to serialized submodules */
  String path_serialized_format_;

  int num_inputs_;
  int num_outputs_;

  TorchFallbackRuntimeNode(String symbol_name, String path_serialized_format, int num_inputs,
                           int num_outputs)
      : symbol_name_(symbol_name),
        path_serialized_format_(path_serialized_format),
        num_inputs_(num_inputs),
        num_outputs_(num_outputs) {}

  /*! \brief The default destructor. */
  virtual ~TorchFallbackRuntimeNode() = default;

  const char* type_key() const final { return "TorchFallbackRuntime"; }

  tvm::runtime::PackedFunc GetFunction(const std::string& name,
                                       const ObjectPtr<Object>& sptr_to_self) override {
    if (name == "get_symbol") {
      return tvm::runtime::PackedFunc(
          [sptr_to_self, this](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue* rv) {
            *rv = this->symbol_name_;
          });
    } else if (this->symbol_name_ == name) {
      return tvm::runtime::PackedFunc(
          [sptr_to_self, this](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue* rv) {
            std::vector<torch::jit::IValue> inputs;
            std::vector<torch::Tensor> outputs;
            torch::jit::script::Module torch_mod = torch::jit::load(path_serialized_format_);
            auto m = torch_mod.get_method("forward");

            for (int i = 0; i < args.size(); i++) {
              const DLTensor* arg;
              if (args[i].IsObjectRef<runtime::NDArray>()) {
                runtime::NDArray arr = args[i];
                arg = arr.operator->();
              } else {
                arg = args[i].operator DLTensor*();
              }
              DLManagedTensor* inp = new DLManagedTensor{};
              inp->dl_tensor = *arg;
              inp->deleter = &monly_deleter;
              // m.num_inputs includes the self argument of forward(self, ...)
              // num_inputs - 1 is the number of (Tensor) inputs
              if (i < static_cast<int>(m.num_inputs()) - 1) {
                inputs.emplace_back(at::fromDLPack(inp));
              } else {
                outputs.emplace_back(at::fromDLPack(inp));
              }
            }
            ICHECK(outputs.size() == 1) << "wrong number of args, can handle only one output";
            torch::Tensor res = torch_mod.forward(inputs).toTensor();
            //*rv = runtime::NDArray::FromDLPack(at::toDLPack(res));
            outputs[0].copy_(res);  // too bad
          });
    } else {
      return tvm::runtime::PackedFunc(nullptr);
    }
  }
};  // namespace relax

tvm::runtime::Module Get(String symbol_name, String path_shared_library, int num_inputs,
                         int num_outputs) {
  auto n = make_object<TorchFallbackRuntimeNode>(symbol_name, path_shared_library, num_inputs,
                                                 num_outputs);
  return tvm::runtime::Module(n);
}
TVM_REGISTER_GLOBAL("relax.CreateTorchFallbackRuntime").set_body_typed(Get);

}  // namespace relax
}  // namespace tvm
