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

// clang-format off
#include <dlpack/dlpack.h>
//#include <tvm/node/reflection.h>
//#include<tvm / relax / tuning_api.h>
#include <tvm/runtime/module.h>
//#include <tvm/runtime/ndarray.h>
//#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>

#include <ATen/DLConvertor.h>
#include <ATen/dlpack.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <chrono>
// torch/csrc/autograd/profiler.h

namespace tvm {
namespace relax {

double time_cpy=0;
static void monly_deleter(DLManagedTensor* self) { delete self; }

class TorchFallbackRuntimeNode : public tvm::runtime::ModuleNode {
 public:
  /*! \brief The symbol name */
  String symbol_name_;
  /*! \brief The path to serialized submodules */
  String path_serialized_format_;

  int num_inputs_;
  int num_outputs_;

  torch::jit::script::Module torch_mod;
  TorchFallbackRuntimeNode(String symbol_name, String path_serialized_format, int num_inputs,
                           int num_outputs)
      : symbol_name_(symbol_name),
        path_serialized_format_(path_serialized_format),
        num_inputs_(num_inputs),
        num_outputs_(num_outputs) { 
          at::init_num_threads();
          torch_mod = torch::jit::load(path_serialized_format_); 
          torch::NoGradGuard no_grad;
          torch_mod.eval();
          torch::autograd::profiler::RecordProfile guard("gemfield/gemfield.pt.trace.json");

          // Run on GPU
          torch::Device device = torch::kCPU;
          if (torch::cuda::is_available()) {
          //  std::cout << "CUDA is available!";
          //  device = torch::kCUDA;
          }
          torch_mod.to(device);
        }

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
              if (i < static_cast<int>(m.num_inputs())-1) {
                inputs.emplace_back(at::fromDLPack(inp));
              } else {
                outputs.emplace_back(at::fromDLPack(inp));
              }
            }
            //auto t1 = std::chrono::high_resolution_clock::now();
            auto res = torch_mod.forward(inputs);
            //auto t2 = std::chrono::high_resolution_clock::now();

            //double time_fwd = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()/1000.0;
            
            //std::cout << "==============================\n";
            //t1 = std::chrono::high_resolution_clock::now();
            // too bad. need explicit copy
            
            auto t1 = std::chrono::high_resolution_clock::now();
            if(res.isTuple()){
              std::cout << " ## copy tuple...\n";
              auto elems = res.toTuple()->elements();
              ICHECK(elems.size() == outputs.size());
              for(size_t i=0;i<elems.size();i++){
                outputs[i].copy_(elems[i].toTensor());
              }
            }else if(res.isTensor()){
              //std::cout << " ## copy tensor...\n";
              outputs[0].copy_(res.toTensor());  
            }else{
              ICHECK(0) << "Undefined output type.";
            }
            auto t2 = std::chrono::high_resolution_clock::now();
            double time_cpy = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()/1000.0;

            //std::cout << " -- fwd: " << time_fwd << " ms\n";
            std::cout << " -- cpy: " << time_cpy << " ms\n";
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
