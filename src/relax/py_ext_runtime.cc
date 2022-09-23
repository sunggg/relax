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
 * \file src/runtime/contrib/pymod/pymod.cc
 * \brief Runtime module for python external module.
 */
#include <tvm/node/reflection.h>
#include <tvm/relax/tuning_api.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>

//#ifndef TVM_PYTHON_EXTERNAL_RUNTIME_MODULE_
//#define TVM_PYTHON_EXTERNAL_RUNTIME_MODULE_
namespace tvm {
namespace relax {
class ExtRuntimeNode : public tvm::runtime::ModuleNode {
 public:
  /*! \brief The only subgraph name for this module. */
  String symbol_name_;
  /*! \brief The path to the shared library. */
  String path_generated_code_;
  /*! \brief The ffi key for runtime. */
  String ffi_key_runtime_;

  int num_inputs_;
  int num_outputs_;
  // TODO: support const bindings

  ExtRuntimeNode(String symbol_name, String path_generated_code, String ffi_key_runtime,
                 int num_inputs, int num_outputs)
      : symbol_name_(symbol_name),
        path_generated_code_(path_generated_code),
        ffi_key_runtime_(ffi_key_runtime),
        num_inputs_(num_inputs),
        num_outputs_(num_outputs) {}

  /*! \brief The default destructor. */
  virtual ~ExtRuntimeNode() = default;

  const char* type_key() const final { return "PyExtRuntime"; }

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
            static const tvm::runtime::PackedFunc* runtime_func =
                tvm::runtime::Registry::Get(ffi_key_runtime_);
            ICHECK(runtime_func);
            Array<ObjectRef> arr_args({path_generated_code_});
            arr_args.push_back(Integer(num_inputs_));
            for (int i = 0; i < num_inputs_; i++) {
              ICHECK(args[i].type_code() == kTVMNDArrayHandle ||
                     args[i].type_code() == kTVMDLTensorHandle)
                  << "Expect NDArray or DLTensor as inputs";
              arr_args.push_back(args[i]);
            }
            DLTensor* out_tensor = args[num_inputs_].operator DLTensor*();
            tvm::runtime::NDArray res = CallPackedWithArgsInArray(*runtime_func, arr_args);
            res.CopyTo(out_tensor);
          });
    } else {
      return tvm::runtime::PackedFunc(nullptr);
    }
  }
};

tvm::runtime::Module Get(String symbol_name, String path_generated_code, String ffi_key_runtime,
                         int num_inputs, int num_outputs) {
  auto n = make_object<ExtRuntimeNode>(symbol_name, path_generated_code, ffi_key_runtime,
                                       num_inputs, num_outputs);
  return tvm::runtime::Module(n);
}
TVM_REGISTER_GLOBAL("relax.CreatePyExtRuntime").set_body_typed(Get);

}  // namespace relax
}  // namespace tvm

//#endif  // TVM_PYTHON_EXTERNAL_RUNTIME_MODULE_
