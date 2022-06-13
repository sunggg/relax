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
 * \file src/relax/transform/tuning_api/database.cc
 * \brief Implementation of tuning APIs.
 */

#include <tvm/relax/tuning_api.h>

#include "../../../meta_schedule/utils.h"

namespace tvm {
namespace relax {

// TODO(sunggg): Do we need workload in TuningRecord?
TuningRecord::TuningRecord(Trace trace, Array<FloatImm> run_secs, meta_schedule::Workload workload,
                           Target target, Array<meta_schedule::ArgInfo> args_info) {
  ObjectPtr<TuningRecordNode> n = make_object<TuningRecordNode>();
  n->trace = std::move(trace);
  n->run_secs = std::move(run_secs);
  n->workload = std::move(workload);
  n->target = std::move(target);
  n->args_info = std::move(args_info);
  data_ = std::move(n);
}

ObjectRef TuningRecordNode::AsJSON() const {
  Array<ObjectRef> json_args_info;
  json_args_info.reserve(args_info.size());
  for (const meta_schedule::ArgInfo& arg_info : args_info) {
    json_args_info.push_back(arg_info->AsJSON());
  }
  return Array<ObjectRef>{trace->AsJSON(), run_secs, workload->AsJSON(), target->Export(),
                          json_args_info};
}

TuningRecord TuningRecord::FromJSON(const ObjectRef& json_obj) {
  Trace trace{nullptr};
  Array<FloatImm> run_secs{nullptr};
  meta_schedule::Workload workload{nullptr};
  Target target{nullptr};
  Array<meta_schedule::ArgInfo> args_info;
  try {
    const ArrayNode* json_array = json_obj.as<ArrayNode>();
    CHECK(json_array && json_array->size() == 5);
    // Load json[0] => trace
    const ObjectRef& json_trace = json_array->at(0);
    trace = Trace::FromJSON(json_trace);
    // Load json[1] => run_secs
    run_secs = Downcast<Array<FloatImm>>(json_array->at(1));
    // Load json[2] => workload
    const ObjectRef& json_workload = json_array->at(2);
    workload = meta_schedule::Workload::FromJSON(json_workload);
    // Load json[3] => target
    target = Target(Downcast<Map<String, ObjectRef>>(json_array->at(3)));
    // Load json[4] => args_info
    {
      const ArrayNode* json_args_info = json_array->at(4).as<ArrayNode>();
      args_info.reserve(json_args_info->size());
      for (const ObjectRef& json_arg_info : *json_args_info) {
        args_info.push_back(meta_schedule::ArgInfo::FromJSON(json_arg_info));
      }
    }
  } catch (const std::runtime_error& e) {  // includes tvm::Error and dmlc::Error
    LOG(FATAL) << "ValueError: Unable to parse the JSON object: " << json_obj
               << "\nThe error is: " << e.what();
  }
  return TuningRecord(trace, run_secs, workload, target, args_info);
}

/**************** FFI ****************/
TVM_REGISTER_NODE_TYPE(TuningRecordNode);
TVM_REGISTER_GLOBAL("relax.tuning_api.TuningRecord")
    .set_body_typed([](Trace trace, Array<FloatImm> run_secs, meta_schedule::Workload workload,
                       Target target, Array<meta_schedule::ArgInfo> args_info) {
      return TuningRecord(trace, run_secs, workload, target, args_info);
    });

TVM_REGISTER_GLOBAL("relax.tuning_api.TuningRecordAsJSON")
    .set_body_method<TuningRecord>(&TuningRecordNode::AsJSON);

TVM_REGISTER_GLOBAL("relax.tuning_api.TuningRecordFromJSON").set_body_typed(TuningRecord::FromJSON);

// TVM_REGISTER_NODE_TYPE(JSONDatabaseNode);
// TVM_REGISTER_GLOBAL("relax.tuning_API.DatabaseJSONDatabase").set_body_typed(Database::JSONDatabase);
}  // namespace relax
}  // namespace tvm
