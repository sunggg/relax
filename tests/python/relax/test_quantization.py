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

from __future__ import annotations
import onnx
import tvm
from tvm import relay
from tvm.relay import transform
from tvm.relax.testing import relay_translator
from tvm.meta_schedule.integration import extract_task_from_relax, extract_task_from_relay
from os.path import exists
from tvm.contrib.download import download_testdata
import logging

log = logging.getLogger(__name__)


def import_onnx_with_qat(
    model_path,
    shape_dict,
    json_path,
    params_path,
):
    quant_model = onnx.load(model_path)
    mod, params = relay.frontend.from_onnx(quant_model, shape_dict, freeze_params=True)
    seq = tvm.transform.Sequential(
        [
            transform.InferType(),
            transform.FoldConstant(),
            transform.SimplifyInference(),
            transform.FoldScaleAxis(),
        ]
    )

    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)

    mod = tvm.relay.transform.FakeQuantizationToInteger(use_qat=True)(mod)
    # Lower qnn ops to relay ops
    mod = relay.qnn.transform.CanonicalizeOps()(mod)

    with open(json_path, "w") as fo:
        fo.write(tvm.ir.save_json(mod))

    with open(params_path, "wb") as fo:
        fo.write(relay.save_param_dict(params))
    return mod, params


def deserialize_relay(json_path, params_path):
    with open(json_path, "r") as fi:
        mod = tvm.ir.load_json(fi.read())

    with open(params_path, "rb") as fi:
        params = relay.load_param_dict(fi.read())

    return mod, params


def test_task_extraction():
    batch_size = 1
    seq_len = 128
    shape_dict = {
        "input_ids": (batch_size, seq_len),
        "segment_ids": (batch_size, seq_len),
        "input_mask": (batch_size, seq_len),
    }
    model_name = "bert-base-qat.onnx"
    json_path = "bert-base-int8.json"
    params_path = "bert-base-int8.params"

    if exists(json_path) and exists(params_path):
        relay_mod, params = deserialize_relay(json_path, params_path)
    else:
        # download model
        url = f"https://github.com/tlc-pack/TLCBench/raw/main/models/{model_name}"
        log.info("Downloading quantized bert-base model.")
        model_path = download_testdata(url, model_name, module="tlcbench")
        relay_mod, params = import_onnx_with_qat(model_path, shape_dict, json_path, params_path)

    target, dev = tvm.target.Target("cuda"), tvm.cuda()

    relax_mod = relay_translator.from_relay(relay_mod["main"])
    extracted_tasks = extract_task_from_relax(relax_mod, target)
    for i, tsk in enumerate(extracted_tasks):
        print(f"[{i}] {tsk.task_name}, {tsk.mod}")


if __name__ == "__main__":
    test_task_extraction()
