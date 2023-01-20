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
# pylint: disable=import-self, invalid-name, unused-argument
"""Unit tests for various models and operators"""

import sys, os
import tempfile
import numpy as np
from torch.nn import Module
from torch.nn import functional as F
from typing import Any, Dict
import torch
from torch.profiler import profile, record_function, ProfilerActivity

import tvm
import tvm.testing
from tvm import relax
from tvm import meta_schedule as ms
from tvm.relax.testing import transform
import time
import copy

sys.setrecursionlimit(10000)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def verify_model(model_name, input_data=None, rtol=1e-5, atol=1e-5, use_cpu=False, num_runs=20):
    """Assert that the output of a compiled model matches with that of its
    baseline."""
    input_data = [] if input_data is None else input_data

    if isinstance(input_data, list):
        baseline_model = model_name
        baseline_input = input_data
    elif isinstance(input_data, torch.Tensor) or not input_data.shape:
        baseline_model = model_name
        baseline_input = [input_data]
    else:
        assert False, "Unexpected input format"

    with torch.no_grad():
        torch_input = copy.deepcopy(baseline_input)
        torch_model = copy.deepcopy(baseline_model)
        if not use_cpu and torch.cuda.is_available():
            if isinstance(torch_model, torch.nn.Module):
                torch_model = torch_model.cuda()
            torch_input = [inp.cuda() for inp in torch_input]

        torch.jit.script(torch_model).save("torch_baseline.tmp")
        torch_script_mod = torch.jit.load("torch_baseline.tmp")
        """
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                # torch.profiler.ProfilerActivity.CUDA
            ],
            record_shapes=True,
            # In this example with wait=1, warmup=1, active=2,
            # profiler will skip the first step/iteration,
            # start warming up on the second, record
            # the third and the forth iterations, by gemfield.
            # schedule=torch.profiler.schedule(wait=1, warmup=1, active=2),
            # on_trace_ready=torch.profiler.tensorboard_trace_handler("./gemfield"),
        ) as prof:
            for iter in range(num_runs):
                start = time.time()
                torch_outputs = torch_script_mod(*[input.clone() for input in torch_input])
                end = time.time()
                prof.step()
        print(prof.key_averages())
        """
        for iter in range(num_runs):
            start = time.time()
            torch_outputs = torch_script_mod(*[input.clone() for input in torch_input])
            end = time.time()
        torch_time = (end - start) * 1000

        # print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))
        # send a signal to the profiler that the next iteration has started
        #    p.step()

        # assert 0

        # for i in range(num_runs):
        #    start = time.time()
        #    # baseline_outputs = baseline_model(*[input.clone() for input in baseline_input])
        #    baseline_outputs = baseline_script_mod(*[input.clone() for input in baseline_input])
        #    end = time.time()
        # torch_time = (end - start) * 1000

    if isinstance(torch_outputs, tuple):
        torch_outputs = tuple(out.cpu().numpy() for out in torch_outputs)
    else:
        torch_outputs = (torch_outputs.cpu().numpy(),)

    input_names = [f"y{idx}" for idx, _ in enumerate(baseline_input)]
    input_infos = dict(
        zip(input_names, [(list(tensor.shape), "float32") for tensor in baseline_input])
    )

    if use_cpu:
        target, dev = tvm.target.Target("llvm --num-cores=64"), tvm.cpu()
    else:
        target, dev = tvm.target.Target("nvidia/geforce-rtx-3070"), tvm.cuda()

    mod = relax.frontend.from_torch_fx(baseline_model, input_infos)
    assert relax.analysis.well_formed(mod)

    with tempfile.TemporaryDirectory() as work_dir:
        with tvm.transform.PassContext(opt_level=3):
            mod = transform.LowerWithRelayOpStrategyPass(target)(mod)
            db = ms.relax_integration.tune_relax(
                mod=mod,
                params=None,
                target=target,
                work_dir=work_dir,
                max_trials_global=50,
                task_scheduler="round-robin",
            )
            assert relax.analysis.well_formed(mod)
            ex = ms.relax_integration.compile_relax(db, mod, target, params=None)

    vm = relax.VirtualMachine(ex, dev)

    # Set up TVM inputs
    inputs = [tvm.nd.array(inp.clone().numpy(), dev) for inp in baseline_input]

    # Run
    for i in range(num_runs):
        start = time.time()
        outputs = vm["main"](*inputs)
        end = time.time()

    # start = time.time()
    # outputs = vm["main"](*inputs)
    # end = time.time()
    tvm_time = (end - start) * 1000
    # assert 0

    print(f"Elapsed time (Torch) : {torch_time:.3f} ms")
    print(f"Elapsed time (TVM)   : {tvm_time:.3f} ms")

    if not isinstance(outputs, list):
        outputs = [outputs]

    # Compare with torch side results
    # for i, torch_output in enumerate(torch_outputs):
    #    output = outputs[i].numpy()
    #    tvm.testing.assert_allclose(torch_output, output, rtol=rtol, atol=atol)


@tvm.testing.uses_gpu
def test_forward_add():
    torch.set_grad_enabled(False)
    input_shape = [10]

    class Mod1(Module):
        def forward(self, x0, x1):
            # use python builtin op
            return x0 + x1

    class Mod2(Module):
        def forward(self, x0, x1):
            # use torch op
            return torch.add(x0, x1)

    class Mod3(Module):
        def forward(self, x):
            return x + 1

    class Mod4(Module):
        def forward(self, x):
            return torch.add(x, 1)

    class Mod5(Module):
        def forward(self, x):
            y = torch.ones(input_shape, dtype=torch.float)
            if torch.cuda.is_available():
                y = y.cuda()
            return x + y

    class Mod6(Module):
        def forward(self, x):
            y = torch.ones(input_shape, dtype=torch.float)
            if torch.cuda.is_available():
                y = y.cuda()
            return torch.add(x, y)

    input_data = torch.rand(input_shape).float()
    verify_model(Mod1(), input_data=[input_data, input_data])
    verify_model(Mod2(), input_data=[input_data, input_data])
    verify_model(Mod3(), input_data=input_data)
    verify_model(Mod4(), input_data=input_data)
    verify_model(Mod5(), input_data=input_data)
    verify_model(Mod6(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_matmul():
    torch.set_grad_enabled(False)
    input_shape = [10]

    class Mod1(Module):
        def forward(self, x0, x1):
            # use python builtin op
            return x0 * x1

    class Mod2(Module):
        def forward(self, x0, x1):
            # use torch op
            return torch.mul(x0, x1)

    class Mod3(Module):
        def forward(self, x):
            return x * 2

    class Mod4(Module):
        def forward(self, x):
            return torch.mul(x, 2)

    class Mod5(Module):
        def forward(self, x):
            y = torch.ones(input_shape, dtype=torch.float) * 2
            if torch.cuda.is_available():
                y = y.cuda()
            return x * y

    class Mod6(Module):
        def forward(self, x):
            y = torch.ones(input_shape, dtype=torch.float) * 2
            if torch.cuda.is_available():
                y = y.cuda()
            return torch.mul(x, y)

    input_data = torch.rand(input_shape).float()
    verify_model(Mod1(), input_data=[input_data, input_data])
    verify_model(Mod2(), input_data=[input_data, input_data])
    verify_model(Mod3(), input_data=input_data)
    verify_model(Mod4(), input_data=input_data)
    verify_model(Mod5(), input_data=input_data)
    verify_model(Mod6(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_abs():
    torch.set_grad_enabled(False)
    input_shape = [10]

    class Mod1(Module):
        def forward(self, x):
            return torch.abs(x)

    input_data = torch.rand(input_shape).float() - 0.5
    verify_model(Mod1(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_mixed():
    torch.set_grad_enabled(False)
    input_shape = [10]

    class Mod1(Module):
        def forward(self, x):
            return torch.abs(x) * x + x

    input_data = torch.rand(input_shape).float() - 0.5
    verify_model(Mod1(), input_data=input_data)


# @tvm.testing.uses_gpu
def test_forward_model(name):
    from torchvision import models

    if name in ["resnet3d_18"]:
        input_shape = (1, 3, 3, 228, 228)
    else:
        input_shape = (1, 3, 228, 228)

    print(f"Run {name}..., input shape {input_shape}")
    params: Dict[str, Any] = {}
    if name in ["resnet_18", "resnet_50"]:
        model = getattr(models, name.replace("_", ""))
    elif name == "wide_resnet_50":
        model = getattr(models, "wide_resnet50_2")
    elif name == "resnext_50":
        model = getattr(models, "resnext50_32x4d")
    elif name == "mobilenet_v2":
        model = getattr(models, name)
    elif name == "mobilenet_v3":
        model = getattr(models, name + "_large")
    elif name == "inception_v3":
        model = getattr(models, name)
        params["aux_logits"] = False
    elif name == "densenet_121":
        model = getattr(models, name.replace("_", ""))
    elif name == "resnet3d_18":
        model = models.video.r3d_18
    elif name == "vgg_16":
        model = getattr(models, name.replace("_", ""))
    try:
        model = model(**params, weights=None)
    except TypeError:
        model = model(**params, pretrained=False)

    dtype = "float32"
    input_data = torch.randn(input_shape).type(  # pylint: disable=no-member
        {
            "float32": torch.float32,  # pylint: disable=no-member
        }[dtype]
    )

    model.eval()
    verify_model(model, input_data=input_data, use_cpu=True)
    print("Completed!")


def test_forward_bert(name="bert_base"):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import transformers
    from transformers.models.bert import modeling_bert as bert

    config_dict = {
        "bert_tiny": transformers.BertConfig(
            num_hidden_layers=6,
            hidden_size=512,
            intermediate_size=2048,
            num_attention_heads=8,
            return_dict=False,
        ),
        "bert_base": transformers.BertConfig(
            num_hidden_layers=12,
            hidden_size=768,
            intermediate_size=3072,
            num_attention_heads=12,
            return_dict=False,
        ),
        "bert_medium": transformers.BertConfig(
            num_hidden_layers=12,
            hidden_size=1024,
            intermediate_size=4096,
            num_attention_heads=16,
            return_dict=False,
        ),
        "bert_large": transformers.BertConfig(
            num_hidden_layers=24,
            hidden_size=1024,
            intermediate_size=4096,
            num_attention_heads=16,
            return_dict=False,
        ),
    }
    configuration = config_dict[name]
    model = bert.BertSelfAttention(configuration)

    dtype = "int64"
    input_data = torch.randint(10000, (768, 768, 768)).type(torch.float32)
    model.eval()
    verify_model(model, input_data=input_data, use_cpu=True)
    print("Completed!")


if __name__ == "__main__":
    model_names = [
        "resnet_18",
        # "resnet_50",
        # "mobilenet_v2",
        # "mobilenet_v3",
        # "wide_resnet_50",
        # "resnext_50",
        # "inception_v3",
        # "densenet_121",
        # "vgg_16",
        # "resnet3d_18",
    ]
    # test_forward_model("resnet_18")
    # test_forward_model("resnet_50")
    test_forward_model("resnext_50")
    # test_forward_bert()
    # test_forward_mixed()
    # test_forward_matmul()
    # test_forward_abs()
    # tvm.testing.main()
