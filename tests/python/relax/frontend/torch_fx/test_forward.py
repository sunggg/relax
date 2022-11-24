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

import sys
import tempfile
import numpy as np
import torch
from torch.nn import Module
from torch.nn import functional as F

import tvm
import tvm.testing
from tvm import relax
from tvm import meta_schedule as ms
from tvm.relax.testing import transform


sys.setrecursionlimit(10000)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def verify_model(
    model_name,
    input_data=None,
    rtol=1e-5,
    atol=1e-5,
    use_cpu=False,
):
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

    if not use_cpu and torch.cuda.is_available():
        if isinstance(baseline_model, torch.nn.Module):
            baseline_model = baseline_model.cuda()
        baseline_input = [inp.cuda() for inp in baseline_input]

    with torch.no_grad():
        baseline_outputs = baseline_model(*[input.clone() for input in baseline_input])

    if isinstance(baseline_outputs, tuple):
        baseline_outputs = tuple(out.cpu().numpy() for out in baseline_outputs)
    else:
        baseline_outputs = (baseline_outputs.cpu().numpy(),)

    input_names = [f"y{idx}" for idx, _ in enumerate(baseline_input)]
    input_infos = dict(
        zip(input_names, [(list(tensor.shape), "float32") for tensor in baseline_input])
    )

    target, dev = tvm.target.Target("llvm --num-cores=16"), tvm.cpu()
    mod = relax.frontend.from_torch_fx(baseline_model, input_infos)

    with tempfile.TemporaryDirectory() as work_dir:
        with tvm.transform.PassContext(opt_level=3):
            mod = transform.LowerWithRelayOpStrategyPass(target)(mod)
            db = ms.relax_integration.tune_relax(
                mod=mod,
                params=None,
                target=target,
                work_dir=work_dir,
                num_trials_per_iter=10,
                max_trials_global=10,
                task_scheduler="round-robin",
            )
            ex = ms.relax_integration.compile_relax(db, mod, target, params=None)

    print(mod)
    vm = relax.VirtualMachine(ex, dev)

    # Set up TVM inputs
    inputs = [tvm.nd.array(inp.clone().cpu().numpy(), dev) for inp in baseline_input]

    # Run
    outputs = vm["main"](*inputs)

    if not isinstance(outputs, list):
        outputs = [outputs]

    # Compare with torch side results
    for i, baseline_outputs in enumerate(baseline_outputs):
        output = outputs[i].numpy()
        tvm.testing.assert_allclose(baseline_outputs, output, rtol=rtol, atol=atol)


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
def test_forward_model():
    from torchvision import models

    input_shape = (1, 3, 228, 228)
    dtype = "float32"
    input_data = torch.randn(input_shape).type(  # pylint: disable=no-member
        {
            "float32": torch.float32,  # pylint: disable=no-member
        }[dtype]
    )
    resnet = models.resnet18()
    resnet.eval()
    verify_model(resnet, input_data=input_data, use_cpu=True)


if __name__ == "__main__":
    test_forward_model()
    # test_forward_mixed()
    # test_forward_matmul()
    # test_forward_abs()
    # tvm.testing.main()
