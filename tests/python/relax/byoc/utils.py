from __future__ import annotations
import numpy as np
import tvm
from tvm import relax
import tvm.relay.testing
import subprocess
from tvm.script import relax as R
import tempfile
from tvm import meta_schedule as ms
from tvm.relax.testing import transform
from tvm.relax.transform.tuning_api import Trace

# MS tuning config
tune_config = ms.TuneConfig(
    strategy="evolutionary",
    num_trials_per_iter=4,
    max_trials_per_task=4,
    max_trials_global=8,
)


def gen_ground_truth(mod, target, dev, inputs):
    # Lower and run tuning
    # Since there is no default schedule for GPU in MS yet, this is necessary
    with tempfile.TemporaryDirectory() as work_dir:
        with tvm.transform.PassContext(trace=Trace(mod), opt_level=0):
            seq = tvm.transform.Sequential(
                [
                    transform.LowerWithRelayOpStrategyPass(target),
                    relax.transform.MetaScheduleTuneIRMod(
                        target,
                        config=tune_config,
                        work_dir=work_dir,
                        database=None,
                    ),
                    relax.transform.MetaScheduleApplyHistoryBest(target, work_dir=work_dir),
                ]
            )
            new_mod = seq(mod)
    assert relax.analysis.well_formed(new_mod)
    exec = relax.vm.build(new_mod, target, params={})
    vm = relax.VirtualMachine(exec, dev)
    return vm["main"](*inputs)


def check_executable(exec, dev, inputs, expected):
    vm = relax.VirtualMachine(exec, dev)
    # Measure the performance w/o tuning log
    out = vm["main"](*inputs)
    tvm.testing.assert_allclose(out.numpy(), expected)
