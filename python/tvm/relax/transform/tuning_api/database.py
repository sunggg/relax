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
"""Relax Tuning Pass Database"""
from typing import Any, Callable, List
import tvm
from tvm.runtime import Object
from .primitives import Trace

from tvm._ffi import register_object
from . import _ffi_api
from tvm.meta_schedule.arg_info import ArgInfo
from tvm.meta_schedule.database import Workload
from tvm.meta_schedule.utils import _json_de_tvm


@register_object("relax.tuning_api.TuningRecord")
class TuningRecord(Object):
    """The class of tuning records.

    Parameters
    ----------
    trace : relax.Trace
        The trace of the tuning record.
    run_secs : List[float]
        The run time of the tuning record.
    workload : Workload
        The workload of the tuning record.
    target : tvm.target.Target
        The target of the tuning record.
    args_info : List[ArgInfo]
        The argument information of the tuning record.
    """

    trace: Trace
    run_secs: List[float]
    workload: Workload
    target: tvm.target.Target
    args_info: List[ArgInfo]

    def __init__(
        self,
        trace: Trace,
        run_secs: List[float],
        workload: Workload,
        target: tvm.target.Target,
        args_info: List[ArgInfo],
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.TuningRecord,  # type: ignore # pylint: disable=no-member
            trace,
            run_secs,
            workload,
            target,
            args_info,
        )

    def as_json(self) -> Any:
        """Export the tuning record to a JSON string.

        Returns
        -------
        json_str : str
            The JSON string exported.
        """
        return _json_de_tvm(_ffi_api.TuningRecordAsJSON(self))  # type: ignore # pylint: disable=no-member

    @staticmethod
    def from_json(json_obj: Any) -> "TuningRecord":
        """Create a tuning record from a json object.

        Parameters
        ----------
        json_obj : Any
            The json object to parse.
        workload : Workload
            The workload.

        Returns
        -------
        tuning_record : TuningRecord
            The parsed tuning record.
        """
        return _ffi_api.TuningRecordFromJSON(json_obj)  # type: ignore # pylint: disable=no-member
