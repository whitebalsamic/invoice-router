import os
import platform
import re
import subprocess
from fnmatch import fnmatch
from functools import lru_cache
from typing import List, Optional

from pydantic import BaseModel

from .schema import AppConfig, ProcessingMachineProfile


class MachineInfo(BaseModel):
    system: str
    architecture: str
    cpu_count: int
    memory_gb: Optional[float] = None
    model_name: Optional[str] = None
    model_identifier: Optional[str] = None


def _read_command_output(cmd: List[str]) -> Optional[str]:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip() or None
    except Exception:
        return None


def _detect_memory_gb(system_name: str) -> Optional[float]:
    if system_name == "Darwin":
        output = _read_command_output(["sysctl", "-n", "hw.memsize"])
        if output and output.isdigit():
            return round(int(output) / (1024**3), 1)
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        return round((page_size * page_count) / (1024**3), 1)
    except (AttributeError, OSError, ValueError):
        return None


def _detect_macos_model_info() -> tuple[Optional[str], Optional[str]]:
    profiler_output = _read_command_output(["system_profiler", "SPHardwareDataType"])
    model_name = None
    model_identifier = None
    if profiler_output:
        name_match = re.search(r"Model Name:\\s*(.+)", profiler_output)
        identifier_match = re.search(r"Model Identifier:\\s*(.+)", profiler_output)
        if name_match:
            model_name = name_match.group(1).strip()
        if identifier_match:
            model_identifier = identifier_match.group(1).strip()
    if not model_identifier:
        model_identifier = _read_command_output(["sysctl", "-n", "hw.model"])
    return model_name, model_identifier


@lru_cache(maxsize=1)
def _detect_machine_info() -> MachineInfo:
    system_name = platform.system()
    architecture = platform.machine()
    cpu_count = os.cpu_count() or 1
    memory_gb = _detect_memory_gb(system_name)
    model_name = None
    model_identifier = None
    if system_name == "Darwin":
        model_name, model_identifier = _detect_macos_model_info()
    return MachineInfo(
        system=system_name,
        architecture=architecture,
        cpu_count=cpu_count,
        memory_gb=memory_gb,
        model_name=model_name,
        model_identifier=model_identifier,
    )


def _pattern_matches(value: Optional[str], patterns: List[str]) -> bool:
    if not patterns:
        return True
    if not value:
        return False
    return any(fnmatch(value, pattern) for pattern in patterns)


def _machine_profile_matches(rule: ProcessingMachineProfile, machine: MachineInfo) -> bool:
    if rule.system and rule.system.lower() != machine.system.lower():
        return False
    if rule.architecture and rule.architecture.lower() != machine.architecture.lower():
        return False
    if not _pattern_matches(machine.model_name, rule.model_name_patterns):
        return False
    if not _pattern_matches(machine.model_identifier, rule.model_identifier_patterns):
        return False
    if rule.min_memory_gb is not None and (
        machine.memory_gb is None or machine.memory_gb < rule.min_memory_gb
    ):
        return False
    if rule.max_memory_gb is not None and (
        machine.memory_gb is None or machine.memory_gb > rule.max_memory_gb
    ):
        return False
    if rule.min_cpu_count is not None and machine.cpu_count < rule.min_cpu_count:
        return False
    if rule.max_cpu_count is not None and machine.cpu_count > rule.max_cpu_count:
        return False
    return True


def _apply_processing_profile(app_config: AppConfig, machine: Optional[MachineInfo] = None) -> None:
    processing = app_config.processing
    profiles = processing.profiles or {}
    machine = machine or _detect_machine_info()

    processing.detected_system = machine.system
    processing.detected_architecture = machine.architecture
    processing.detected_model_name = machine.model_name
    processing.detected_model_identifier = machine.model_identifier
    processing.detected_memory_gb = machine.memory_gb
    processing.detected_cpu_count = machine.cpu_count

    active_profile = None
    if processing.default_profile and processing.default_profile in profiles:
        default_profile = profiles[processing.default_profile]
        if default_profile.batch_size is not None:
            processing.batch_size = default_profile.batch_size
        if default_profile.worker_concurrency is not None:
            processing.worker_concurrency = default_profile.worker_concurrency
        active_profile = processing.default_profile

    if processing.auto_profile:
        for rule in processing.machine_profiles:
            profile = profiles.get(rule.profile)
            if profile is None or not _machine_profile_matches(rule, machine):
                continue
            if profile.batch_size is not None:
                processing.batch_size = profile.batch_size
            if profile.worker_concurrency is not None:
                processing.worker_concurrency = profile.worker_concurrency
            active_profile = rule.profile
            break

    processing.applied_profile = active_profile
