# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Pure opt-in policy for the initial AscendC A2/A3 scheduler scope."""

from dataclasses import dataclass

AUTO_SCHEDULE = "tl.ascend_auto_schedule"
AUTO_SYNC = "tl.ascend_auto_sync"


class OptionsError(ValueError):
    """An enabled scheduler configuration contradicts supported semantics."""


@dataclass(frozen=True)
class ScheduleOptions:
    enabled: bool = False
    unified_sync: bool = False
    target_model: str = ""
    platform: str = ""


def _boolean(value, key):
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    # TVM PassContext bool options arrive as IntImm; importing TVM is unnecessary.
    if (hasattr(value, "dtype") and str(value.dtype) in ("bool", "int8", "int32", "int64") and
            hasattr(value, "value") and isinstance(value.value, int) and value.value in (0, 1)):
        return bool(value.value)
    raise OptionsError(f"{key} must be a boolean or integer 0/1, got {value!r}")


def schedule_enabled(config) -> bool:
    """Read only the opt-in key, preserving strict TVM-compatible booleans."""
    return _boolean((config or {}).get(AUTO_SCHEDULE, False), AUTO_SCHEDULE)


def resolve_options(config, target_model, platform) -> ScheduleOptions:
    """Validate enabled policy without modifying caller options.

    Absent/false auto-schedule leaves all legacy settings untouched. Enabled
    scheduling owns synchronization, so an explicitly disabled auto-sync is a
    contradiction. An absent auto-sync key permits unified synchronization.
    VS settings do not choose a second algorithm in this bootstrap.
    """
    config = {} if config is None else config
    enabled = schedule_enabled(config)
    if not enabled:
        return ScheduleOptions()
    target_model, platform = str(target_model), str(platform)
    if target_model != "ascendc" or platform not in ("A2", "A3"):
        raise OptionsError(
            "tl.ascend_auto_schedule initially supports target_model='ascendc' "
            f"and platform A2/A3; got target_model={target_model!r}, platform={platform!r}")
    if AUTO_SYNC in config and not _boolean(config[AUTO_SYNC], AUTO_SYNC):
        raise OptionsError("tl.ascend_auto_schedule requires unified synchronization; "
                           "remove explicit tl.ascend_auto_sync=False or set it to True")
    return ScheduleOptions(True, True, target_model, platform)
