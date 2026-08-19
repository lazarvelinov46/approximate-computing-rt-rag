"""Knob abstraction: each knob toggles the pipeline from precise to increasingly approximate.

A 'setting' is a dict picking one value per knob. The baseline setting uses every knob's
first (precise) value. Sweeps vary ONE knob while holding the rest at baseline.

TODO(Phase 2): iter_single_knob_settings(knobs_cfg) -> yields settings
TODO(Phase 4): iter_joint_settings(knobs_cfg, axes) -> yields grid settings
"""
def baseline_setting(knobs_cfg):
    raise NotImplementedError("Phase 2")
