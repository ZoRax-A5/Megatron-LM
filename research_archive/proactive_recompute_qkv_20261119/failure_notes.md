# Intermediate test failures

1. The first QKV fixture snapshot assumed every `state_dict()` item was a Tensor. A legal `None` buffer caused `AttributeError` while cloning the fixture. The fixture was changed to preserve `None`; the final QKV test passes.
2. The repeated-step lifecycle test initially omitted the `pending_recompute_checkpoints` import and raised `NameError`. The import was added; the final suite passes.
3. A single-GPU attention fixture supplied a `ProcessGroupCollection` without the optional DP group. The runtime guard was updated to use the explicit group when present and to require a single-process default group otherwise; the final suite passes.

The full output for the second failure is preserved in `pytest_attempt_before_step_lifecycle_fix.log`; the third is in `pytest_attempt_incomplete_dp_group.log`. Final output is in `final_pytest.log`.
