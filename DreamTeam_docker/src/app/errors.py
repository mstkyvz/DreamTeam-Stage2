# -*- coding: utf-8 -*-
u"""Typed failure modes.

The contract says: exit 0 only after the complete output is written, nonzero on unrecoverable
errors. A bare `Exception` cannot express which of those happened, so every failure the pipeline
can produce is one of these, and `main` maps them to exit codes. `exit_code` is part of the type
because the judge reads the exit code, not the traceback.
"""
from __future__ import annotations


class PipelineError(Exception):
    u"""Base for every failure this image raises deliberately."""

    exit_code = 1


class UsageError(PipelineError):
    u"""Bad invocation: wrong argument count, missing input, input aliasing output."""

    exit_code = 2


class InputContractError(PipelineError):
    u"""The input CSV violates the organizer's schema (missing columns, duplicate row_id)."""

    exit_code = 2


class OutputContractError(PipelineError):
    u"""What we produced does not satisfy the submission contract. Never silently downgraded."""


class EngineError(PipelineError):
    u"""An inference engine failed to boot, or booted into an unexpected configuration."""


class EngineBootTimeout(EngineError):
    u"""Boot exceeded its budget. Distinguished from EngineError so retries can target it."""


class RouterUnavailable(PipelineError):
    u"""A router was requested whose inputs this image does not contain.

    Raised only when a caller explicitly demands the learned router. The default path degrades to
    margin routing with a loud warning instead, because a working submission beats a perfect one.
    """
