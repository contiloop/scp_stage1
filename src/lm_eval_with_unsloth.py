"""Run lm-eval-harness after importing Unsloth first.

This keeps benchmark subprocesses on the same patched import path as training/eval.
"""

import runpy

import unsloth  # must be imported before transformers/lm_eval


def main():
    try:
        runpy.run_module("lm_eval.__main__", run_name="__main__")
    except ImportError:
        runpy.run_module("lm_eval", run_name="__main__")


if __name__ == "__main__":
    main()
