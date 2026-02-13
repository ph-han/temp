---
name: ai-model-python-style
description: Enforce Python coding style for AI model research code. Use when writing or refactoring Python related to model architecture, training, evaluation, data pipelines, or experiment tooling, with strict requirements for explicit type annotations and avoiding unnecessary comments.
---

# AI Model Python Style

Apply these rules to all Python edits.

## Core Rules

1. Add explicit type annotations for function parameters, return values, and important variables.
2. Avoid unnecessary comments; write comments only when non-obvious reasoning is required.
3. Prefer clear, research-friendly naming for model, tensor, batch, metric, and experiment concepts.
4. Keep functions focused and composable so experiments are easy to modify and compare.
5. Use `uv` for Python environment, dependency, and command execution workflows in this project.
6. Before any file edit, report the patch plan first: target files, exact intent, expected impact, and validation command.
7. Apply edits only after the patch plan has been shown to the user in the same turn.

## Editing Checklist

1. Check new and modified functions for complete type hints.
2. Remove comments that restate obvious code behavior.
3. Keep docstrings concise and useful for API usage and tensor shape expectations.
4. Preserve behavior while improving readability and experiment iteration speed.
5. Run project Python commands via `uv` (for example, `uv run ...`) instead of direct `python`/`pip` usage when possible.
6. Confirm that a pre-edit patch report was provided before making changes.
