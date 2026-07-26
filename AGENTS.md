# Auris project instructions

## Python environment

- Use `D:\AI\Auris\reader\.venv` as the project's Python environment.
- Run Python commands from the repository root with
  `reader\.venv\Scripts\python.exe`, or from `reader` with
  `.venv\Scripts\python.exe`.
- Do not use the system `python` command for project tests, scripts, or
  dependency checks.
- Run the full test suite from `reader` with:
  `.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"`

## Browser-based UI verification

- For every browser-based UI check, use local Playwright instead of the Codex
  in-app Browser integration.
- Capture a screenshot and inspect browser console errors when validating a
  rendered frontend change, unless the task does not require visual validation.
