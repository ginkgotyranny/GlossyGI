# Repository Guidelines

## Project Structure

- `Source/Falcor` contains the core C++ framework; `Source/Modules`, `RenderPasses`, `Samples`, `Tools`, `Mogwai`, and `plugins` hold extensions and applications.
- Keep shaders beside their host pass or test and use the repository suffixes (for example, `.cs.slang` or `.rt.slang`).
- `data/` and `media/` contain runtime assets, `docs/` contains developer documentation, and `tests/` contains unit, Python, and image tests. Generated files belong in ignored `build/`, `tests/data/`, or Packman directories.

## Build and Development

Run commands from the repository root after cloning submodules:

```bat
setup.bat
cmake --preset windows-ninja-msvc
cmake --build build/windows-ninja-msvc --config Release
```

Use `setup_vs2022.bat` for a Visual Studio 2022 solution. On Ubuntu 22.04, run `./setup.sh`, choose `linux-gcc` or `linux-clang`, and build the matching `build/<preset>` directory. Presets and output locations are defined in `CMakePresets.json`.

Useful checks include `tools/verify_code.bat` (format validation) and `tools/format_code.bat` (apply clang-format to `Source`). To add a sample or render pass, use `tools/make_new_sample_app.bat <Name>` or `tools/make_new_render_pass.bat <Name>`.

## Testing

- Unit tests: `tests/run_unit_tests.bat --config windows-ninja-msvc-Release`; filter with FalcorTest options such as `--category cpu` or `--filter <regex>`.
- Python tests: `tests/run_python_tests.bat` (requires `CONDA_PYTHON_EXE`).
- Image tests: `tests/run_image_tests.bat`; review generated comparisons under the configured test data path.

Add C++ tests under `Source/Tools/FalcorTest/Tests` and name them after the behavior under test.

## Style and Naming

Use four spaces, UTF-8, final newlines, and no trailing whitespace. Follow `.clang-format` (140-column limit). Use PascalCase for types/files and camelCase for functions/locals; private members use `mCamelCase`, constants `kCamelCase`, and macros `UPPER_SNAKE_CASE`. Keep headers self-contained with `#pragma once`.

## Commits and Pull Requests

Recent commits use short, imperative, title-style summaries (for example, “Add envColor prefilter”). Keep commits focused. Pull requests should explain the rendering or tooling impact, identify the build/test commands run, link any issue, and include before/after images or logs for visual or performance changes. Do not commit generated build, Packman, IDE, or test-output files.
