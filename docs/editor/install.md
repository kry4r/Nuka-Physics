# Install the editor

The editor is a desktop application on top of the engine. It runs on **Windows 10 or 11 (x64)** today;
Linux builds of the editor are in progress, and the engine itself builds on Linux as
[Build the engine](../getting-started.md) describes.

There are two ways to run it:

- **The packaged app**: a `Nuka Editor` folder holding `nuka-editor.exe`, a private Python and the `nuka`
  package. Unzip it anywhere and run the exe; nothing else needs installing besides the GPU driver.
- **From source**: build the engine, the editor library and the Python extensions, then start the editor
  with `python -m nuka.editor`. Scripts and training runs use the same build.

## Requirements

| | Needed |
| --- | --- |
| GPU | An NVIDIA GPU with a current driver (CUDA 13 runtime, Vulkan 1.3) |
| Display | 1920 × 1080 or larger is comfortable; the UI scales with the display and with ++ctrl+equal++ / ++ctrl+minus++ |

To build from source you also need:

- Visual Studio 2022 with the C++ workload (MSVC)
- CUDA Toolkit 13
- Vulkan SDK 1.4
- CMake 3.20 or newer and Ninja
- Python 3.10 or newer, with `numpy`; `torch` and `rl_games==1.6.5` to train policies

The first configure also fetches SDL3, the UI fonts and icons, and OpenUSD, which it builds once.
Expect the first build to take a while.

## Build from source

Run these from the repository root in an *x64 Native Tools Command Prompt for VS 2022*.

1. Configure and build the engine and the editor:

    ```bat
    cmake -S . -B build-win-editor -G Ninja -DCMAKE_BUILD_TYPE=Release ^
          -DNK_BUILD_EDITOR=ON -DNK_BUILD_TESTS=OFF
    cmake --build build-win-editor --target nuka nuka_editor nuka_editor_exe
    ```

    Pass `-DNK_CUDA_ARCHITECTURES=<sm>` (for example `120` for an RTX 50 series card) to build for your
    GPU only, which is much faster.

2. Build the Python extensions against the same CUDA toolkit:

    ```bat
    cmake -S python -B build-win-py -G Ninja -DCMAKE_BUILD_TYPE=Release ^
          -DPython_EXECUTABLE=C:\path\to\python.exe
    cmake --build build-win-py
    ```

3. Put the pieces next to the `nuka` package. The editor loads them from the package folder:

    | From | Into `python/nuka/` |
    | --- | --- |
    | `build-win-py/` | `_nuka_ext*.pyd`, `_editor_ext*.pyd` |
    | `build-win-editor/src/` | `nuka.dll`, `nuka_editor.dll`, `nukausd_*.dll`, `tbb.dll`, and the folders `usd/`, `assets/`, `extensions/` |
    | the CUDA toolkit's `bin/x64/` | `cudart64_13.dll` |

4. Make the package importable, for example with `set PYTHONPATH=%CD%\python`, and check it:

    ```bat
    python -c "import nuka, nuka.editor; print(nuka.__engine_version__)"
    ```

## Start the editor

```bat
python -m nuka.editor                      :: the start page
python -m nuka.editor scene.nks            :: open a scene
python -m nuka.editor robot.urdf           :: import an asset into a new document
nuka-editor scene.nks --layout robot       :: the same, through the console script
```

With the packaged app, run `nuka-editor.exe` with the same arguments, or drop a scene on it.

A scene given while an editor already runs with the same profile opens in that window; `--new-window`
starts another editor. Every option is listed in [Command line](../reference/command-line.md).

## The profile folder

Each user's settings live in one folder:

- Windows: `%APPDATA%\Nuka\editor`
- Linux: `$XDG_CONFIG_HOME/nuka/editor`, else `~/.config/nuka/editor`

It holds `settings.toml` (the UI, asset folders, script and training settings, agent access),
`recent.json` (the last 12 scenes), `logs/` (the editor log and the agents' audit files), `autosave/`,
`extensions/` (your extensions), `thumbnails/` and `recordings/`. `--profile DIR` uses another folder,
which is how you keep separate setups side by side.

## Troubleshooting

- **The window does not open, or closes at once.** Run with `--console` (the packaged app) or from a
  terminal and read the output; `--log FILE` writes the log to a file. A missing Vulkan driver or CUDA
  runtime is named there.
- **`ImportError` on `nuka.editor`.** The editor library was not built (`NK_BUILD_EDITOR=ON`), or its DLLs
  are not beside the package. `python -m nuka.editor --headless scene.nks` still runs the world without
  the editor.
- **Training says Python is not ready.** The packaged app's own Python carries no `torch`. Point the
  Training panel at a Python that has `torch` and `rl_games` ([Training](training.md#the-python-it-trains-with)).
