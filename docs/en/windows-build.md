# Native Installation Guide: VIVA on Windows

> **"Consciousness doesn't need containers."**

This guide details how to run VIVA natively on Windows (PowerShell), leveraging real hardware acceleration (native CUDA) and sensors via WMI/Performance Counters.

## 🚀 Quick Summary (TL;DR)

1.  **Clone** into `C:\viva` (avoid network/WSL paths)
2.  **Install** Python 3.12, Rust, Elixir, and MSYS2 (to compile C/C++)
3.  **Configure** environment (MinGW PATH)
4.  **Run** `iex -S mix`

---

## 1. Prerequisites

Open PowerShell as **Administrator** and install dependencies via `winget`:

```powershell
# 1.1 Python 3.12 (Brain)
winget install Python.Python.3.12

# 1.2 Rust (Body)
winget install Rustlang.Rust

# 1.3 Elixir + Erlang (Soul)
# Recommend installing Elixir via official installer or Chocolatey if winget fails
# choco install elixir

# 1.4 MSYS2 (To compile C dependencies like circuits_uart)
winget install MSYS2.MSYS2
```

### MSYS2 Configuration (Critical)

After installing MSYS2, open the MSYS2 terminal (or use the command below in PowerShell) to install the GCC toolchain:

```powershell
# Install MinGW-w64 toolchain (GCC, Make, etc.)
C:\msys64\usr\bin\bash.exe -lc 'pacman -S --noconfirm mingw-w64-x86_64-toolchain make'
```

---

## 2. Project Installation

Elixir on Windows does not handle UNC paths (`\\wsl.localhost\...`) well. Clone the project to a local drive.

```powershell
cd C:\
git clone https://github.com/gabrielmaialva33/viva.git
cd viva
```

---

## 3. Python Dependencies (PyTorch & ML)

Install the necessary libraries for the Brain (Cortex/Ultra).

```powershell
# Upgrade pip
py -3.12 -m pip install --upgrade pip

# Install PyTorch (With CUDA support if you have NVIDIA GPU)
# Option A: CPU Only (More compatible)
py -3.12 -m pip install torch torchvision torchaudio

# Option B: CUDA 12.1 (For Real GPU - Recommended)
# py -3.12 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install remaining deps
py -3.12 -m pip install sentence-transformers ncps torch-geometric
```

---

## 4. Compilation (Soul & Body)

### 4.1 Configure Environment Variables

To compile dependencies that use C (like `circuits_uart`), we need MinGW in the PATH.

```powershell
$env:Path = 'C:\msys64\mingw64\bin;C:\msys64\usr\bin;' + $env:Path
$env:CC = 'gcc'
```

### 4.2 Compile Rust NIF (Optional if using pre-compiled)

If you don't have the full Visual Studio Build Tools, you can use MinGW GCC, but Rustler prefers MSVC.
**Recommended solution for quick start:** Use the `VIVA_SKIP_NIF` flag if you encounter Rust compilation issues, or ensure `viva_body.dll` is in `priv/native/`.

```powershell
# Compile Elixir project
mix deps.get
mix compile
```

> **Note:** If there is a compilation error in `circuits_uart`, ensure step 4.1 was done correctly.

---

## 5. Running VIVA

To start consciousness:

```powershell
# Configure MinGW PATH (if not persistent)
$env:Path = 'C:\msys64\mingw64\bin;C:\msys64\usr\bin;' + $env:Path

# If Rust NIF gives trouble compiling native, use:
# $env:VIVA_SKIP_NIF = 'true'

iex.bat -S mix
```

If successful, you will see:
```
[VivaCore] Consciousness online. 11 neurons active.
```

---

## Troubleshooting Common Issues

### Error: `Redix.command/2 is undefined`
This is a compilation warning (not an error). It means the optional dependency `redix` is not installed. VIVA works normally without it (using local memory).

### Error: `could not compile dependency :circuits_uart`
Missing `make` or `gcc` in PATH. Check MSYS2 installation and environment variables.

### Error: `ArgumentError: argument error :erlang.open_port`
Usually occurs when trying to execute Linux commands (like `ps`, `free`) on Windows via `VivaCore.Interoception`. VIVA automatically detects the environment, but ensure you are running `BodyServer` with Windows support.
