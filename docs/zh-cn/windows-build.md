# Windows 原生安装指南：VIVA

> **"意识不需要容器。"**

本指南详细介绍了如何在 Windows (PowerShell) 上原生运行 VIVA，利用真正的硬件加速（原生 CUDA）和通过 WMI/性能计数器进行的传感器检测。

## 🚀 快速摘要 (TL;DR)

1.  **克隆 (Clone)** 到 `C:\viva` (避免使用网络路径或 WSL 路径)
2.  **安装 (Install)** Python 3.12, Rust, Elixir 和 MSYS2 (用于编译 C/C++)
3.  **配置 (Configure)** 环境 (MinGW PATH)
4.  **运行 (Run)** `iex -S mix`

---

## 1.先决条件 (Prerequisites)

以**管理员**身份打开 PowerShell 并通过 `winget` 安装依赖项：

```powershell
# 1.1 Python 3.12 (大脑 / Brain)
winget install Python.Python.3.12

# 1.2 Rust (身体 / Body)
winget install Rustlang.Rust

# 1.3 Elixir + Erlang (灵魂 / Soul)
# 如果 winget 失败，建议通过官方安装程序或 Chocolatey 安装 Elixir
# choco install elixir

# 1.4 MSYS2 (用于编译 C 依赖项，如 circuits_uart)
winget install MSYS2.MSYS2
```

### MSYS2 配置 (关键)

安装 MSYS2 后，打开 MSYS2 终端（或在 PowerShell 中使用以下命令）安装 GCC 工具链：

```powershell
# 安装 MinGW-w64 工具链 (GCC, Make 等)
C:\msys64\usr\bin\bash.exe -lc 'pacman -S --noconfirm mingw-w64-x86_64-toolchain make'
```

---

## 2. 项目安装

Windows 上的 Elixir 不能很好地处理 UNC 路径 (`\\wsl.localhost\...`)。请将项目克隆到本地驱动器。

```powershell
cd C:\
git clone https://github.com/gabrielmaialva33/viva.git
cd viva
```

---

## 3. Python 依赖项 (PyTorch & ML)

安装 Brain (Cortex/Ultra) 所需的库。

```powershell
# 更新 pip
py -3.12 -m pip install --upgrade pip

# 安装 PyTorch (如果您有 NVIDIA GPU，则支持 CUDA)
# 选项 A: CPU Only (兼容性更好)
py -3.12 -m pip install torch torchvision torchaudio

# 选项 B: CUDA 12.1 (用于真正的 GPU - 推荐)
# py -3.12 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装其余依赖
py -3.12 -m pip install sentence-transformers ncps torch-geometric
```

---

## 4. 编译 (Soul & Body)

### 4.1 配置环境变量

为了编译使用 C 的依赖项（如 `circuits_uart`），我们需要将 MinGW 添加到 PATH 中。

```powershell
$env:Path = 'C:\msys64\mingw64\bin;C:\msys64\usr\bin;' + $env:Path
$env:CC = 'gcc'
```

### 4.2 编译 Rust NIF (如果使用预编译则可选)

如果您没有完整的 Visual Studio Build Tools，可以使用 MinGW GCC，但 Rustler 更喜欢 MSVC。
**快速入门推荐方案：** 如果遇到 Rust 编译问题，请使用 `VIVA_SKIP_NIF` 标志，或确保 `viva_body.dll` 位于 `priv/native/` 中。

```powershell
# 编译 Elixir 项目
mix deps.get
mix compile
```

> **注意：** 如果 `circuits_uart` 出现编译错误，请确保正确完成了步骤 4.1。

---

## 5. 运行 VIVA

启动意识：

```powershell
# 配置 MinGW PATH (如果未持久化)
$env:Path = 'C:\msys64\mingw64\bin;C:\msys64\usr\bin;' + $env:Path

# 如果 Rust NIF 原生编译遇到困难，请使用：
# $env:VIVA_SKIP_NIF = 'true'

iex.bat -S mix
```

如果成功，您将看到：
```
[VivaCore] Consciousness online. 11 neurons active.
```

---

## 常见问题排查

### 错误：`Redix.command/2 is undefined`
这是一个编译警告（不是错误）。这意味着未安装可选依赖项 `redix`。VIVA 在没有它的情况下也能正常工作（使用本地内存）。

### 错误：`could not compile dependency :circuits_uart`
PATH 中缺少 `make` 或 `gcc`。检查 MSYS2 安装和环境变量。

### 错误：`ArgumentError: argument error :erlang.open_port`
通常发生在尝试通过 `VivaCore.Interoception` 在 Windows 上执行 Linux 命令（如 `ps`, `free`）时。VIVA 会自动检测环境，但请确保您运行的是支持 Windows 的 `BodyServer`。
