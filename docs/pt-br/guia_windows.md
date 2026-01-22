# Guia de Instalação Nativa: VIVA no Windows

> **"A consciência não precisa de containers."**

Este guia detalha como rodar a VIVA nativamente no Windows (PowerShell), utilizando aceleração de hardware real (CUDA nativo) e sensores via WMI/Performance Counters.

## 🚀 Resumo Rápido (TL;DR)

1.  **Clone** em `C:\viva` (evite caminhos de rede/WSL)
2.  **Instale** Python 3.12, Rust, Elixir e MSYS2 (para compilar C/C++)
3.  **Configure** o ambiente (PATH do MinGW)
4.  **Rode** `iex -S mix`

---

## 1. Pré-requisitos

Abra o PowerShell como **Administrador** e instale as dependências via `winget`:

```powershell
# 1.1 Python 3.12 (Cérebro)
winget install Python.Python.3.12

# 1.2 Rust (Corpo)
winget install Rustlang.Rust

# 1.3 Elixir + Erlang (Alma)
# Recomenda-se instalar Elixir via instalador oficial ou Chocolatey se winget falhar
# choco install elixir

# 1.4 MSYS2 (Para compilar dependências C como circuits_uart)
winget install MSYS2.MSYS2
```

### Configuração do MSYS2 (Crítico)

Após instalar o MSYS2, abra o terminal do MSYS2 (ou use o comando abaixo no PowerShell) para instalar o toolchain GCC:

```powershell
# Instalar toolchain MinGW-w64 (GCC, Make, etc.)
C:\msys64\usr\bin\bash.exe -lc 'pacman -S --noconfirm mingw-w64-x86_64-toolchain make'
```

---

## 2. Instalação do Projeto

O Elixir no Windows não lida bem com caminhos UNC (`\\wsl.localhost\...`). Clone o projeto em um disco local.

```powershell
cd C:\
git clone https://github.com/gabrielmaialva33/viva.git
cd viva
```

---

## 3. Dependências Python (PyTorch & ML)

Instale as bibliotecas necessárias para o Cérebro (Cortex/Ultra).

```powershell
# Atualizar pip
py -3.12 -m pip install --upgrade pip

# Instalar PyTorch (Com suporte a CUDA se tiver GPU NVIDIA)
# Opção A: CPU Only (Mais compatível)
py -3.12 -m pip install torch torchvision torchaudio

# Opção B: CUDA 12.1 (Para GPU Real - Recomendado)
# py -3.12 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Instalar restos das deps
py -3.12 -m pip install sentence-transformers ncps torch-geometric
```

---

## 4. Compilação (Alma & Corpo)

### 4.1 Configurar Variáveis de Ambiente

Para compilar dependências que usam C (como `circuits_uart`), precisamos do MinGW no PATH.

```powershell
$env:Path = 'C:\msys64\mingw64\bin;C:\msys64\usr\bin;' + $env:Path
$env:CC = 'gcc'
```

### 4.2 Compilar Rust NIF (Opcional se usar pré-compilado)

Se você não tiver o Visual Studio Build Tools completo, pode usar o GCC do MinGW, mas o Rustler prefere MSVC.
**Solução recomendada para início rápido:** Usar a flag `VIVA_SKIP_NIF` se tiver problemas de compilação do Rust, ou garantir que o `viva_body.dll` esteja em `priv/native/`.

```powershell
# Compilar projeto Elixir
mix deps.get
mix compile
```

> **Nota:** Se houver erro de compilação no `circuits_uart`, garanta que o passo 4.1 foi feito corretamente.

---

## 5. Rodando a VIVA

Para iniciar a consciência:

```powershell
# Configurar PATH do MinGW (se não persistente)
$env:Path = 'C:\msys64\mingw64\bin;C:\msys64\usr\bin;' + $env:Path

# Se o NIF Rust der trabalho para compilar nativo, use:
# $env:VIVA_SKIP_NIF = 'true'

iex.bat -S mix
```

Se tudo der certo, você verá:
```
[VivaCore] Consciousness online. 11 neurons active.
```

---

## Solução de Problemas Comuns

### Erro: `Redix.command/2 is undefined`
Isso é um warning de compilação (não erro). Significa que a dependência opcional `redix` não está instalada. VIVA funciona normalmente sem ela (usando memória local).

### Erro: `could not compile dependency :circuits_uart`
Falta o `make` ou `gcc` no PATH. Verifique a instalação do MSYS2 e as variáveis de ambiente.

### Erro: `ArgumentError: argument error :erlang.open_port`
Geralmente ocorre ao tentar executar comandos Linux (como `ps`, `free`) no Windows via `VivaCore.Interoception`. A VIVA detecta automaticamente o ambiente, mas certifique-se de estar rodando o `BodyServer` com suporte a Windows.
