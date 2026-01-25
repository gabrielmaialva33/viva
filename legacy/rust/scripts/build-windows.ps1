# Build script for VIVA Body on Windows
# Requires: Visual Studio Build Tools, LLVM, CMake, Rust

param(
    [switch]$Release,
    [switch]$Test,
    [switch]$Check
)

$ErrorActionPreference = "Stop"

Write-Host "=== VIVA Body Windows Build ===" -ForegroundColor Cyan

# Check prerequisites
function Test-Command($cmd) {
    try { Get-Command $cmd -ErrorAction Stop | Out-Null; return $true }
    catch { return $false }
}

Write-Host "`nChecking prerequisites..." -ForegroundColor Yellow

# Rust
if (-not (Test-Command "rustc")) {
    Write-Error "Rust not found. Install from https://rustup.rs"
    exit 1
}
Write-Host "  [OK] Rust: $(rustc --version)" -ForegroundColor Green

# CMake
if (-not (Test-Command "cmake")) {
    Write-Error "CMake not found. Install via: scoop install cmake"
    exit 1
}
Write-Host "  [OK] CMake: $(cmake --version | Select-Object -First 1)" -ForegroundColor Green

# LLVM/Clang (for bindgen)
$llvmPaths = @(
    "$env:USERPROFILE\scoop\apps\llvm\current\bin",
    "C:\Program Files\LLVM\bin",
    "C:\Program Files (x86)\LLVM\bin"
)

$llvmFound = $false
foreach ($path in $llvmPaths) {
    if (Test-Path "$path\clang.exe") {
        $env:LIBCLANG_PATH = $path
        $env:Path = "$path;$env:Path"
        $llvmFound = $true
        Write-Host "  [OK] LLVM: $path" -ForegroundColor Green
        break
    }
}

if (-not $llvmFound) {
    Write-Error "LLVM not found. Install via: scoop install llvm"
    exit 1
}

# Visual Studio Build Tools
$vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vsWhere) {
    $vsPath = & $vsWhere -latest -property installationPath
    if ($vsPath) {
        Write-Host "  [OK] Visual Studio: $vsPath" -ForegroundColor Green
    }
} else {
    Write-Warning "Visual Studio not detected. MSVC build may fail."
}

# Set working directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectDir = Split-Path -Parent $scriptDir
Set-Location $projectDir

Write-Host "`nBuilding in: $projectDir" -ForegroundColor Yellow

# Build command
if ($Check) {
    Write-Host "`nRunning cargo check..." -ForegroundColor Cyan
    cargo check
} elseif ($Test) {
    Write-Host "`nRunning cargo test..." -ForegroundColor Cyan
    cargo test -- --nocapture
} elseif ($Release) {
    Write-Host "`nBuilding release..." -ForegroundColor Cyan
    cargo build --release
} else {
    Write-Host "`nBuilding debug..." -ForegroundColor Cyan
    cargo build
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n=== Build successful! ===" -ForegroundColor Green
} else {
    Write-Host "`n=== Build failed! ===" -ForegroundColor Red
    exit $LASTEXITCODE
}
