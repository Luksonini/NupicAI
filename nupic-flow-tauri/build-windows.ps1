param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^https://')]
    [string]$ServerUrl
)

$ErrorActionPreference = 'Stop'
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDir

if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    throw 'Rust is missing. Install it from https://rustup.rs/.'
}
try {
    cargo tauri --version | Out-Null
} catch {
    throw "cargo-tauri is missing. Run: cargo install tauri-cli --version '^2' --locked"
}

$env:NUPICAI_SERVER_URL = $ServerUrl.TrimEnd('/')
cargo tauri build --bundles nsis
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$Installer = Get-ChildItem "$ProjectDir\target\release\bundle\nsis\*.exe" | Select-Object -First 1
if (-not $Installer) { throw 'Tauri finished, but the NSIS installer is missing.' }

$PublishDir = Join-Path (Split-Path -Parent $ProjectDir) 'runtime\downloads'
New-Item -ItemType Directory -Force -Path $PublishDir | Out-Null
$Destination = Join-Path $PublishDir 'nupicai-flow-windows-x86_64.exe'
Copy-Item -Force $Installer.FullName $Destination
Write-Host "Windows installer: $Destination"
