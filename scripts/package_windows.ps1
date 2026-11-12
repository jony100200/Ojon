# Builds Ojon and makes a Windows zip: both programs, the bundled data, the guides,
# and a per-user installer. Output: dist\Ojon-<version>-windows-x64.zip
#
#   pwsh scripts\package_windows.ps1            # build, then package
#   pwsh scripts\package_windows.ps1 -SkipBuild # package what is already built
param(
    [int]$Jobs = 2,
    [switch]$SkipBuild
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

$version = (Select-String -Path Cargo.toml -Pattern '^version\s*=\s*"(.+)"' | Select-Object -First 1).Matches[0].Groups[1].Value
if (-not $SkipBuild) {
    cargo build --release --locked --features gui --bin ojon --bin ojon-gui -j $Jobs
    if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }
}

$name = "Ojon-$version-windows-x64"
$stage = Join-Path $root "dist\$name"
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Force $stage | Out-Null

foreach ($exe in "ojon.exe", "ojon-gui.exe") {
    $built = Join-Path $root "target\release\$exe"
    if (-not (Test-Path $built)) { throw "$exe is not built: run without -SkipBuild" }
    Copy-Item $built $stage
}
Copy-Item (Join-Path $root "data") $stage -Recurse
Copy-Item (Join-Path $root "LICENSE"), (Join-Path $root "README.md"), (Join-Path $root "USER_GUIDE.md") $stage
Copy-Item (Join-Path $root "packaging\windows\*") $stage

$zip = Join-Path $root "dist\$name.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path $stage -DestinationPath $zip
$hash = (Get-FileHash $zip -Algorithm SHA256).Hash
Set-Content -Path "$zip.sha256" -Value "$hash  $name.zip" -Encoding ascii
"{0}  ({1:N1} MB)" -f $zip, ((Get-Item $zip).Length / 1MB)
"SHA256 $hash"
