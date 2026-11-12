# Installs Ojon for the current user: copies this folder to
# %LOCALAPPDATA%\Programs\Ojon and adds a Start Menu shortcut. No admin rights needed.
#
#   -AddToPath   also put the `ojon` command on your PATH
param([switch]$AddToPath)
$ErrorActionPreference = "Stop"
$source = $PSScriptRoot
$target = Join-Path $env:LOCALAPPDATA "Programs\Ojon"

if (-not (Test-Path (Join-Path $source "ojon-gui.exe"))) {
    throw "ojon-gui.exe is not in this folder. Run this script from the unzipped Ojon folder."
}
if (Get-Process ojon-gui, ojon -ErrorAction SilentlyContinue) {
    throw "Ojon is running. Close it, then run this again."
}

New-Item -ItemType Directory -Force $target | Out-Null
Copy-Item (Join-Path $source "*") $target -Recurse -Force

$menu = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"
$shell = New-Object -ComObject WScript.Shell
$link = $shell.CreateShortcut((Join-Path $menu "Ojon.lnk"))
$link.TargetPath = Join-Path $target "ojon-gui.exe"
$link.WorkingDirectory = $target
$link.Description = "Ojon: make models smaller"
$link.Save()

if ($AddToPath) {
    $path = [Environment]::GetEnvironmentVariable("Path", "User")
    if (($path -split ";") -notcontains $target) {
        [Environment]::SetEnvironmentVariable("Path", ($path.TrimEnd(";") + ";" + $target), "User")
        "Added to your PATH. Open a new terminal to use the ojon command."
    }
}

"Ojon is installed in $target"
"Start it from the Start Menu, or run: `"$target\ojon-gui.exe`""
