# Removes Ojon for the current user: the program folder, the Start Menu shortcut and
# the PATH entry. Your settings, job history and models are left alone.
$ErrorActionPreference = "Stop"
$target = Join-Path $env:LOCALAPPDATA "Programs\Ojon"

if (Get-Process ojon-gui, ojon -ErrorAction SilentlyContinue) {
    throw "Ojon is running. Close it, then run this again."
}

$shortcut = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Ojon.lnk"
if (Test-Path $shortcut) { Remove-Item $shortcut -Force }

$path = [Environment]::GetEnvironmentVariable("Path", "User")
$kept = ($path -split ";") | Where-Object { $_ -and $_ -ne $target }
if (($kept -join ";") -ne $path.TrimEnd(";")) {
    [Environment]::SetEnvironmentVariable("Path", ($kept -join ";"), "User")
}

if (Test-Path $target) {
    # This script may be running from inside the folder it removes.
    Set-Location $env:TEMP
    Remove-Item $target -Recurse -Force
}
"Ojon was removed. Settings and models were not touched."
