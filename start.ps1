# RagMate Unified PowerShell Launcher
Set-Location -Path $PSScriptRoot

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Starting RagMate Unified Launcher..." -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

if (Test-Path "backend\.venv\Scripts\python.exe") {
    & "backend\.venv\Scripts\python.exe" run.py @args
} else {
    python run.py @args
}
