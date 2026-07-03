# Запуск Nornickel Ore Analyzer (Windows)
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

Write-Host "Starting API on http://127.0.0.1:8000 ..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$Root'; py -m uvicorn app.main:app --host 127.0.0.1 --port 8000"

Start-Sleep -Seconds 2

Write-Host "Starting UI on http://127.0.0.1:5173 ..."
Set-Location "$Root\web"
if (-not (Test-Path node_modules)) { npm install }
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$Root\web'; npm run dev"

Write-Host ""
Write-Host "Done! Open in browser: http://127.0.0.1:5173"
Write-Host "API docs: http://127.0.0.1:8000/docs"
