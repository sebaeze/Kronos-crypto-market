param (
    [string]$Symbol = "JELLYJELLYUSDT",
    [int]$Epochs = 3,
    [int]$BatchSize = 4,
    [int]$AccumSteps = 4,
    [string]$DataDir = "finetune_ohlcv_json"
)

Write-Host "Starting Kronos Local Fine-Tuning Pipeline" -ForegroundColor Cyan
Write-Host "Symbol: $Symbol" -ForegroundColor Yellow
Write-Host "Epochs: $Epochs" -ForegroundColor Yellow
Write-Host "Batch Size: $BatchSize" -ForegroundColor Yellow
Write-Host "Accumulation Steps: $AccumSteps" -ForegroundColor Yellow
Write-Host "Data Directory: $DataDir" -ForegroundColor Yellow

$VenvPython = ".\.venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Error "Virtual environment python not found at $VenvPython. Please run setup_env.ps1 first."
    exit 1
}

Write-Host "`nExecuting training script..." -ForegroundColor Cyan

& $VenvPython train_kronos.py --symbol $Symbol --epochs $Epochs --batch-size $BatchSize --accum-steps $AccumSteps --data-dir $DataDir

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nTraining completed successfully!" -ForegroundColor Green
    Write-Host "You can test the model by running: & $VenvPython local_inference.py --symbol `"$Symbol`"" -ForegroundColor Cyan
} else {
    Write-Error "`nTraining failed with exit code $LASTEXITCODE"
}
