param (
    [string]$Symbol = "JELLYJELLYUSDT"
)

Write-Host "Setting up environment for Kronos fine-tuning ($Symbol)..." -ForegroundColor Cyan

# Create and checkout a new branch
$BranchName = "feature/local-finetuning_$Symbol"
Write-Host "Creating branch: $BranchName" -ForegroundColor Yellow
git checkout -b $BranchName
if ($LASTEXITCODE -ne 0) {
    Write-Host "Branch might already exist or git error. Checking out existing branch..." -ForegroundColor Yellow
    git checkout $BranchName
}

# Setup Python virtual environment
Write-Host "Creating virtual environment .venv..." -ForegroundColor Yellow
python -m venv .venv
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to create virtual environment."
    exit 1
}

# We need to activate it within the current process just for this script, 
# or just use the python executable directly from the venv.
$VenvPython = ".\.venv\Scripts\python.exe"
$VenvPip = ".\.venv\Scripts\pip.exe"

Write-Host "Upgrading pip..." -ForegroundColor Yellow
& $VenvPython -m pip install --upgrade pip

Write-Host "Installing CPU-only PyTorch..." -ForegroundColor Yellow
& $VenvPip install torch --index-url https://download.pytorch.org/whl/cpu

Write-Host "Installing dependencies..." -ForegroundColor Yellow
& $VenvPip install transformers pandas einops huggingface_hub matplotlib tqdm safetensors

Write-Host "Environment setup complete!" -ForegroundColor Green
Write-Host "Please activate the virtual environment manually by running: .\.venv\Scripts\Activate.ps1" -ForegroundColor Cyan
