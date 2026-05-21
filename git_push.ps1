param (
    [string]$Symbol = "JELLYJELLYUSDT"
)

$BranchName = "feature/local-finetuning_$Symbol"

Write-Host "Updating Git repository for branch: $BranchName" -ForegroundColor Cyan

# Stage files
Write-Host "Staging files..." -ForegroundColor Yellow
git add setup_env.ps1 git_push.ps1 run_training.ps1 data_loader.py train_kronos.py local_inference.py .gitignore docs/0_running_training.md

# Commit
Write-Host "Committing changes..." -ForegroundColor Yellow
git commit -m "Add local fine-tuning pipeline for Kronos ($Symbol)"

# Push
Write-Host "Pushing branch to origin..." -ForegroundColor Yellow
git push origin $BranchName

Write-Host "Git update complete!" -ForegroundColor Green
