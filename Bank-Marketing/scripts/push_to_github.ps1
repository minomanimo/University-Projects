param(
    [string]$RemoteUrl = "https://github.com/minomanimo/University-Projects.git",
    [string]$Branch = "main",
    [string]$CommitMessage = "Initial project commit"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path ".git")) {
    git init
}

git add .

$hasStagedChanges = & git diff --cached --name-only
if (-not $hasStagedChanges) {
    Write-Host "No staged changes to commit."
    exit 0
}

git commit -m $CommitMessage

$remoteExists = $false
try {
    $null = git remote get-url origin
    $remoteExists = $true
} catch {
    $remoteExists = $false
}

if (-not $remoteExists) {
    git remote add origin $RemoteUrl
} else {
    git remote set-url origin $RemoteUrl
}

git branch -M $Branch
git push -u origin $Branch
