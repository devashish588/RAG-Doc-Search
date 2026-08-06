# Deploys the static frontend to GitHub Pages via a `gh-pages` branch.
# Run from PowerShell in the repo root:
#   .\deploy-gh-pages.ps1
#
# The gh-pages worktree is created OUTSIDE the repo (in the user temp dir) so
# that `git -C <worktree>` never resolves back to the main repository.
#
# Before running:
#   1. Edit frontend/config.js and set window.API_BASE to your Render backend URL.
#   2. In GitHub: Repo -> Settings -> Pages -> Deploy from branch `gh-pages` / root.
$ErrorActionPreference = "Continue" # native git stderr prints but never aborts

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$frontend = Join-Path $repoRoot "frontend"
if (-not (Test-Path $frontend)) {
    Write-Error "frontend/ not found at $frontend"
    exit 1
}

$wt = Join-Path $env:TEMP "opencode"

# Clean any stale worktree registration then prep the branch.
git -C $repoRoot worktree prune ""
git -C $repoRoot fetch origin gh-pages 2>&1 | Out-Null
git -C $repoRoot rev-parse --verify origin/gh-pages 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) {
    git -C $repoRoot branch -f gh-pages origin/gh-pages 2>&1 | Out-Null
} else {
    git -C $repoRoot rev-parse --verify gh-pages 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        git -C $repoRoot branch gh-pages HEAD 2>&1 | Out-Null
    }
}

$work = Join-Path $wt "gh-pages"
Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
git -C $repoRoot worktree add -f $work gh-pages

try {
    # Replace the branch contents with the frontend folder.
    Get-ChildItem -LiteralPath $work -Force | Remove-Item -Recurse -Force
    Copy-Item -Path "$frontend\*" -Destination $work -Recurse -Force

    git -C $work add -A
    git -C $work diff --cached --quiet
    if ($LASTEXITCODE -ne 0) {
        git -C $work commit -m "Deploy frontend to GitHub Pages"
        git -C $work push origin gh-pages
        Write-Output "Deployed. Site will appear at https://devashish588.github.io/RAG-Doc-Search/"
    } else {
        Write-Output "No changes to deploy. GitHub Pages is up to date."
    }
} finally {
    git -C $repoRoot worktree remove --force $work 2>&1 | Out-Null
    Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
}