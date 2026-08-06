# Deploys the static frontend to GitHub Pages via a `gh-pages` branch.
# Run from PowerShell in the repo root:
#   .\deploy-gh-pages.ps1
#
# Before running:
#   1. Edit frontend/config.js and set window.API_BASE to your Render backend URL.
#   2. In GitHub: Repo -> Settings -> Pages -> Deploy from branch `gh-pages` / root.
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$frontend = Join-Path $repoRoot "frontend"
$worktree = Join-Path $repoRoot "gh-pages"

if (-not (Test-Path $frontend)) {
    Write-Error "frontend/ not found at $frontend"
}

git -C $repoRoot fetch origin gh-pages 2>$null

# Reuse an existing gh-pages branch if present, else create one.
if (git -C $repoRoot rev-parse --verify origin/gh-pages 2>$null) {
    git -C $repoRoot branch -f gh-pages origin/gh-pages
} else {
    git -C $repoRoot branch gh-pages 2>$null
}

git -C $repoRoot worktree add -f $worktree gh-pages

try {
    # Replace the branch contents with the frontend folder.
    Get-ChildItem -LiteralPath $worktree -Force | Remove-Item -Recurse -Force
    Copy-Item -Path "$frontend\*" -Destination $worktree -Recurse -Force

    git -C $worktree add -A
    if (git -C $worktree diff --cached --quiet) {
        Write-Output "No changes to deploy."
    } else {
        git -C $worktree commit -m "Deploy frontend to GitHub Pages"
        git -C $worktree push origin gh-pages
        Write-Output "Deployed. Site will appear at https://<user>.github.io/RAG-Doc-Search/"
    }
} finally {
    git -C $repoRoot worktree remove --force $worktree
}
