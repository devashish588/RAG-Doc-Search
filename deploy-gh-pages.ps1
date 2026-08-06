# Deploys the static frontend to GitHub Pages via a `gh-pages` branch.
# Run from PowerShell in the repo root:
#   .\deploy-gh-pages.ps1
#
# How it works:
#   * A DETACHED worktree is created in the user temp dir. We NEVER delete the
#     worktree's `.git` file (that would make `git -C` walk up and operate on
#     whichever repo sits above the temp dir). Files are cleared with
#     `git rm -rf .` instead.
#   * After committing the frontend, the `gh-pages` branch is re-pointed at the
#     new commit and force-pushed (the branch is a build artifact).
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

# Detached worktree in the temp dir (outside the repo, keeps .git intact).
$work = Join-Path $env:TEMP "opencode-gh-pages-deploy"

git -C $repoRoot worktree prune
Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
git -C $repoRoot worktree add --detach $work HEAD

try {
    # Clear tracked files WITHOUT removing .git, then copy in the frontend.
    git -C $work rm -rf . 2>&1 | Out-Null
    Copy-Item -Path "$frontend\*" -Destination $work -Recurse -Force

    git -C $work add -A
    git -C $work diff --cached --quiet
    if ($LASTEXITCODE -ne 0) {
        git -C $work commit -m "Deploy frontend to GitHub Pages"
        git -C $work branch -f gh-pages HEAD
        git -C $work push --force origin gh-pages
        Write-Output "Deployed. Site will appear at https://devashish588.github.io/RAG-Doc-Search/"
    } else {
        Write-Output "No changes to deploy. GitHub Pages is up to date."
    }
} finally {
    git -C $repoRoot worktree remove --force $work 2>&1 | Out-Null
    Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
}