# GitHub Tasks Helper Script
# PowerShell script for managing GitHub Issues and Projects via CLI

param(
    [Parameter(Position=0)]
    [string]$Command = "help",
    
    [Parameter(Position=1)]
    [string]$Title = "",
    
    [Parameter(Position=2)]
    [string]$Body = "",
    
    [string]$Labels = "",
    [string]$Project = "",
    [int]$IssueNumber = 0,
    [string]$Comment = "",
    [string]$Status = ""
)

$Repo = "gavinlouuu-kpt/ITS209"

# Refresh PATH to include newly installed programs (like GitHub CLI)
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

function Show-Help {
    Write-Host @"
GitHub Tasks Helper Script
Usage: .\github-tasks.ps1 <command> [options]

Commands:
  create <title> [body]          Create a new issue
    -Labels <labels>              Comma-separated labels (e.g., "bug,urgent")
    -Project <project-name>       Add to project by name
    
  list                            List all open issues
  view <issue-number>             View issue details
  comment <issue-number> <text>   Add comment to issue
  close <issue-number>            Close an issue
  reopen <issue-number>           Reopen a closed issue
  status <issue-number> <status>  Update issue status (open/closed)
  
  projects                        List all projects
  project-view <project-number>   View project details
  
  help                            Show this help message

Examples:
  .\github-tasks.ps1 create "Fix sensor bug" "Description here"
  .\github-tasks.ps1 create "New feature" "Desc" -Labels "enhancement" -Project "ITS209 Development"
  .\github-tasks.ps1 list
  .\github-tasks.ps1 view 123
  .\github-tasks.ps1 comment 123 "Working on this now"
  .\github-tasks.ps1 close 123
  .\github-tasks.ps1 projects

"@
}

function Test-GitHubCLI {
    if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
        Write-Host "Error: GitHub CLI (gh) is not installed." -ForegroundColor Red
        Write-Host "Install it with: winget install --id GitHub.cli" -ForegroundColor Yellow
        Write-Host "Or visit: https://cli.github.com/" -ForegroundColor Yellow
        exit 1
    }
    
    $authStatus = gh auth status 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: GitHub CLI is not authenticated." -ForegroundColor Red
        Write-Host "Run: gh auth login" -ForegroundColor Yellow
        exit 1
    }
}

function New-GitHubIssue {
    param(
        [string]$Title,
        [string]$Body,
        [string]$Labels = "",
        [string]$Project = ""
    )
    
    if ([string]::IsNullOrWhiteSpace($Title)) {
        Write-Host "Error: Title is required for creating an issue." -ForegroundColor Red
        return
    }
    
    # Create issue first without labels (to avoid failure if labels don't exist)
    $args = @("issue", "create", "--repo", $Repo, "--title", $Title)
    
    if (-not [string]::IsNullOrWhiteSpace($Body)) {
        $args += "--body"
        $args += $Body
    }
    
    Write-Host "Creating issue: $Title" -ForegroundColor Cyan
    $issueOutput = gh $args 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Issue created successfully!" -ForegroundColor Green
        
        # Extract issue number from output
        $issueNumber = $issueOutput | Select-String -Pattern 'issues/(\d+)' | ForEach-Object { $_.Matches.Groups[1].Value }
        
        # Try to add labels if provided (ignore errors if labels don't exist)
        if ($issueNumber -and -not [string]::IsNullOrWhiteSpace($Labels)) {
            $labelArray = $Labels -split ','
            foreach ($label in $labelArray) {
                $label = $label.Trim()
                if ($label) {
                    gh issue edit --repo $Repo $issueNumber --add-label $label 2>&1 | Out-Null
                    if ($LASTEXITCODE -ne 0) {
                        Write-Host "  Note: Label '$label' not found in repository" -ForegroundColor Yellow
                    }
                }
            }
        }
        
        # Try to add to project if provided
        if ($issueNumber -and -not [string]::IsNullOrWhiteSpace($Project)) {
            gh issue edit --repo $Repo $issueNumber --add-project $Project 2>&1 | Out-Null
        }
        
        Write-Host "Issue URL: $issueOutput" -ForegroundColor Cyan
    } else {
        Write-Host "Failed to create issue:" -ForegroundColor Red
        Write-Host $issueOutput -ForegroundColor Red
    }
}

function Get-GitHubIssues {
    Write-Host "Fetching issues..." -ForegroundColor Cyan
    gh issue list --repo $Repo
}

function Show-GitHubIssue {
    param([int]$IssueNumber)
    
    if ($IssueNumber -eq 0) {
        Write-Host "Error: Issue number is required." -ForegroundColor Red
        return
    }
    
    gh issue view $IssueNumber --repo $Repo
}

function Add-GitHubIssueComment {
    param(
        [int]$IssueNumber,
        [string]$Comment
    )
    
    if ($IssueNumber -eq 0) {
        Write-Host "Error: Issue number is required." -ForegroundColor Red
        return
    }
    
    if ([string]::IsNullOrWhiteSpace($Comment)) {
        Write-Host "Error: Comment text is required." -ForegroundColor Red
        return
    }
    
    gh issue comment $IssueNumber --repo $Repo --body $Comment
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Comment added successfully!" -ForegroundColor Green
    }
}

function Close-GitHubIssue {
    param([int]$IssueNumber)
    
    if ($IssueNumber -eq 0) {
        Write-Host "Error: Issue number is required." -ForegroundColor Red
        return
    }
    
    gh issue close $IssueNumber --repo $Repo
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Issue closed successfully!" -ForegroundColor Green
    }
}

function Reopen-GitHubIssue {
    param([int]$IssueNumber)
    
    if ($IssueNumber -eq 0) {
        Write-Host "Error: Issue number is required." -ForegroundColor Red
        return
    }
    
    gh issue reopen $IssueNumber --repo $Repo
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Issue reopened successfully!" -ForegroundColor Green
    }
}

function Get-GitHubProjects {
    Write-Host "Fetching projects..." -ForegroundColor Cyan
    gh project list --owner gavinlouuu-kpt
}

function Show-GitHubProject {
    param([int]$ProjectNumber)
    
    if ($ProjectNumber -eq 0) {
        Write-Host "Error: Project number is required." -ForegroundColor Red
        return
    }
    
    gh project view $ProjectNumber --owner gavinlouuu-kpt
}

# Main script logic
Test-GitHubCLI

switch ($Command.ToLower()) {
    "create" {
        New-GitHubIssue -Title $Title -Body $Body -Labels $Labels -Project $Project
    }
    "list" {
        Get-GitHubIssues
    }
    "view" {
        if ($IssueNumber -eq 0 -and $Title -ne "") {
            $IssueNumber = [int]$Title
        }
        Show-GitHubIssue -IssueNumber $IssueNumber
    }
    "comment" {
        if ($IssueNumber -eq 0 -and $Title -ne "") {
            $IssueNumber = [int]$Title
        }
        if ([string]::IsNullOrWhiteSpace($Comment) -and -not [string]::IsNullOrWhiteSpace($Body)) {
            $Comment = $Body
        }
        Add-GitHubIssueComment -IssueNumber $IssueNumber -Comment $Comment
    }
    "close" {
        if ($IssueNumber -eq 0 -and $Title -ne "") {
            $IssueNumber = [int]$Title
        }
        Close-GitHubIssue -IssueNumber $IssueNumber
    }
    "reopen" {
        if ($IssueNumber -eq 0 -and $Title -ne "") {
            $IssueNumber = [int]$Title
        }
        Reopen-GitHubIssue -IssueNumber $IssueNumber
    }
    "projects" {
        Get-GitHubProjects
    }
    "project-view" {
        if ($IssueNumber -eq 0 -and $Title -ne "") {
            $IssueNumber = [int]$Title
        }
        Show-GitHubProject -ProjectNumber $IssueNumber
    }
    "help" {
        Show-Help
    }
    default {
        Write-Host "Unknown command: $Command" -ForegroundColor Red
        Write-Host ""
        Show-Help
    }
}


