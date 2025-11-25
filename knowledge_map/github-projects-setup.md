# GitHub Projects Setup Guide

## Overview

This guide will help you set up GitHub Projects for task management in the ITS209 repository. GitHub Projects provides a Kanban-style board for organizing and tracking work.

## Prerequisites

- A GitHub account
- Access to the repository: `https://github.com/gavinlouuu-kpt/ITS209`
- GitHub CLI (`gh`) installed (optional, for command-line operations)

## Step 1: Create a GitHub Project

1. Navigate to your repository: `https://github.com/gavinlouuu-kpt/ITS209`
2. Click on the **Projects** tab (or go to `https://github.com/gavinlouuu-kpt/ITS209/projects`)
3. Click **New project**
4. Choose **Board** layout (recommended for task tracking)
5. Name your project (e.g., "ITS209 Development" or "ITS209 Tasks")
6. Click **Create project**

## Step 2: Configure Project Columns

GitHub Projects starts with default columns. Recommended structure:

- **Backlog** - Tasks to be done
- **Todo** - Ready to start
- **In Progress** - Currently being worked on
- **In Review** - Code review or testing
- **Done** - Completed tasks

To customize columns:
1. Click the **+** button at the top of a column to add a new column
2. Click the column header to rename or delete columns
3. Drag columns to reorder them

## Step 3: Link Issues to Projects

### Method 1: From GitHub UI

1. Create an Issue:
   - Go to the **Issues** tab
   - Click **New issue**
   - Fill in title and description
   - Add labels if needed
   - Click **Submit new issue**

2. Add Issue to Project:
   - Open the Issue
   - In the right sidebar, find **Projects**
   - Select your project
   - Choose the appropriate column (e.g., "Backlog")

### Method 2: From Project Board

1. Open your Project board
2. Click **+ Add item**
3. Select **Create new issue** or search for existing issues
4. The issue will be added to the selected column

## Step 4: Set Up GitHub CLI (Optional)

For command-line task management:

### Install GitHub CLI

**Windows (PowerShell):**
```powershell
winget install --id GitHub.cli
```

**macOS:**
```bash
brew install gh
```

**Linux:**
```bash
# Debian/Ubuntu
sudo apt install gh

# Fedora
sudo dnf install gh
```

### Authenticate GitHub CLI

```bash
gh auth login
```

Follow the prompts to authenticate. Choose:
- GitHub.com
- HTTPS
- Authenticate Git with your GitHub credentials (Yes)
- Login with a web browser (recommended)

### Verify Installation

```bash
gh --version
gh auth status
```

## Step 5: Configure Project Automation (Optional)

### GitHub Actions Workflow

A GitHub Actions workflow is included in `.github/workflows/project-sync.yml` that can:
- Automatically add new issues to the project
- Move issues between columns based on labels
- Add pull requests to the project
- Auto-close issues when PRs are merged

**To enable the workflow:**
1. Create your GitHub Project first (see Step 1)
2. Note your project number from the project URL (e.g., if URL is `https://github.com/users/gavinlouuu-kpt/projects/2`, the number is `2`)
3. Edit `.github/workflows/project-sync.yml`
4. Replace all instances of `/projects/1` with your actual project number (e.g., `/projects/2`)
5. Commit and push the changes
6. The workflow will automatically run when issues or PRs are created

### GitHub Projects Built-in Automation

GitHub Projects also has built-in automation:
1. Open your Project board
2. Click the **...** menu (three dots) in the top right
3. Select **Workflows** or **Settings**
4. Configure automation rules as needed

## Best Practices

1. **Use Labels**: Create labels for task types (e.g., `bug`, `feature`, `enhancement`, `documentation`)
2. **Link PRs**: When creating pull requests, reference issues using `#issue-number` or `Fixes #issue-number`
3. **Update Status**: Move issues to "In Progress" when starting work
4. **Close Issues**: When a PR is merged, the linked issue will automatically close
5. **Milestones**: Use milestones to group related issues for releases

## Quick Links

- Repository: https://github.com/gavinlouuu-kpt/ITS209
- Issues: https://github.com/gavinlouuu-kpt/ITS209/issues
- Projects: https://github.com/gavinlouuu-kpt/ITS209/projects

## Next Steps

After setting up your project board, see `task-management.md` for detailed workflows on creating and managing tasks.

