# Task Management Workflow

This guide covers how to create and manage tasks using GitHub Projects for the ITS209 project.

## Creating Tasks

### Method 1: GitHub Web UI

1. **Navigate to Issues**
   - Go to: https://github.com/gavinlouuu-kpt/ITS209/issues
   - Click **New issue**

2. **Fill in Task Details**
   - **Title**: Clear, descriptive title (e.g., "Implement temperature sensor calibration")
   - **Description**: Detailed information including:
     - What needs to be done
     - Why it's needed
     - Acceptance criteria
     - Any relevant code references
   - **Labels**: Add appropriate labels (e.g., `enhancement`, `bug`, `documentation`)
   - **Assignees**: Assign yourself or team members
   - **Projects**: Select your project board
   - **Milestone**: Link to a milestone if applicable

3. **Submit**
   - Click **Submit new issue**
   - The issue will appear in your selected project column

### Method 2: GitHub CLI

Use the helper script `scripts/github-tasks.ps1` for quick task creation:

```powershell
# Create a new task
.\scripts\github-tasks.ps1 create "Task title" "Task description"

# Create with labels
.\scripts\github-tasks.ps1 create "Fix sensor reading" "Description" -labels "bug,urgent"

# Create and add to project
.\scripts\github-tasks.ps1 create "New feature" "Description" -project "ITS209 Development"
```

Or use GitHub CLI directly:

```bash
# Create issue
gh issue create --title "Task title" --body "Task description" --label "enhancement"

# Create and add to project
gh issue create --title "Task title" --body "Task description" --project "ITS209 Development"
```

## Managing Tasks

### Updating Task Status

**Via Project Board:**
1. Open your Project board
2. Drag the issue card to the appropriate column:
   - **Backlog** → **Todo**: Task is ready to start
   - **Todo** → **In Progress**: Starting work
   - **In Progress** → **In Review**: Code ready for review
   - **In Review** → **Done**: Completed

**Via GitHub CLI:**
```bash
# List all issues
gh issue list

# View specific issue
gh issue view <issue-number>

# Add comment to issue
gh issue comment <issue-number> --body "Update: Working on this now"

# Close issue
gh issue close <issue-number>
```

### Linking Tasks to Code

**In Commits:**
```bash
git commit -m "Implement temperature calibration

Fixes #123"
```

**In Pull Requests:**
- Reference issues in PR description: `Closes #123` or `Fixes #123`
- GitHub will automatically link the PR to the issue
- When PR is merged, the issue will auto-close

**In Code Comments:**
```cpp
// TODO: Add error handling for sensor faults (#123)
```

## Task Organization

### Using Labels

Recommended labels:
- `bug` - Something isn't working
- `enhancement` - New feature or request
- `documentation` - Documentation improvements
- `question` - Further information is requested
- `urgent` - High priority
- `good first issue` - Good for newcomers

Create labels:
1. Go to Issues → Labels
2. Click **New label**
3. Set name, description, and color

### Using Milestones

Milestones group related issues for releases or sprints:

1. Go to Issues → Milestones
2. Click **New milestone**
3. Set title, description, and due date
4. Link issues to milestones when creating or editing them

### Using Projects

- **One Project Board**: Recommended for small projects
- **Multiple Boards**: Use separate boards for different areas (e.g., "Hardware", "Software", "Testing")

## Common Workflows

### Starting Work on a Task

1. Move issue from **Backlog** or **Todo** to **In Progress**
2. Create a new branch: `git checkout -b feature/issue-123-task-name`
3. Work on the task
4. Commit with issue reference: `git commit -m "Work on #123"`
5. Create PR when ready

### Completing a Task

1. Move issue to **In Review** when PR is created
2. After PR review and merge, issue moves to **Done** (if auto-close enabled)
3. Or manually move to **Done** column

### Task Templates

Create issue templates for consistency:

1. Create `.github/ISSUE_TEMPLATE/` directory
2. Add template files (e.g., `bug_report.md`, `feature_request.md`)
3. Templates will appear when creating new issues

Example template (`.github/ISSUE_TEMPLATE/task.md`):
```markdown
---
name: Task
about: A general task or work item
title: ''
labels: ''
assignees: ''
---

## Description
<!-- What needs to be done? -->

## Why
<!-- Why is this needed? -->

## Acceptance Criteria
- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Criterion 3

## Notes
<!-- Any additional information -->
```

## Quick Reference

### GitHub CLI Commands

```bash
# List issues
gh issue list

# Create issue
gh issue create --title "Title" --body "Description"

# View issue
gh issue view <number>

# Add comment
gh issue comment <number> --body "Comment"

# Close issue
gh issue close <number>

# Reopen issue
gh issue reopen <number>

# List projects
gh project list

# View project
gh project view <number>
```

### Helper Script Usage

See `scripts/github-tasks.ps1` for detailed usage:
```powershell
.\scripts\github-tasks.ps1 help
```

## Tips

1. **Be Descriptive**: Good issue descriptions save time later
2. **Use Checklists**: Break down large tasks into checkboxes
3. **Link Related Issues**: Reference related issues in descriptions
4. **Update Regularly**: Move issues through columns as work progresses
5. **Close When Done**: Keep the board clean by closing completed issues

