from __future__ import annotations

import subprocess

import click
from rich.console import Console

console = Console()


class GitIntegration:
    def __init__(self):
        self.git_available = self._check_git()
    
    def _check_git(self) -> bool:
        try:
            subprocess.run(["git", "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def generate_commit_message(self, diff: str | None = None) -> str:
        if not self.git_available:
            return "Git not available"
        
        try:
            if diff:
                result = subprocess.run(
                    ["git", "diff", "--cached"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                diff_content = result.stdout
            else:
                result = subprocess.run(
                    ["git", "diff", "HEAD~1"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                diff_content = result.stdout
            
            return self._analyze_diff_for_commit(diff_content)
        except subprocess.CalledProcessError as e:
            return f"Error getting git diff: {e}"
    
    def _analyze_diff_for_commit(self, diff: str) -> str:
        if not diff:
            return "Initial commit"
        
        lines_added = len([line for line in diff.split('\n') if line.startswith('+')])
        lines_removed = len([line for line in diff.split('\n') if line.startswith('-')])
        
        if lines_added > 0 and lines_removed == 0:
            return f"Add new features ({lines_added} lines)"
        elif lines_removed > 0 and lines_added == 0:
            return f"Remove code ({lines_removed} lines)"
        elif lines_added > lines_removed:
            return f"Update code (+{lines_added}, -{lines_removed})"
        else:
            return f"Refactor code (+{lines_added}, -{lines_removed})"
    
    def review_changes(self, diff_ref: str = "HEAD~1") -> str:
        if not self.git_available:
            return "Git not available"
        
        try:
            result = subprocess.run(
                ["git", "diff", diff_ref],
                capture_output=True,
                text=True,
                check=True
            )
            
            return result.stdout
        except subprocess.CalledProcessError as e:
            return f"Error getting diff: {e}"
    
    def generate_changelog(self, since_ref: str = "HEAD~5") -> str:
        if not self.git_available:
            return "Git not available"
        
        try:
            result = subprocess.run(
                ["git", "log", f"{since_ref}..HEAD", "--oneline"],
                capture_output=True,
                text=True,
                check=True
            )
            
            commits = result.stdout.strip().split('\n')
            
            changelog = "## Changelog\n\n"
            for commit in commits:
                changelog += f"- {commit}\n"
            
            return changelog
        except subprocess.CalledProcessError as e:
            return f"Error getting git log: {e}"
    
    def summarize_pr(self, pr_branch: str = "main") -> str:
        if not self.git_available:
            return "Git not available"
        
        try:
            result = subprocess.run(
                ["git", "log", f"{pr_branch}..HEAD", "--oneline"],
                capture_output=True,
                text=True,
                check=True
            )
            
            commits = result.stdout.strip().split('\n')
            commit_count = len(commits) if commits[0] else 0
            
            summary = f"PR Summary: {commit_count} commits\n\n"
            for commit in commits[:5]:
                summary += f"- {commit}\n"
            
            if commit_count > 5:
                summary += f"... and {commit_count - 5} more commits"
            
            return summary
        except subprocess.CalledProcessError as e:
            return f"Error getting PR diff: {e}"


@click.group("git")
def git_group():
    """Git integration features."""
    pass


@git_group.command("commit")
@click.option("--forge", is_flag=True, help="Use AI to write commit message")
def git_commit(forge: bool):
    """Generate AI commit message."""
    if forge:
        integration = GitIntegration()
        message = integration.generate_commit_message()
        
        console.print("\n[bold cyan]AI Commit Message:[/]\n")
        console.print(message)
        
        if click.confirm("\nUse this commit message?"):
            try:
                subprocess.run(["git", "commit", "-m", message], check=True)
                console.print("[green]✓[/] Commit created")
            except subprocess.CalledProcessError as e:
                console.print(f"[red]Failed to commit: {e}[/]")
    else:
        console.print("[yellow]Use --forge flag to generate AI commit message[/]")


@git_group.command("review")
@click.option("--diff", default="HEAD~1", help="Git reference to review")
def git_review(diff: str):
    """Review git changes with AI."""
    integration = GitIntegration()
    changes = integration.review_changes(diff)
    
    console.print(f"\n[bold cyan]Git Changes ({diff}):[/]\n")
    console.print(changes[:500])
    
    if len(changes) > 500:
        console.print("\n[dim]... (truncated)[/]")


@git_group.command("changelog")
@click.option("--since", default="HEAD~5", help="Git reference for changelog")
def git_changelog(since: str):
    """Generate changelog from commits."""
    integration = GitIntegration()
    changelog = integration.generate_changelog(since)
    
    console.print(changelog)


@git_group.command("pr-summary")
@click.option("--branch", default="main", help="Base branch")
def git_pr_summary(branch: str):
    """Summarize PR changes."""
    integration = GitIntegration()
    summary = integration.summarize_pr(branch)
    
    console.print(summary)


git_command = git_group