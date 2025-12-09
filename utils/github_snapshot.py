import subprocess
import os
import csv
from datetime import datetime

def get_commit_history(repo_path=".", max_commits=50, save_csv=True):
    
    if not os.path.isdir(os.path.join(repo_path, ".git")):
        raise ValueError(f"Not a git repo: {repo_path}")

    os.chdir(repo_path)
    
    format_str = "%H|%an|%ad|%s"
    cmd = ["git", "log", f"--max-count={max_commits}", f"--pretty=format:{format_str}", "--date=iso"]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    lines = result.stdout.strip().split("\n")

    commits = []
    for line in lines:
        parts = line.split("|", 3)
        if len(parts) == 4:
            commit = {
                "hash": parts[0],
                "author": parts[1],
                "date": parts[2],
                "message": parts[3],
            }
            
            files_cmd = ["git", "show", "--name-only", "--pretty=format:", commit["hash"]]
            files_out = subprocess.run(files_cmd, capture_output=True, text=True)
            files = [f for f in files_out.stdout.splitlines() if f.strip()]
            commit["files_changed"] = files
            commits.append(commit)

    if save_csv:
        csv_path = os.path.join(repo_path, f"logs/commit_history_snapshot/{datetime.date}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["hash", "author", "date", "message", "files_changed"])
            for c in commits:
                writer.writerow([
                    c["hash"],
                    c["author"],
                    c["date"],
                    c["message"],
                    "; ".join(c["files_changed"]),
                ])
        print(f"[✓] Commit history saved to {csv_path}")

    return commits


if __name__ == "__main__":
    repo_dir = 'github.com/narendrarane50/MSML640-Final_Project'  
    history = get_commit_history(repo_dir, max_commits=20)
    print(f"Found {len(history)} commits.")
    for c in history[:5]:  
        print(f"{c['date']} | {c['author']} | {c['message'][:50]}...")
