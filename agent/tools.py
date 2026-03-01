from __future__ import annotations

import ast
import fnmatch
import json
import os
import signal
import shutil
import subprocess
import tempfile
import threading
import urllib.error
import urllib.request
import re as _re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


_MAX_WALK_ENTRIES = 10000


class ToolError(RuntimeError):
    """Base class for tool execution failures."""


@dataclass(slots=True)
class WorkspaceTools:
    root: Path
    max_file_chars: int = 20000
    max_files_listed: int = 400
    max_search_hits: int = 200
    exa_api_key: str | None = None
    exa_base_url: str = "https://api.exa.ai"
    mariadb_host: str = "127.0.0.1"
    mariadb_port: int = 3306
    mariadb_user: str = "root"
    mariadb_password: str = ""
    mariadb_database: str | None = None
    command_timeout_sec: int = 45
    max_shell_output_chars: int = 16000
    shell: str = "/bin/sh"
    apple_mode: bool = False

    _bg_jobs: dict[int, tuple[subprocess.Popen, Any, Path]] = field(init=False)
    _bg_next_id: int = field(init=False)
    _files_read: set[Path] = field(init=False)
    _parallel_write_claims: dict[str, dict[Path, str]] = field(init=False)
    _parallel_lock: threading.Lock = field(init=False)
    _scope_local: threading.local = field(init=False)

    def __post_init__(self) -> None:
        self.root = self.root.expanduser().resolve()
        if not self.root.exists():
            raise ToolError(f"Workspace does not exist: {self.root}")
        self._bg_jobs = {}
        self._bg_next_id = 1
        self._files_read = set()
        self._parallel_write_claims = {}
        self._parallel_lock = threading.Lock()
        self._scope_local = threading.local()

    def _clip(self, text: str, max_chars: int) -> str:
        # If in Apple mode, we force a much smaller clip for all observations
        effective_max = 1000 if self.apple_mode else max_chars
        if len(text) <= effective_max:
            return text
        half = effective_max // 2
        return text[:half] + f"\n\n[... clipped {len(text) - effective_max} chars ...]\n\n" + text[-half:]

    def list_files(self, glob: str | None = None) -> str:
        """List files in the workspace, optionally filtered by a glob pattern."""
        files = self._repo_files(glob, self.max_files_listed)
        if not files:
            return "(no files found)"
        return "\n".join(files)

    def search_files(self, pattern: str, glob: str | None = None) -> str:
        """Search for a regex pattern in workspace files."""
        # Use ripgrep if available for speed and better defaults.
        if shutil.which("rg"):
            cmd = ["rg", "--column", "--line-number", "--no-heading", "--color", "never", pattern]
            if glob:
                cmd.extend(["-g", glob])
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=self.root,
                    capture_output=True,
                    text=True,
                    timeout=self.command_timeout_sec,
                    start_new_session=True,
                )
            except subprocess.TimeoutExpired:
                return "Search timed out."
            
            output = proc.stdout or ""
            if not output and proc.stderr:
                return f"ripgrep error: {proc.stderr}"
            if not output:
                return "(no matches)"
            return self._clip(output, self.max_shell_output_chars)

        # Fallback to simple python search.
        files = self._repo_files(glob, self.max_files_listed)
        regex = _re.compile(pattern, _re.IGNORECASE)
        matches = []
        for rel in files:
            p = self.root / rel
            try:
                content = p.read_text(encoding="utf-8", errors="replace")
                for idx, line in enumerate(content.splitlines(), 1):
                    if regex.search(line):
                        matches.append(f"{rel}:{idx}:{line}")
                        if len(matches) >= self.max_search_hits:
                            break
            except OSError:
                continue
            if len(matches) >= self.max_search_hits:
                break
        return "\n".join(matches) if matches else "(no matches)"

    def mariadb_query(self, query: str, database: str | None = None) -> str:
        """Execute a SQL query against MariaDB and return formatted results."""
        try:
            import pymysql
        except ImportError:
            return "Error: 'pymysql' library not installed."

        db_name = database or self.mariadb_database
        try:
            connection = pymysql.connect(
                host=self.mariadb_host,
                port=self.mariadb_port,
                user=self.mariadb_user,
                password=self.mariadb_password,
                database=db_name,
                cursorclass=pymysql.cursors.DictCursor,
                connect_timeout=10
            )
            try:
                with connection.cursor() as cursor:
                    cursor.execute(query)
                    if query.strip().lower().startswith(("select", "show", "describe", "explain")):
                        rows = cursor.fetchall()
                        if not rows:
                            return "(no results)"
                        
                        total_rows = len(rows)
                        # Detect if it's a simple list (like SHOW TABLES)
                        is_show_tables = "show tables" in query.lower()
                        
                        if is_show_tables:
                            # Return a simpler format for table lists (headerless bullet points)
                            # Increased limit to 100 to show entire typical schemas
                            table_names = [str(list(row.values())[0]) for row in rows[:100]]
                            result = "Tables found in database:\n" + "\n".join(f"- {name}" for name in table_names)
                            if total_rows > 100:
                                result += f"\n\n[... and {total_rows - 100} more tables. Use LIMIT to see more.]"
                            return self._clip(result, self.max_shell_output_chars)

                        # Normal table formatting for other queries
                        headers = list(rows[0].keys())
                        col_widths = {h: len(h) for h in headers}
                        for row in rows:
                            for h in headers:
                                col_widths[h] = max(col_widths[h], len(str(row[h])))
                        
                        lines = []
                        header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
                        lines.append(header_line)
                        lines.append("-" * len(header_line))
                        for row in rows:
                            lines.append(" | ".join(str(row[h]).ljust(col_widths[h]) for h in headers))
                        
                        result = "\n".join(lines)
                        return self._clip(result, self.max_shell_output_chars)
                    else:
                        connection.commit()
                        return f"Query executed successfully. Rows affected: {cursor.rowcount}"
            finally:
                connection.close()
        except Exception as exc:
            return f"MariaDB Error: {exc}"

    def _repo_files(self, glob: str | None, max_files: int) -> list[str]:
        lines: list[str]
        if shutil.which("rg"):
            cmd = ["rg", "--files", "--hidden", "-g", "!.git"]
            if glob:
                cmd.extend(["-g", glob])
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=self.root,
                    capture_output=True,
                    text=True,
                    timeout=self.command_timeout_sec,
                    start_new_session=True,
                )
            except subprocess.TimeoutExpired:
                return []
            lines = [ln for ln in (proc.stdout or "").splitlines() if ln.strip()]
        else:
            lines = []
            count = 0
            for dirpath, dirnames, filenames in os.walk(self.root):
                dirnames[:] = [d for d in dirnames if d != ".git"]
                count += len(filenames)
                if count > _MAX_WALK_ENTRIES:
                    break
                for fn in filenames:
                    rel = (Path(dirpath) / fn).relative_to(self.root).as_posix()
                    if glob and not fnmatch.fnmatch(rel, glob):
                        continue
                    lines.append(rel)
        return lines[:max_files]

    def _python_symbols(self, text: str) -> list[dict[str, Any]]:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return []
        symbols: list[dict[str, Any]] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols.append({"kind": "function", "name": node.name, "line": int(node.lineno)})
            elif isinstance(node, ast.ClassDef):
                symbols.append({"kind": "class", "name": node.name, "line": int(node.lineno)})
        return symbols

    def repo_map(self, glob: str | None = None) -> str:
        """Return a tree-like map of the workspace, including top-level symbols."""
        files = self._repo_files(glob, 100) # Capped for prompt sanity
        if not files:
            return "(no files found)"
        
        # Build tree structure
        tree = {}
        for f in files:
            parts = f.split("/")
            curr = tree
            for p in parts:
                curr = curr.setdefault(p, {})

        def render(node, indent=""):
            lines = []
            keys = sorted(node.keys())
            for idx, k in enumerate(keys):
                is_last = (idx == len(keys) - 1)
                prefix = indent + ("└── " if is_last else "├── ")
                lines.append(prefix + k)
                new_indent = indent + ("    " if is_last else "│   ")
                lines.extend(render(node[k], new_indent))
            return lines

        return "\n".join(render(tree))

    def read_file(self, file_path: str, start_line: int = 1, end_line: int | None = None) -> str:
        """Read a file's content, optionally within a line range."""
        p = self._resolve_path(file_path)
        if not p.exists():
            return f"Error: File not found: {file_path}"
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            total = len(lines)
            start = max(0, start_line - 1)
            end = min(total, end_line) if end_line else total
            content = "\n".join(lines[start:end])
            
            header = f"File {file_path} (lines {start_line}-{end} of {total}):\n"
            return header + self._clip(content, self.max_file_chars)
        except OSError as exc:
            return f"Error reading file: {exc}"

    def write_file(self, file_path: str, content: str) -> str:
        """Write content to a file, overwriting existing content."""
        p = self._resolve_path(file_path)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return f"Successfully wrote to {file_path}"
        except OSError as exc:
            return f"Error writing file: {exc}"

    def edit_file(self, file_path: str, old_str: str, new_str: str) -> str:
        """Replace exact occurrences of a string in a file."""
        p = self._resolve_path(file_path)
        if not p.exists():
            return f"Error: File not found: {file_path}"
        try:
            content = p.read_text(encoding="utf-8")
            if old_str not in content:
                return f"Error: String not found in {file_path}"
            new_content = content.replace(old_str, new_str)
            p.write_text(new_content, encoding="utf-8")
            return f"Successfully edited {file_path}"
        except OSError as exc:
            return f"Error editing file: {exc}"

    def hashline_edit(self, file_path: str, line_hash: str, new_content: str) -> str:
        """Replace a line identified by its MD5 hash."""
        import hashlib
        p = self._resolve_path(file_path)
        if not p.exists():
            return f"Error: File not found: {file_path}"
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
            for idx, ln in enumerate(lines):
                h = hashlib.md5(ln.encode("utf-8")).hexdigest()
                if h == line_hash:
                    lines[idx] = new_content
                    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
                    return f"Successfully replaced line in {file_path}"
            return f"Error: Line with hash {line_hash} not found in {file_path}"
        except OSError as exc:
            return f"Error editing file: {exc}"

    def apply_patch(self, patch: str) -> str:
        """Apply a unified diff format patch to the workspace."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as tf:
            tf.write(patch)
            patch_path = tf.name
        
        try:
            cmd = ["patch", "-p1", "-i", patch_path]
            proc = subprocess.run(
                cmd,
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=self.command_timeout_sec,
                start_new_session=True,
            )
            if proc.returncode == 0:
                return "Patch applied successfully."
            return f"Error applying patch: {proc.stderr or proc.stdout}"
        except Exception as exc:
            return f"Error applying patch: {exc}"
        finally:
            os.unlink(patch_path)

    def run_shell(self, command: str) -> str:
        """Run a shell command and return the output."""
        try:
            proc = subprocess.run(
                command,
                shell=True,
                executable=self.shell,
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=self.command_timeout_sec,
                start_new_session=True,
            )
            output = proc.stdout or ""
            if proc.stderr:
                output += f"\nStderr:\n{proc.stderr}"
            if not output:
                return "(no output)"
            return self._clip(output, self.max_shell_output_chars)
        except subprocess.TimeoutExpired:
            return f"Command timed out after {self.command_timeout_sec} seconds."
        except Exception as exc:
            return f"Error executing command: {exc}"

    def run_shell_bg(self, command: str) -> str:
        """Run a shell command in the background."""
        out_file = tempfile.NamedTemporaryFile(dir=self.root, suffix=".bg_out", delete=False)
        out_path = Path(out_file.name)
        try:
            proc = subprocess.Popen(
                command,
                shell=True,
                executable=self.shell,
                cwd=self.root,
                stdout=out_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception as exc:
            return f"Failed to start background command: {exc}"
        
        job_id = self._bg_next_id
        self._bg_next_id += 1
        self._bg_jobs[job_id] = (proc, out_file, out_path)
        return f"Background job started: job_id={job_id}, pid={proc.pid}"

    def check_shell_bg(self, job_id: int) -> str:
        """Check the status and output of a background job."""
        if job_id not in self._bg_jobs:
            return f"Error: Background job {job_id} not found."
        
        proc, _, out_path = self._bg_jobs[job_id]
        status = "Running" if proc.poll() is None else f"Finished (Exit Code: {proc.returncode})"
        
        try:
            output = out_path.read_text(encoding="utf-8", errors="replace")
            clipped = self._clip(output, self.max_shell_output_chars)
            return f"Status: {status}\nOutput:\n{clipped}"
        except OSError as exc:
            return f"Status: {status}\nError reading output: {exc}"

    def kill_shell_bg(self, job_id: int) -> str:
        """Kill a background job."""
        if job_id not in self._bg_jobs:
            return f"Error: Background job {job_id} not found."
        
        proc, _, _ = self._bg_jobs[job_id]
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            return f"Successfully sent SIGTERM to background job {job_id}."
        except Exception as exc:
            return f"Error killing job: {exc}"

    def web_search(self, query: str) -> str:
        """Perform a web search using Exa."""
        if not self.exa_api_key:
            return "Error: EXA_API_KEY not configured."
        
        url = self.exa_base_url.rstrip("/") + "/search"
        payload = {
            "query": query,
            "useAutoprompt": True,
            "numResults": 5,
            "highlights": True,
        }
        headers = {
            "x-api-key": self.exa_api_key,
            "content-type": "application/json",
        }
        
        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                results = []
                for r in data.get("results", []):
                    title = r.get("title", "(no title)")
                    url = r.get("url", "(no url)")
                    highlight = r.get("highlights", ["(no highlight)"])[0]
                    results.append(f"Title: {title}\nURL: {url}\nSnippet: {highlight}\n")
                return "\n".join(results) if results else "(no results)"
        except Exception as exc:
            return f"Web search error: {exc}"

    def fetch_url(self, urls: list[str]) -> str:
        """Fetch and extract text content from one or more URLs."""
        if not self.exa_api_key:
            return "Error: EXA_API_KEY not configured."
        
        url = self.exa_base_url.rstrip("/") + "/contents"
        payload = {"urls": urls}
        headers = {
            "x-api-key": self.exa_api_key,
            "content-type": "application/json",
        }
        
        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                pages = []
                for r in data.get("results", []):
                    pages.append(
                        {
                            "url": r.get("url"),
                            "title": r.get("title"),
                            "text": self._clip(str(r.get("text", "")), 8000),
                        }
                    )
                output = {"pages": pages, "total": len(pages)}
                return self._clip(json.dumps(output, indent=2, ensure_ascii=True), self.max_file_chars)
        except Exception as exc:
            return f"Error fetching URLs: {exc}"

    def _resolve_path(self, raw_path: str) -> Path:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        return candidate.expanduser().resolve()
