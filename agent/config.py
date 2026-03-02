from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROVIDER_DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-5.2",
    "anthropic": "claude-opus-4-6",
    "openrouter": "anthropic/claude-sonnet-4-5",
    "cerebras": "qwen-3-235b-a22b-instruct-2507",
    "apple": "apple-foundation-model",
    "mlx": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
}


@dataclass(slots=True)
class AgentConfig:
    workspace: Path
    provider: str = "auto"
    model: str = ""
    reasoning_effort: str | None = "high"
    base_url: str = "https://api.openai.com/v1"  # Legacy alias for OpenAI-compatible base URL.
    api_key: str | None = None  # Legacy alias for OpenAI key.
    openai_base_url: str = "https://api.openai.com/v1"
    anthropic_base_url: str = "https://api.anthropic.com/v1"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    cerebras_base_url: str = "https://api.cerebras.ai/v1"
    apple_base_url: str = "http://127.0.0.1:8080/v1"
    mlx_base_url: str = "http://127.0.0.1:8080/v1"
    exa_base_url: str = "https://api.exa.ai"
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    openrouter_api_key: str | None = None
    cerebras_api_key: str | None = None
    apple_api_key: str | None = "local-apple"
    mlx_api_key: str | None = "mlx"
    exa_api_key: str | None = None
    voyage_api_key: str | None = None
    max_depth: int = 4
    max_steps_per_call: int = 100
    max_observation_chars: int = 5000
    command_timeout_sec: int = 45
    shell: str = "/bin/sh"
    max_files_listed: int = 100
    max_file_chars: int = 5000
    max_search_hits: int = 50
    max_shell_output_chars: int = 5000
    session_root_dir: str = ".openplanter"
    max_persisted_observations: int = 400
    max_solve_seconds: int = 0
    recursive: bool = True
    min_subtask_depth: int = 0
    acceptance_criteria: bool = True
    max_plan_chars: int = 40_000
    demo: bool = False
    mariadb_host: str = "127.0.0.1"
    mariadb_port: int = 3306
    mariadb_user: str = "root"
    mariadb_password: str = ""
    mariadb_database: str | None = None
    apple_discovered: bool = False
    mlx_discovered: bool = False
    use_mariadb: bool = False
    model_timeout_sec: int = 120

    def discover_apple_bridge(self) -> None:
        """Find or start the local Apple Intelligence bridge."""
        is_apple_provider = (self.provider == "apple")
        is_apple_model = (self.model and self.model.startswith("apple"))
        is_template = "{port}" in (self.apple_base_url or "")
        is_local_target = "127.0.0.1" in (self.apple_base_url or "") or "localhost" in (self.apple_base_url or "") or is_template
        
        if not (is_apple_provider or is_apple_model):
            return
            
        if not is_local_target:
            return

        def find_bridge_port():
            # 1. Check for the port file in /tmp
            tmp_file = Path("/tmp/openplanter_bridge_port")
            if tmp_file.exists():
                try:
                    p = tmp_file.read_text(encoding="utf-8").strip()
                    if p.isdigit(): return p
                except Exception: pass
            
            # 2. Fallback: Scan running processes
            try:
                import subprocess
                ps_out = subprocess.check_output(["ps", "aux"], text=True)
                for line in ps_out.splitlines():
                    if "ps aux" in line: continue
                    if "apple-bridge" in line or "App serve" in line:
                        parts = line.split()
                        if len(parts) < 2: continue
                        pid = parts[1]
                        lsof_out = subprocess.check_output(["lsof", "-i", "-a", "-p", pid], text=True)
                        for lline in lsof_out.splitlines():
                            if "(LISTEN)" in lline:
                                addr_part = lline.split()[-2]
                                return addr_part.split(":")[-1]
            except Exception: pass
            return None

        # 1. Check if a bridge is ALREADY running
        existing_port = find_bridge_port()
        if existing_port:
            if is_template:
                self.apple_base_url = self.apple_base_url.replace("{port}", existing_port)
            else:
                self.apple_base_url = f"http://127.0.0.1:{existing_port}/v1"
            self.apple_discovered = True
            return

        # 2. If not running, clean up and start a new one
        print("🛑 Apple Bridge not found. Cleaning up stale processes...")
        subprocess.run(["pkill", "-9", "apple-bridge"], capture_output=True)
        subprocess.run(["pkill", "-9", "App"], capture_output=True)
        try: Path("/tmp/openplanter_bridge_port").unlink(missing_ok=True)
        except: pass
        
        time.sleep(0.5)

        bridge_bin = self.workspace / "agent" / "appleai" / "apple-bridge"
        if bridge_bin.exists():
            print(f"🛠 Starting local Apple Bridge...")
            subprocess.Popen(
                [str(bridge_bin), "serve"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            # Wait for it to start and bind
            for _ in range(200):
                time.sleep(0.1)
                port = find_bridge_port()
                if port:
                    print(f"✅ Bridge started on port {port}.")
                    if is_template:
                        self.apple_base_url = self.apple_base_url.replace("{port}", port)
                    else:
                        self.apple_base_url = f"http://127.0.0.1:{port}/v1"
                    self.apple_discovered = True
                    return
        
        # Final fallback if discovery failed
        if is_template:
            self.apple_base_url = self.apple_base_url.replace("{port}", "8080")
            print("⚠️ Discovery failed. Defaulting to 8080.")

    def discover_mlx_server(self) -> None:
        """Find or start the local MLX server."""
        if self.provider != "mlx":
            return
            
        import socket

        def get_free_port() -> int:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', 0))
                return s.getsockname()[1]

        def check_mlx_port(port: int) -> bool:
            """Return True if a TCP listener on *port* is accepting connections."""
            try:
                with socket.create_connection(("127.0.0.1", int(port)), timeout=0.5):
                    return True
            except OSError:
                return False

        # First, try to see if ANY mlx server is running by checking process list
        try:
            ps_out = subprocess.check_output(["ps", "aux"], text=True)
            for line in ps_out.splitlines():
                if "ps aux" in line:
                    continue
                if "mlx_lm.server" in line or ("mlx_lm" in line and "server" in line):
                    parts = line.split()
                    pid = parts[1]
                    try:
                        lsof_out = subprocess.check_output(["lsof", "-i", "-a", "-p", pid], text=True)
                        for lline in lsof_out.splitlines():
                            if "(LISTEN)" in lline:
                                p = lline.split()[-2].split(":")[-1]
                                if check_mlx_port(p):
                                    self.mlx_base_url = f"http://127.0.0.1:{p}/v1"
                                    self.mlx_discovered = True
                                    print(f"✅ MLX Server already running on port {p}")
                                    return
                    except: pass
        except: pass

        port = get_free_port()
        print(f"🛑 Booting MLX Server with {self.model} on port {port}...")
        print("   (Note: If this is the first time, it will download the model from Hugging Face which may take a few minutes.)")
        try:
            def _find_mlx_python() -> Path:
                """Return a Python executable that has mlx_lm installed."""
                candidates = [
                    self.workspace / ".venv" / "bin" / "python3",
                    self.workspace / ".venv" / "bin" / "python",
                    self.workspace / ".venv312" / "bin" / "python3",
                    self.workspace / ".venv312" / "bin" / "python",
                    Path(sys.executable),
                ]
                for candidate in candidates:
                    try:
                        result = subprocess.run(
                            [str(candidate), "-c", "import mlx_lm"],
                            capture_output=True, timeout=10,
                        )
                        if result.returncode == 0:
                            return candidate
                    except Exception:
                        pass
                print(f"⚠️ Could not find a Python with mlx_lm installed. MLX server will likely fail to start.")
                return Path(sys.executable)

            mlx_python = _find_mlx_python()
            decode_concurrency = os.getenv("OPENPLANTER_MLX_DECODE_CONCURRENCY", "2").strip() or "2"
            prompt_concurrency = os.getenv("OPENPLANTER_MLX_PROMPT_CONCURRENCY", "1").strip() or "1"
            max_tokens = os.getenv("OPENPLANTER_MLX_MAX_TOKENS", "2048").strip() or "2048"
            print(
                "   MLX launch profile: "
                f"decode={decode_concurrency}, prompt={prompt_concurrency}, max_tokens={max_tokens}, "
                f"python={mlx_python}"
            )
            cmd = [
                str(mlx_python),
                "-m",
                "mlx_lm",
                "server",
                "--model",
                self.model,
                "--port",
                str(port),
                "--decode-concurrency",
                decode_concurrency,
                "--prompt-concurrency",
                prompt_concurrency,
                "--max-tokens",
                max_tokens,
            ]
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            for _ in range(6000): # Allow up to 10 minutes for model download/load
                time.sleep(0.1)
                if check_mlx_port(port):
                    print(f"✅ MLX Server is live on port {port}.")
                    self.mlx_base_url = f"http://127.0.0.1:{port}/v1"
                    self.mlx_discovered = True
                    return
        except Exception as e:
            print(f"⚠️ Failed to boot MLX server: {e}")

    @classmethod
    def from_env(cls, workspace: str | Path) -> "AgentConfig":
        from dotenv import load_dotenv
        load_dotenv()
        ws = Path(workspace).expanduser().resolve()
        
        provider = os.getenv("OPENPLANTER_PROVIDER", "auto").strip().lower() or "auto"
        model = os.getenv("OPENPLANTER_MODEL", "")
        
        is_apple_active = (provider == "apple" or (provider == "auto" and model.startswith("apple")))
        
        default_timeout = 600 if is_apple_active else 45
        cmd_timeout_env = os.getenv("OPENPLANTER_CMD_TIMEOUT")
        effective_timeout = int(cmd_timeout_env) if cmd_timeout_env else default_timeout
        
        default_model_timeout = 600 if is_apple_active else 120
        model_timeout_env = os.getenv("OPENPLANTER_MODEL_TIMEOUT")
        effective_model_timeout = int(model_timeout_env) if model_timeout_env else default_model_timeout
        
        default_obs_chars = 5000 if is_apple_active else 6000
        default_shell_chars = 5000 if is_apple_active else 16000

        mariadb_db = os.getenv("MARIADB_DATABASE") or os.getenv("OPENPLANTER_DB") or os.getenv("DB")
        use_mariadb_env = os.getenv("OPENPLANTER_USE_MARIADB", "").strip().lower()
        if use_mariadb_env in ("1", "true", "yes", "y"):
            use_mariadb = True
        elif use_mariadb_env in ("0", "false", "no", "n"):
            use_mariadb = False
        else:
            use_mariadb = bool(mariadb_db)

        apple_base_url = os.getenv("OPENPLANTER_APPLE_BASE_URL") or os.getenv("APPLE_BASE_URL", "http://127.0.0.1:8080/v1")
        mlx_base_url = os.getenv("OPENPLANTER_MLX_BASE_URL") or os.getenv("MLX_BASE_URL", "http://127.0.0.1:8080/v1")
        openai_base_url = os.getenv("OPENPLANTER_OPENAI_BASE_URL") or os.getenv(
            "OPENPLANTER_BASE_URL",
            "https://api.openai.com/v1",
        )
        openai_api_key = (
            os.getenv("OPENPLANTER_OPENAI_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )

        return cls(
            workspace=ws,
            provider=provider,
            model=model,
            reasoning_effort=(os.getenv("OPENPLANTER_REASONING_EFFORT", "high").strip().lower() or None),
            base_url=openai_base_url,
            api_key=openai_api_key,
            openai_base_url=openai_base_url,
            anthropic_base_url=os.getenv("OPENPLANTER_ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1"),
            openrouter_base_url=os.getenv("OPENPLANTER_OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            cerebras_base_url=os.getenv("OPENPLANTER_CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1"),
            apple_base_url=apple_base_url,
            mlx_base_url=mlx_base_url,
            exa_base_url=os.getenv("OPENPLANTER_EXA_BASE_URL", "https://api.exa.ai"),
            openai_api_key=openai_api_key,
            anthropic_api_key=os.getenv("OPENPLANTER_ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY"),
            openrouter_api_key=os.getenv("OPENPLANTER_OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY"),
            cerebras_api_key=os.getenv("OPENPLANTER_CEREBRAS_API_KEY") or os.getenv("CEREBRAS_API_KEY"),
            apple_api_key=os.getenv("OPENPLANTER_APPLE_API_KEY") or os.getenv("APPLE_API_KEY", "local-apple"),
            mlx_api_key=os.getenv("OPENPLANTER_MLX_API_KEY") or os.getenv("MLX_API_KEY", "mlx"),
            exa_api_key=os.getenv("OPENPLANTER_EXA_API_KEY") or os.getenv("EXA_API_KEY"),
            voyage_api_key=os.getenv("OPENPLANTER_VOYAGE_API_KEY") or os.getenv("VOYAGE_API_KEY"),
            max_depth=int(os.getenv("OPENPLANTER_MAX_DEPTH", "4")),
            max_steps_per_call=int(os.getenv("OPENPLANTER_MAX_STEPS", "100")),
            max_observation_chars=int(os.getenv("OPENPLANTER_MAX_OBS_CHARS", str(default_obs_chars))),
            command_timeout_sec=effective_timeout,
            shell=os.getenv("OPENPLANTER_SHELL", "/bin/sh"),
            max_files_listed=int(os.getenv("OPENPLANTER_MAX_FILES", "100")),
            max_file_chars=int(os.getenv("OPENPLANTER_MAX_FILE_CHARS", "5000")),
            max_search_hits=int(os.getenv("OPENPLANTER_MAX_SEARCH_HITS", "50")),
            max_shell_output_chars=int(os.getenv("OPENPLANTER_MAX_SHELL_CHARS", str(default_shell_chars))),
            session_root_dir=os.getenv("OPENPLANTER_SESSION_DIR", ".openplanter"),
            max_persisted_observations=int(os.getenv("OPENPLANTER_MAX_PERSISTED_OBS", "400")),
            max_solve_seconds=int(os.getenv("OPENPLANTER_MAX_SOLVE_SECONDS", "0")),
            recursive=os.getenv("OPENPLANTER_RECURSIVE", "true").strip().lower() in ("1", "true", "yes"),
            min_subtask_depth=int(os.getenv("OPENPLANTER_MIN_SUBTASK_DEPTH", "0")),
            acceptance_criteria=os.getenv("OPENPLANTER_ACCEPTANCE_CRITERIA", "true").strip().lower() in ("1", "true", "yes"),
            max_plan_chars=int(os.getenv("OPENPLANTER_MAX_PLAN_CHARS", "40000")),
            demo=os.getenv("OPENPLANTER_DEMO", "").strip().lower() in ("1", "true", "yes"),
            mariadb_host=os.getenv("MARIADB_HOST", "127.0.0.1"),
            mariadb_port=int(os.getenv("MARIADB_PORT", "3306")),
            mariadb_user=os.getenv("MARIADB_USER", "root"),
            mariadb_password=os.getenv("MARIADB_PASSWORD", ""),
            mariadb_database=mariadb_db,
            apple_discovered=False,
            use_mariadb=use_mariadb,
            model_timeout_sec=effective_model_timeout,
        )
