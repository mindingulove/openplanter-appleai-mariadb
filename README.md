# OpenPlanter (Silicon & SQL Fork)

A recursive-language-model investigation agent with a terminal UI. This version is a specialized fork of the original OpenPlanter, enhanced for on-device performance and direct database integration.

### 🚀 How this fork differs:
- **Local MLX AI (8K Context):** Native integration with Apple Silicon via `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit`. Run investigations fully local with an **8,192-token context window**. The MLX server is auto-discovered or auto-booted with zero config.
- **Privacy-First by Default:** No cloud API keys required. All inference runs on-device using your Mac's GPU via the MLX framework.
- **Direct Database Access:** Built-in support for MariaDB/MySQL. The agent treats your database as a first-class workspace component, autonomously discovering schemas and executing complex analytical queries.
- **Deep Persistence:** Enhanced `settings.json` that remembers not just models, but custom provider URLs, database credentials, and dynamic port mappings using `{port}` templates.
- **Multi-Provider:** Seamlessly switch between local MLX, OpenAI, Anthropic, OpenRouter, and Cerebras with a single flag.

OpenPlanter ingests heterogeneous datasets — corporate registries, campaign finance records, lobbying disclosures, government contracts, and more — resolves entities across them, and surfaces non-obvious connections through evidence-backed analysis. It operates autonomously with file I/O, shell execution, web search, recursive sub-agent delegation, and direct database access. **The configured MariaDB/MySQL database is treated as an integral part of the agent's workspace.**

## Quickstart

```bash
# Install
pip install -e .

# Install MLX for on-device inference (Apple Silicon only)
pip install mlx-lm

# Launch with local MLX model (auto-boots server)
python3 -m agent --provider mlx

# Or launch the interactive TUI
openplanter-agent --workspace /path/to/your/project
```

Or run a single task headlessly:

```bash
openplanter-agent --provider mlx --task "Cross-reference vendor payments against lobbying disclosures and flag overlaps" --workspace ./data
```

## Supported Providers

| Provider | Default Model | Notes |
|----------|---------------|-------|
| **MLX** | `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` | On-device, no API key, 8K context |
| OpenAI | `gpt-5.2` | `OPENAI_API_KEY` |
| Anthropic | `claude-opus-4-6` | `ANTHROPIC_API_KEY` |
| OpenRouter | `anthropic/claude-sonnet-4-5` | `OPENROUTER_API_KEY` |
| Cerebras | `qwen-3-235b-a22b-instruct-2507` | `CEREBRAS_API_KEY` |

### 🧠 MLX On-Device Inference

OpenPlanter supports running fully local on Apple Silicon using the [MLX](https://github.com/ml-explore/mlx) framework.

- **Zero-Config:** No API key required. Just install `mlx-lm` and run with `--provider mlx`.
- **Auto-Discovery:** Detects any already-running `mlx_lm` server via process scanning.
- **Auto-Boot:** If no server is found, automatically launches one with the configured model.
- **8K Context Window:** Uses `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` (~4GB RAM), which supports 8,192 tokens of context.
- **Configurable Model:** Override via `OPENPLANTER_MODEL` or `--model mlx-community/YOUR-MODEL`.
- **Configurable Tokens:** Set `OPENPLANTER_MLX_MAX_TOKENS` (default: 4096 output tokens per response).

```bash
# Use a larger model
python3 -m agent --provider mlx --model mlx-community/Qwen2.5-Coder-14B-Instruct-4bit

# Use a smaller/faster model
python3 -m agent --provider mlx --model mlx-community/Qwen2.5-Coder-3B-Instruct-4bit
```

### 🌐 OpenRouter Integration
OpenPlanter works natively with OpenRouter. If a model name contains a `/` (e.g. `google/gemini-2.0-flash-001`), the agent automatically routes through OpenRouter using your `OPENROUTER_API_KEY`.

## Agent Tools

The agent has access to 20 tools, organized around its investigation workflow:

**Dataset ingestion & workspace** — `list_files`, `search_files`, `repo_map`, `read_file`, `write_file`, `edit_file`, `hashline_edit`, `apply_patch` — load, inspect, and transform source datasets; write structured findings.

**Database Access** — `mariadb_query` — execute SQL queries against local or remote MariaDB/MySQL databases and receive formatted results. **(Note: Requires `use_mariadb=yes`)**

**Shell execution** — `run_shell`, `run_shell_bg`, `check_shell_bg`, `kill_shell_bg` — run analysis scripts, data pipelines, and validation checks.

**Web** — `web_search` (Exa), `fetch_url` — pull public records, verify entities, and retrieve supplementary data.

**Planning & delegation** — `think`, `subtask`, `execute`, `list_artifacts`, `read_artifact` — decompose investigations into focused sub-tasks, each with acceptance criteria and independent verification.

## MariaDB / MySQL Integration

Configure your database connection via `.env` or environment variables:
```env
MARIADB_HOST=127.0.0.1
MARIADB_PORT=3306
MARIADB_USER=your_user
MARIADB_PASSWORD=your_pass
MARIADB_DATABASE=your_db
# Aliases
DB=your_db
# Toggles
OPENPLANTER_USE_MARIADB=yes
```
The agent automatically enables MariaDB integration if a database is provided. You can explicitly toggle it using `OPENPLANTER_USE_MARIADB=no`.

## CLI Reference

```bash
openplanter-agent [options]
```

### Model & Provider
| Flag | Description |
|------|-------------|
| `--provider NAME` | `auto`, `mlx`, `openrouter`, `openai`, `anthropic`, `cerebras` |
| `--model NAME` | Model name (e.g. `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit`) |
| `--reasoning-effort LEVEL`| `low`, `medium`, `high`, or `none` |
| `--list-models` | Fetch available models from the provider API |

### Database Integration
| Flag | Description |
|------|-------------|
| `--use-mariadb` | Enable MariaDB/SQL integration (default if --db is set) |
| `--no-mariadb` | Disable MariaDB integration (agent will not see DB tools) |
| `--default-use-mariadb` | Save MariaDB integration as enabled by default |
| `--default-no-mariadb` | Save MariaDB integration as disabled by default |
| `--db VAL` | Short alias for the MariaDB database name |

### Execution & UI
| Flag | Description |
|------|-------------|
| `--task OBJECTIVE` | Run a single task and exit (headless) |
| `--timeout N` | Shell timeout in seconds (default: 45) |
| `--recursive` | Enable recursive sub-agent delegation |
| `--acceptance-criteria` | Judge subtask results with a lightweight model |
| `--demo` | Censor entity names and workspace paths in output |

## TUI Commands

Inside the interactive REPL:

| Command | Action |
|---------|--------|
| `/model` | Show current model and provider |
| `/model NAME` | Switch model (e.g. `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit`) |
| `/model NAME --save` | Switch and persist model & provider URL as default |
| `/model list [all]` | List available models |
| `/db NAME [--save]` | Quickly set and save the active MariaDB database |
| `/mariadb` | Show current database configuration and status |
| `/mariadb enable/disable` | Toggle MariaDB integration live |
| `/mariadb <key> <val>` | Set MariaDB config (host, port, user, pass, db) |
| `/mariadb <key> <val> --save` | Set and persist database settings to `settings.json` |
| `/reasoning LEVEL` | Change reasoning effort |
| `/status` | Show session status and token usage |
| `/clear` | Clear the screen |
| `/quit` | Exit |

## Project Structure

```
agent/
  __main__.py    CLI entry point and REPL
  engine.py      Recursive language model engine
  runtime.py     Session persistence and lifecycle
  model.py       Provider-agnostic LLM abstraction
  builder.py     Engine/model factory (MLX auto-discovery)
  tools.py       Workspace tool implementations (inc. MariaDB)
  tool_defs.py   Tool JSON schemas
  prompts.py     System prompt construction
  config.py      Configuration and MLX Auto-Discovery logic
  credentials.py Credential management
  tui.py         Rich terminal UI + Slash Commands
  settings.py    Persistent settings (inc. DB & Provider URLs)
```

## Development

```bash
# Install in editable mode
pip install -e .

# Install MLX for local inference
pip install mlx-lm

# Run with local MLX model
python3 -m agent --provider mlx

# Run with a specific MLX model
python3 -m agent --provider mlx --model mlx-community/Qwen2.5-Coder-14B-Instruct-4bit

# Run with cloud provider
python3 -m agent --provider anthropic
```

## License

See [VISION.md](VISION.md) for the project's design philosophy and roadmap.
