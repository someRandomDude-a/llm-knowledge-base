# llm-knowledge-base

A FastAPI HTTP service that turns **Google Gemini** into a tool-using agent by
bridging it to one or more **[MCP (Model Context Protocol)](https://modelcontextprotocol.io)**
servers. Every conversation turn is persisted to **PostgreSQL** so each user
gets a rolling window of context across requests, while Gemini is free to
call whatever tools the connected MCP servers expose.

The project ships in two flavours that share the same core:

| Mode | Entry point | Use case |
|------|-------------|----------|
| **CLI REPL** | `python code/main.py` | Local testing, single-user chat |
| **HTTP API** | `python code/serve.py` (dev) / Docker (prod) | Multi-user service with persistent history |

> **Looking for a starting config?** Copy [`.env.example`](./.env.example) to
> `.env` and fill in your secrets. The example covers every environment
> variable the app reads.

---

## Table of contents

- [llm-knowledge-base](#llm-knowledge-base)
  - [Table of contents](#table-of-contents)
  - [Architecture](#architecture)
  - [Features](#features)
  - [Get started](#get-started)
    - [1. Local Python (CLI)](#1-local-python-cli)
    - [2. Local Python (HTTP API)](#2-local-python-http-api)
    - [3. Docker (image)](#3-docker-image)
    - [4. Docker Compose (recommended)](#4-docker-compose-recommended)
  - [Environment variables](#environment-variables)
  - [Preparing `mcp.json`](#preparing-mcpjson)
    - [Two transport types are supported](#two-transport-types-are-supported)
    - [Putting it together](#putting-it-together)
    - [How `${...}` expansion works](#how--expansion-works)
    - [Verifying the config](#verifying-the-config)
  - [Docker Compose — explained](#docker-compose--explained)
    - [Ready-to-use `docker-compose.yml`](#ready-to-use-docker-composeyml)
    - [Running it](#running-it)
    - [What each piece does](#what-each-piece-does)
    - [Tips](#tips)
  - [API reference](#api-reference)
    - [`GET /health`](#get-health)
    - [`GET /tools`](#get-tools)
    - [`POST /chat`](#post-chat)
    - [`GET /docs` and `GET /redoc`](#get-docs-and-get-redoc)
  - [CLI reference](#cli-reference)
  - [Database schema](#database-schema)
  - [How the tool loop works](#how-the-tool-loop-works)
  - [Project layout](#project-layout)
  - [Troubleshooting](#troubleshooting)
  - [License](#license)

---

## Architecture

```
                     ┌────────────────────┐
                     │   Client / curl    │
                     └─────────┬──────────┘
                               │ HTTP (ASGI)
                               ▼
                     ┌────────────────────┐
                     │   FastAPI app      │
                     │   (api.py)         │
                     │   /health /tools   │
                     │   /chat /docs      │
                     └─────┬──────┬───────┘
                           │      │
              lifespan     │      │  ChatHistoryDB
   (startup/shutdown ──────┤      │ (psycopg2 pool)
            per worker)    │      │
                           ▼      ▼
                  ┌────────────┐  ┌──────────────────┐
                  │ GeminiCore │  │  PostgreSQL      │
                  │ (gemini.py)│  │  user_chat_      │
                  └─────┬──────┘  │  history table   │
                        │         └──────────────────┘
                        │ function calls
                        ▼
              ┌──────────────────────┐
              │  MCP servers         │
              │  (stdio or HTTP,     │
              │   from mcp.json)     │
              └──────────────────────┘
```

The codebase is intentionally split:

- **`code/gemini.py`** — talks to Gemini and to MCP. Owns the tool loop. No
  history, no memory, no persistence.
- **`code/agent.py`** — high-level `Agent` and `AgentManager`. Loads rolling
  context from the DB, calls `GeminiCore`, persists the new turn.
- **`code/db.py`** — thread-safe PostgreSQL pool, schema bootstrap, rolling
  window read.
- **`code/context.py`** — pure data: `Conversation`, `MemoryStore`,
  `ContextBuilder`. No I/O.
- **`code/api.py`** — FastAPI app, Pydantic models, lifespan-managed
  `AgentManager`, HTTP routes.
- **`code/serve.py`** — production entry point that boots uvicorn.
- **`code/main.py`** — CLI REPL entry point.

---

## Features

- 🔌 **Pluggable MCP servers** — any number of `stdio` or streamable-HTTP MCP
  servers, configured in a single `mcp.json`.
- 🧰 **Automatic tool discovery** — every tool an MCP server exposes is
  converted into a Gemini function declaration. Name collisions across
  servers are auto-prefixed (`<server>__<tool>`).
- 🧠 **Persistent rolling context** — every user gets a configurable rolling
  window of past turns re-injected on each request. The DB is append-only;
  nothing is ever deleted.
- 🧵 **One LLM, many users** — `AgentManager` shares a single
  `GeminiCore` (and its MCP sessions) across all per-user `Agent`
  instances. Spin up thousands of users without spawning thousands of MCP
  processes.
- ⚡ **Async-native** — FastAPI handlers `await` Gemini and MCP directly.
  No sync↔async bridge, no thread pools for request handling.
- 🐳 **Docker-ready** — `docker/Dockerfile` (Python 3.14-slim, port 8005) and
  a GitHub Actions workflow that pushes images to GHCR.
- 🔐 **Environment-variable expansion** — `${VAR}` references inside
  `mcp.json` are resolved from the process environment at startup, so
  secrets never need to live in the file itself.
- 📖 **Auto-generated OpenAPI** — interactive docs at `/docs` (Swagger UI)
  and `/redoc` come for free with FastAPI.

---

## Get started

Pick one of the four paths below. The first three are for poking at the
project; the fourth (Docker Compose) is the recommended way to actually run
it.

### 1. Local Python (CLI)

The CLI mode talks directly to Gemini and your MCP servers, but it does
**not** touch PostgreSQL — it's just for trying things out.

```bash
# 1. Clone and enter the repo
git clone https://github.com/<you>/llm-knowledge-base.git
cd llm-knowledge-base

# 2. Create a venv and install deps
python3.11+ -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Provide your Gemini key
cp .env.example .env                   # then edit .env

# 4. Tell it which MCP servers to use
cp code/mcp.json.example code/mcp.json # then edit (see below)
#   (or just hand-author code/mcp.json — see "Preparing mcp.json")

# 5. Run
python code/main.py
```

You'll get a `You:` prompt. Type anything; the assistant will answer and
may call out to your MCP tools when relevant. `exit` / `quit` to leave.

### 2. Local Python (HTTP API)

Same setup, but you'll also need a PostgreSQL instance.

```bash
# 1-3. Same as above

# 4. Start PostgreSQL any way you like, e.g.:
docker run -d --name llm-pg -p 5432:5432 \
  -e POSTGRES_USER=llm -e POSTGRES_PASSWORD=llm -e POSTGRES_DB=llm \
  postgres:16

# 5. Fill in DATABASE_URL in your .env
cat >> .env <<'EOF'
DATABASE_URL=postgresql://llm:llm@localhost:5432/llm
CHAT_WINDOW_SIZE=20
HOST=0.0.0.0
PORT=8005
EOF

# 6. Run the API
python code/serve.py
```

Quick smoke test:

```bash
curl -s http://localhost:8005/health
# {"status":"ok","window_size":20,"users_cached":0}

curl -s http://localhost:8005/tools | jq .
# {"tools":[{"name":"...","description":"..."}, ...]}

# Auto-generated interactive docs
open http://localhost:8005/docs
```

### 3. Docker (image)

The provided `docker/Dockerfile` exposes the API on **port 8005**. It expects
an `mcp.json` mounted into `/app/code/mcp.json` and reads everything else
from environment variables.

```bash
# Build locally
docker build -f docker/Dockerfile -t llm-knowledge-base .

# Run (you supply PG + API in two containers, or extend this command)
docker run --rm -p 8005:8005 \
  -e GEMINI_API_KEY=your-key \
  -e DATABASE_URL=postgresql://llm:llm@host.docker.internal:5432/llm \
  -v "$PWD/code/mcp.json:/app/code/mcp.json:ro" \
  llm-knowledge-base
```

For real usage prefer the Compose recipe below — it wires the database and
the API together.

### 4. Docker Compose (recommended)

See the [Docker Compose — explained](#docker-compose--explained) section for a
full walk-through and a ready-to-use `docker-compose.yml`.

---

## Environment variables

All variables are read from the process environment. The CLI mode and the
local `serve.py` also load them from a `.env` file in the current working
directory (via `python-dotenv`). The container reads them from the
`environment:` block of `docker-compose.yml` (or `--env-file`).

A complete, copy-pasteable reference is in [`.env.example`](./.env.example).

| Variable | Required? | Default | Where it's used | Notes |
|----------|-----------|---------|-----------------|-------|
| `GEMINI_API_KEY` | **Yes** | — | `code/gemini.py` | Used to construct `genai.Client`. The process refuses to start without it. |
| `DATABASE_URL` | Yes for API mode | — | `code/db.py` | Standard PostgreSQL DSN, e.g. `postgresql://user:pass@host:5432/db`. Aliases also accepted: `POSTGRES_CONNECTION_STRING`, `POSTGRES_URL`. |
| `CHAT_WINDOW_SIZE` | No | `20` | `code/agent.py` | Number of past messages re-injected as context for each user. Pure read — old rows are never deleted. |
| `HOST` | No | `0.0.0.0` | `code/serve.py`, `code/api.py` | Bind address for uvicorn. |
| `PORT` | No | `8005` | `code/serve.py`, `code/api.py` | Bind port for uvicorn. |
| `WORKERS` | No | `2` | `code/serve.py`, `docker/Dockerfile` | Number of uvicorn worker processes. Ignored when `UVICORN_RELOAD=1`. |
| `UVICORN_RELOAD` | No | `0` | `code/serve.py` | `1` to enable uvicorn's auto-reload (dev only). Mutually exclusive with `WORKERS`. |
| `LOG_LEVEL` | No | `INFO` | `code/serve.py`, `code/api.py` | Standard Python logging level. Also passed to uvicorn. |

Anything referenced as `${SOMETHING}` inside `mcp.json` is also expanded from
the environment at startup — this is the recommended way to pass secrets to
MCP servers (see below).

---

## Preparing `mcp.json`

`mcp.json` is the **only** file you must create yourself; it is intentionally
git-ignored (`.gitignore` lists it on the last line). It follows the standard
MCP config format and is read at startup by `GeminiCore`.

Place it at `code/mcp.json` when running locally, or mount it at
`/app/code/mcp.json` when running in Docker.

### Two transport types are supported

**stdio** — spawn a local process and talk to it over stdin/stdout:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
      "env": {
        "DEBUG": "1"
      }
    }
  }
}
```

**streamable HTTP** — connect to a remote MCP server over HTTPS:

```json
{
  "mcpServers": {
    "remote-tools": {
      "url": "https://mcp.example.com/mcp",
      "headers": {
        "Authorization": "Bearer ${REMOTE_MCP_TOKEN}"
      }
    }
  }
}
```

> Each server entry must contain either `command` (stdio) or `url` (HTTP).
> Anything else raises `ValueError` at startup.

### Putting it together

You can mix and match as many servers as you want:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/data"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"
      }
    },
    "remote-knowledge": {
      "url": "https://mcp.internal.example.com/knowledge",
      "headers": {
        "Authorization": "Bearer ${INTERNAL_MCP_TOKEN}"
      }
    }
  }
}
```

### How `${...}` expansion works

`GeminiCore._expand_env` walks the parsed JSON tree and replaces every
`${VAR}` occurrence with the value of the `VAR` environment variable (or
empty string if unset). This means:

- ✅ `${GEMINI_API_KEY}`, `${GITHUB_TOKEN}`, etc. are resolved at startup.
- ✅ You can keep secrets out of the file and inside your shell / Docker
  secrets / Kubernetes secrets.
- ⚠️ The expansion is naive — there's no default-value syntax (`${X:-y}`)
  and no escaping. If you need a literal `${...}` in a value, you're out
  of luck; rename the env var.

### Verifying the config

After the API starts, every connected server and every discovered tool is
printed to the logs:

```
[MCP] Starting: filesystem
[MCP] Connected: filesystem
[MCP] filesystem: 11 tools
    - read_file
    - write_file
    - ...
[MCP] Starting: github
[MCP] Connected: github
[MCP] github: 26 tools
[MCP] Connecting HTTP: remote-knowledge
[MCP] URL: https://mcp.internal.example.com/knowledge
[MCP] Connected: remote-knowledge
[MCP] remote-knowledge: 8 tools
[MCP] Connected servers: 3
[MCP] Available tools: 45
```

You can also hit `GET /tools` to get the same list as JSON.

---

## Docker Compose — explained

The image in `docker/Dockerfile` runs **only** the API. PostgreSQL is a
separate concern, and a Compose file is the cleanest way to wire them
together with the right network, volumes, and environment plumbing.

### Ready-to-use `docker-compose.yml`

Drop this at the repo root (next to `docker/`):

```yaml
# docker-compose.yml
services:
  db:
    image: postgres:16-alpine
    restart: unless-stopped
    environment:
      POSTGRES_USER: llm
      POSTGRES_PASSWORD: llm
      POSTGRES_DB: llm
    volumes:
      - llm_pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U llm -d llm"]
      interval: 5s
      timeout: 3s
      retries: 10
    ports:
      - "5432:5432"   # comment out if you don't want host access

  api:
    build:
      context: .
      dockerfile: docker/Dockerfile
    restart: unless-stopped
    depends_on:
      db:
        condition: service_healthy
    environment:
      GEMINI_API_KEY: ${GEMINI_API_KEY:?set GEMINI_API_KEY in your .env}
      DATABASE_URL: postgresql://llm:llm@db:5432/llm
      CHAT_WINDOW_SIZE: "20"
      LOG_LEVEL: INFO
      WORKERS: "2"
    volumes:
      # Mount your mcp.json so you can edit it without rebuilding.
      - ./code/mcp.json:/app/code/mcp.json:ro
    ports:
      - "8005:8005"

volumes:
  llm_pgdata:
```

Then, next to the compose file, a `.env` with your secrets:

```env
# .env (NOT committed — this file is in .gitignore)
GEMINI_API_KEY=sk-...
# Anything you reference as ${...} inside mcp.json can also live here
GITHUB_TOKEN=ghp_...
```

The cleanest way to populate it is to start from the shipped template:

```bash
cp .env.example .env
$EDITOR .env
```

### Running it

```bash
# Bring up the stack
docker compose up -d --build

# Tail logs (look for the [MCP] banner block)
docker compose logs -f api

# Smoke test
curl -s http://localhost:8005/health
curl -s http://localhost:8005/tools | jq .

# Stop
docker compose down              # keep the database volume
docker compose down -v           # also nuke the database volume
```

### What each piece does

- **`db` service** — vanilla `postgres:16-alpine`. The `healthcheck` waits
  for `pg_isready` so the API doesn't race the DB on first boot. Data is
  kept in the named volume `llm_pgdata`, which survives `docker compose
  down` and is only wiped by `down -v`.
- **`api` service** — built from `docker/Dockerfile`. `depends_on: db:
  condition: service_healthy` guarantees the DB is accepting connections
  before the API starts.
- **`DATABASE_URL`** uses the Compose-internal hostname `db` (not
  `localhost`) so the API container can resolve the DB container.
- **`mcp.json` volume mount** lets you edit your MCP config locally and
  just `docker compose restart api` — no rebuild needed. The `:ro` flag
  prevents the container from mutating the file.
- **`${GEMINI_API_KEY:?...}`** syntax fails fast at compose time if the
  env var is missing, instead of letting the container crash later.

### Tips

- Run `docker compose exec db psql -U llm -d llm` to inspect the
  `user_chat_history` table directly.
- To run multiple API replicas behind a load balancer, just bump the
  `WORKERS` env var and put nginx / Caddy / Traefik in front — the app
  is stateless except for the DB.
- For production, switch the `db` port mapping to internal-only and
  front the API with TLS.

---

## API reference

The whole service is documented interactively at **`/docs`** (Swagger UI)
and **`/redoc`** once it's running. Below is the short version.

All endpoints return JSON. Errors come back as `{"detail": "..."}` (FastAPI
default) with a suitable 4xx/5xx code.

### `GET /health`

Liveness probe. Useful for Kubernetes / Docker healthchecks.

```json
{
  "status": "ok",
  "window_size": 20,
  "users_cached": 0
}
```

### `GET /tools`

Lists every tool available to the LLM, in the format the frontend expects.

```json
{
  "tools": [
    { "name": "read_file",            "description": "Read a file from the filesystem" },
    { "name": "github__create_issue", "description": "Create a GitHub issue" }
  ]
}
```

> The `name` field is the **Gemini-side** name. Pass exactly that string
> back to `/chat` in the `tools` array if you want to filter the toolset
> for a given turn. Cross-server name collisions are auto-prefixed with
> `<server>__<tool>`.

### `POST /chat`

Run one LLM turn.

**Request body:**

```json
{
  "user":  "user-1234",                    // required, non-empty
  "query": "Summarise the README",         // required, non-empty
  "tools": ["read_file", "github__list_repos"]   // optional
}
```

- `user` — opaque user identifier. The API keeps a rolling window of past
  messages per user; pick something stable (e.g. a hash of the user ID
  on your side, never the raw email).
- `query` — the user prompt.
- `tools` — optional allow-list. `null` / omitted = all tools available.
  `[]` = no tools (pure chat). Any other list = restrict to those names.
  Unknown names are logged and silently dropped.

**Response body:**

```json
{
  "user": "user-1234",
  "response": "The README describes a FastAPI service that ..."
}
```

**Example:**

```bash
curl -s -X POST http://localhost:8005/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "user": "demo",
    "query": "What files are in /tmp?",
    "tools": ["read_file", "list_directory"]
  }' | jq .
```

### `GET /docs` and `GET /redoc`

Auto-generated OpenAPI documentation. Edit `code/api.py` to customize
titles, descriptions, and tags.

---

## CLI reference

```bash
python code/main.py
```

- `You:` prompt, type your message, hit Enter.
- The assistant may call MCP tools internally; you'll see the same
  `[MCP] Calling: ...` log lines as the API.
- `exit` / `quit` / `Ctrl-D` / `Ctrl-C` to leave.
- The CLI does **not** persist anything to PostgreSQL — context lives
  only in the in-memory `Conversation` for the lifetime of the process.

---

## Database schema

Created automatically on first start by `ChatHistoryDB._init_schema`
(`CREATE TABLE IF NOT EXISTS`).

```sql
CREATE TABLE IF NOT EXISTS user_chat_history (
    id              BIGSERIAL PRIMARY KEY,
    user_id         VARCHAR(255) NOT NULL,
    role            VARCHAR(20)  NOT NULL
        CHECK (role IN ('user', 'assistant')),
    content         TEXT         NOT NULL,
    enabled_tools   JSONB,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_chat_history_user_id_id
ON user_chat_history (user_id, id DESC);
```

Key points:

- The table is **append-only**. Nothing in the app ever deletes rows.
- The rolling window is a *read* of the most recent `CHAT_WINDOW_SIZE`
  rows for that user, in chronological order. Older rows stay forever
  for audit / analytics.
- The `(user_id, id DESC)` index keeps that "last N for this user"
  query cheap.

---

## How the tool loop works

When Gemini decides to call a tool, this is what happens inside
`GeminiCore.get_response` (pseudocode):

```
for up to 20 rounds:
    response = gemini.generate(prompt, tools=filtered_tools)
    if no function_calls in response:
        return response.text
    for each function_call:
        result = await mcp_session.call_tool(name, args)
        append function_response part to contents
    loop again with the new contents
```

- Up to **20 tool rounds** per turn. Hitting the cap throws.
- A failed MCP call is converted into a structured error response and
  fed back to Gemini so the model can react, instead of the whole turn
  blowing up.
- If a single tool name is registered by more than one MCP server, the
  second one is renamed `<server>__<tool>` so Gemini's function namespace
  stays unique.

---

## Project layout

```
llm-knowledge-base/
├── .env.example             # every env var the app reads, with examples
├── .github/workflows/
│   └── build.yml            # builds & pushes the Docker image to GHCR
├── docker/
│   └── Dockerfile           # python:3.14-slim, port 8005, uvicorn CMD
├── code/
│   ├── agent.py             # Agent + AgentManager (async)
│   ├── api.py               # FastAPI app, lifespan, routes
│   ├── context.py           # Conversation, MemoryStore, ContextBuilder
│   ├── db.py                # PostgreSQL pool + rolling window
│   ├── gemini.py            # GeminiCore: Gemini + MCP
│   ├── main.py              # CLI REPL
│   ├── mcp.json             # YOU create this (gitignored)
│   └── serve.py             # uvicorn entry point
├── requirements.txt         # Python deps
├── LICENSE                  # MIT
└── README.md                # you are here
```

---

## Troubleshooting

**`RuntimeError: GEMINI_API_KEY environment variable is not set.`**
The API/CLI can't start without a Gemini key. Copy `.env.example` to `.env`
and fill it in, or set it in your shell / Compose environment.

**`RuntimeError: PostgreSQL connection string is not set.`**
You tried to start the API in HTTP mode without `DATABASE_URL` (or one of
its aliases). Either set it or run the CLI mode instead.

**`FileNotFoundError: MCP config not found: mcp.json`**
`mcp.json` is git-ignored — you have to create it. See
[Preparing `mcp.json`](#preparing-mcpjson).

**`[MCP] FAILED: <name>` followed by `Could not find MCP executable '<cmd>'`**
For stdio servers, the `command` must be on `PATH` *inside the container
or venv you're running in*. For `npx` commands this means Node.js must
be installed; for `uvx` it means `uv`; etc.

**`ModuleNotFoundError: No module named 'psycopg2'`**
The database layer in `code/db.py` uses `psycopg2`, which is **not**
listed in `requirements.txt` (the file is also accidentally encoded as
UTF-16). Add `psycopg2-binary` to your environment to fix it:

```bash
pip install psycopg2-binary
```

**Tools are duplicated with a `<server>__` prefix in `/tools`**
Expected behaviour: two MCP servers exposed a tool with the same name,
so the second one got prefixed to keep Gemini's namespace unique. Either
disable the duplicate on one server or accept the prefix and pass it back
verbatim in `tools`.

**CORS errors from a browser frontend**
The API uses `CORSMiddleware` with permissive defaults for dev.
Tighten `allow_origins` in `code/api.py` for production.

**Container starts but the API never responds on `:8005`**
Check `docker compose logs api`. The most common cause is the
`GeminiCore` failing to connect to one MCP server — the loop logs the
failure and continues, but if the *only* server fails the model has no
tools. Use `docker compose exec api sh` to inspect `/app/code/mcp.json`.

**Uvicorn reload doesn't pick up changes**
`UVICORN_RELOAD=1` watches for `.py` file changes. It does **not**
reload on `mcp.json` edits — restart the container (or `docker compose
restart api`) for those to take effect.

---

## License

MIT — see [LICENSE](./LICENSE).
