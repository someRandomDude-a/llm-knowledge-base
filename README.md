# LLM Knowledge Base Agent

A Gemini-powered agent that connects to MCP servers and provides a persistent, per-user chat interface with long-term memory.

---

## Features

- Gemini API – Uses the latest google-genai SDK.
- MCP Integration – Connects to any MCP server over STDIO or HTTP (SSE).
- Persistent Chat History – PostgreSQL backend with rolling window (no deletions).
- Per-User Context – Each user has isolated conversation history and memories.
- Tool Filtering – Restrict tools per request via API payload.
- FastAPI Endpoints – REST API for chat, health, and tool listing.
- CLI Interface – Interactive terminal for quick testing.
- Docker Support – Ready for containerized deployment.

---

## Prerequisites

- Python 3.10+
- PostgreSQL (if using persistence)
- Gemini API key (Google AI Studio)
- MCP servers (e.g., markdown-vault-mcp)

---

## Installation

### Local Development

```
git clone <your-repo>
cd llm-knowledge-base
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Docker

```bash
docker build -t llm-knowledge-base .
docker run -p 8005:8005 --env-file .env llm-knowledge-base
```

Or use docker-compose.yml (example provided).

---

## Configuration

Create a .env file in the project root based on .env.example.
The following environment variables are required:

| Variable | Description |
|----------|-------------|
| GEMINI_API_KEY | Your Google Gemini API key |
| DATABASE_URL | PostgreSQL connection string (e.g. postgresql://user:pass@localhost:5432/dbname) |

Optional variables:

| Variable | Default | Description |
|----------|---------|-------------|
| CHAT_WINDOW_SIZE | 20 | Number of recent messages to load per user |
| LOG_LEVEL | INFO | Logging level (DEBUG, INFO, WARNING, etc.) |
| MCP_HTTP_TIMEOUT | 60.0 | HTTP MCP connection timeout (seconds) |
| MCP_HTTP_READ_TIMEOUT | 300.0 | HTTP MCP read timeout (seconds) |
| HOST | 0.0.0.0 | Bind address for the API server |
| PORT | 8005 | Port for the API server |
| WORKERS | 2 | Number of uvicorn workers (production) |
| UVICORN_RELOAD | 0 | Set to 1 to enable auto-reload (development) |

---

## MCP Configuration

Create a mcp.json file in the project root to define the MCP servers.
Example for a remote HTTP server:

```json
{
  "mcpServers": {
    "markdown-vault": {
      "url": "https://mcp.example.com/mcp/",
      "headers": {
        "Authorization": "Bearer ${MCP_TOKEN}"
      }
    }
  }
}
```

For local STDIO servers:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/dir"]
    }
  }
}
```

Environment variables in mcp.json are automatically expanded using ${VAR_NAME}.

---

## Running

### CLI (interactive chat)

```bash
python main.py
```

### API Server

```bash
python serve.py
```

Or with uvicorn directly:

```bash
uvicorn api:app --host 0.0.0.0 --port 8005 --workers 2
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET  | /health | Health check with MCP server status |
| GET  | /tools  | List all available MCP tools |
| POST | /chat   | Send a message and get a response |

POST /chat payload:

```json
{
  "user": "unique_user_id",
  "query": "Hello, what can you do?",
  "tools": ["tool1", "tool2"]   // optional, null = all, [] = none
}
```

Response:

```json
{
  "user": "unique_user_id",
  "response": "I can help you with ..."
}
```

---

## Database Schema

The PostgreSQL database uses two tables:

- user_chat_history – stores messages with user_id, role (user/assistant), content, enabled_tools (JSONB), and created_at.
- user_memories – stores per-user memory strings with a composite primary key (user_id, memory_text).

The CHAT_WINDOW_SIZE defines how many recent messages are loaded per request – no data is ever deleted.

---

## Project Structure

```bash
.
├── api.py              # FastAPI application
├── serve.py            # Uvicorn entry point
├── agent.py            # Agent and AgentManager classes
├── gemini_core.py      # Gemini and MCP core logic
├── db.py               # Async PostgreSQL persistence
├── context.py          # Conversation and prompt builder
├── main.py             # CLI interactive mode
├── mcp.json            # MCP server configuration (user-provided)
├── requirements.txt    # Python dependencies
├── .env.example        # Example environment variables
├── README.md           # This file
└── ...
```

---

## MCP Integration Details

- The agent connects to MCP servers during startup.
- Tools are discovered and converted to Gemini function declarations.
- The Gemini model can call these tools during a conversation.
- Results are passed back to Gemini for further reasoning.

All MCP errors are caught and returned to the user as tool-execution errors.

## License

MIT