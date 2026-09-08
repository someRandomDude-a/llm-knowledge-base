# gemini_core.py

import json
import logging
import os
import re
import shutil
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Optional

import httpx2
from google import genai
from google.genai import types
from google.genai.types import FunctionDeclaration, Tool

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client


logger = logging.getLogger(__name__)


class GeminiCore:
    """
    Part 1 of the agent.

    Responsibilities:
        - Gemini API
        - MCP connections
        - MCP sessions
        - MCP tool discovery
        - MCP -> Gemini tool conversion
        - MCP tool execution
        - Gemini responses

    Does NOT handle:
        - conversation history
        - memories
        - long-term context
        - persistence
        - UI
    """

    def __init__(
        self,
        mcp_config_path: str = "mcp.json",
        model: str = "gemini-3.6-flash",
    ):
        self.model = model
        self.mcp_config_path = Path(mcp_config_path)

        # --------------------------------------------------------------
        # Gemini
        # --------------------------------------------------------------
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
        self.client = genai.Client(api_key=api_key)

        # --------------------------------------------------------------
        # MCP
        # --------------------------------------------------------------
        self.exit_stack = AsyncExitStack()
        self.sessions: dict[str, ClientSession] = {}
        self.gemini_tools: list[Tool] = []   # list of Tool objects
        self.tool_servers: dict[str, str] = {}   # gemini_name -> server_name
        self.mcp_tools: dict[str, Any] = {}      # gemini_name -> MCP tool
        self.started = False

    # ==================================================================
    # Environment expansion
    # ==================================================================

    @staticmethod
    def _expand_env(value: Any) -> Any:
        if isinstance(value, str):
            pattern = r"\$\{([^}]+)\}"
            def repl(match: re.Match[str]) -> str:
                return os.getenv(match.group(1), "")
            return re.sub(pattern, repl, value)
        if isinstance(value, list):
            return [GeminiCore._expand_env(item) for item in value]
        if isinstance(value, dict):
            return {key: GeminiCore._expand_env(item) for key, item in value.items()}
        return value

    # ==================================================================
    # Configuration
    # ==================================================================

    def _load_mcp_config(self) -> dict[str, Any]:
        if not self.mcp_config_path.exists():
            raise FileNotFoundError(f"MCP config not found: {self.mcp_config_path}")
        with self.mcp_config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        return self._expand_env(config)

    # ==================================================================
    # Start
    # ==================================================================

    async def start(self) -> None:
        if self.started:
            return

        config = self._load_mcp_config()
        servers = config.get("mcpServers", {})
        for name, server_config in servers.items():
            try:
                await self._connect_server(str(name), server_config)
            except Exception as exc:
                logger.error("[MCP] FAILED: %s - %s: %s", name, type(exc).__name__, exc)

        self.started = True
        logger.info("[MCP] Connected servers: %d", len(self.sessions))
        logger.info("[MCP] Available tools: %d", len(self.gemini_tools))

    # ==================================================================
    # MCP connection dispatcher
    # ==================================================================

    async def _connect_server(self, name: str, config: dict[str, Any]) -> None:
        if "command" in config:
            await self._connect_stdio_server(name, config)
        elif "url" in config:
            await self._connect_http_server(name, config)
        else:
            raise ValueError(f"MCP server '{name}' must specify either 'command' or 'url'.")

    # ==================================================================
    # STDIO MCP
    # ==================================================================

    async def _connect_stdio_server(self, name: str, config: dict[str, Any]) -> None:
        command = config.get("command")
        if not isinstance(command, str):
            raise ValueError(f"Invalid command for MCP server '{name}'.")
        args = [str(arg) for arg in config.get("args", [])]
        configured_env = {str(k): str(v) for k, v in config.get("env", {}).items()}
        process_env = os.environ.copy()
        process_env.update(configured_env)

        executable = shutil.which(command) or shutil.which(f"{command}.cmd")
        if executable is None:
            raise RuntimeError(f"Could not find MCP executable '{command}'.")

        params = StdioServerParameters(
            command=executable,
            args=args,
            env=process_env,
        )

        logger.info("[MCP] Starting: %s", name)
        read, write = await self.exit_stack.enter_async_context(stdio_client(params))
        session = await self.exit_stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self.sessions[name] = session
        logger.info("[MCP] Connected: %s", name)
        await self._register_tools(name, session)

    # ==================================================================
    # HTTP MCP
    # ==================================================================

    async def _connect_http_server(self, name: str, config: dict[str, Any]) -> None:
        url = config.get("url")
        if not isinstance(url, str):
            raise ValueError(f"Invalid URL for MCP server '{name}'.")
        headers = {str(k): str(v) for k, v in config.get("headers", {}).items()}

        # Timeouts from environment (seconds)
        timeout_float = float(os.getenv("MCP_HTTP_TIMEOUT", "60.0"))
        read_timeout = float(os.getenv("MCP_HTTP_READ_TIMEOUT", "300.0"))
        timeout = httpx2.Timeout(timeout_float, read=read_timeout)

        logger.info("[MCP] Connecting HTTP: %s (url=%s)", name, url)
        http_client = httpx2.AsyncClient(
            headers=headers,
            follow_redirects=True,
            timeout=timeout,
        )
        await self.exit_stack.enter_async_context(http_client)

        read_stream, write_stream = await self.exit_stack.enter_async_context(
            streamable_http_client(url, http_client=http_client)
        )
        session = await self.exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
        await session.initialize()
        self.sessions[name] = session
        logger.info("[MCP] Connected: %s", name)
        await self._register_tools(name, session)

    # ==================================================================
    # Schema sanitization (removes fields unsupported by Gemini)
    # ==================================================================

    def _sanitize_schema(self, schema: dict) -> dict:
        """
        Recursively clean a JSON Schema dict to only include properties
        that Gemini's API supports.

        Gemini's API supports the following JSON Schema properties:
        - type, format, description, nullable, enum
        - maxItems, minItems, properties, required, propertyOrdering, items
        - anyOf, oneOf, allOf (combiners are supported)

        Reference: https://github.com/promptfoo/promptfoo/issues/6902
        """
        if not isinstance(schema, dict):
            return schema

        # Only these keys are allowed by Gemini
        supported_keys = {
            "type", "format", "description", "nullable", "enum",
            "maxItems", "minItems", "properties", "required",
            "propertyOrdering", "items", "anyOf", "oneOf", "allOf"
        }

        cleaned = {}
        for key, value in schema.items():
            if key not in supported_keys:
                continue

            if isinstance(value, dict):
                cleaned[key] = self._sanitize_schema(value)
            elif isinstance(value, list):
                cleaned[key] = [
                    self._sanitize_schema(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                cleaned[key] = value

        # Post-process: filter required to match properties
        # This prevents "property is not defined" errors.
        if "required" in cleaned and isinstance(cleaned["required"], list):
            properties = cleaned.get("properties", {})
            if not isinstance(properties, dict):
                properties = {}
            # Keep only required fields that exist in properties
            filtered_required = [req for req in cleaned["required"] if req in properties]
            if filtered_required:
                cleaned["required"] = filtered_required
            else:
                del cleaned["required"]  # remove if empty

        # Gemini expects a single type, not a list
        if "type" in cleaned and isinstance(cleaned["type"], list):
            cleaned["type"] = cleaned["type"][0] if cleaned["type"] else "string"

        return cleaned

    # ==================================================================
    # Tool registration
    # ==================================================================

    async def _register_tools(self, server_name: str, session: ClientSession) -> None:
        result = await session.list_tools()
        tools = result.tools
        logger.info("[MCP] %s: %d tools", server_name, len(tools))

        # Keep track of already registered Gemini names to avoid collisions
        used_names = set(self.tool_servers.keys())

        for tool in tools:
            tool_name = str(tool.name)
            gemini_name = tool_name

            # If collision, prefix with server name and, if still taken, add a counter
            if gemini_name in used_names:
                base = f"{server_name}__{tool_name}"
                gemini_name = base
                counter = 1
                while gemini_name in used_names:
                    gemini_name = f"{base}_{counter}"
                    counter += 1

            used_names.add(gemini_name)

            input_schema = getattr(tool, "inputSchema", None) or getattr(tool, "input_schema", {})
            if not isinstance(input_schema, dict):
                input_schema = {}

            # Sanitize the schema to remove fields Gemini doesn't accept
            sanitized_schema = self._sanitize_schema(input_schema)

            # Build proper FunctionDeclaration
            declaration = FunctionDeclaration(
                name=gemini_name,
                description=tool.description or f"MCP tool: {tool_name}",
                parameters=sanitized_schema,  # type: ignore[arg-type]
            )

            gemini_tool = Tool(function_declarations=[declaration])
            self.gemini_tools.append(gemini_tool)

            self.tool_servers[gemini_name] = server_name
            self.mcp_tools[gemini_name] = tool

            logger.debug("    - %s", gemini_name)

    # ==================================================================
    # Execute MCP tool
    # ==================================================================

    async def _call_mcp_tool(self, gemini_tool_name: str, arguments: dict[str, Any]) -> Any:
        if gemini_tool_name not in self.tool_servers:
            raise RuntimeError(f"Unknown MCP tool: {gemini_tool_name}")

        server_name = self.tool_servers[gemini_tool_name]
        session = self.sessions[server_name]
        mcp_tool = self.mcp_tools[gemini_tool_name]
        actual_tool_name = str(mcp_tool.name)

        logger.info("[MCP] Calling: %s.%s", server_name, actual_tool_name)
        logger.debug("[MCP] Arguments: %s", arguments)

        result = await session.call_tool(actual_tool_name, arguments)
        return self._serialize_mcp_result(result)

    # ==================================================================
    # Serialize MCP result
    # ==================================================================

    @staticmethod
    def _serialize_mcp_result(result: Any) -> Any:
        structured = getattr(result, "structuredContent", None)
        if structured is not None:
            return structured
        structured = getattr(result, "structured_content", None)
        if structured is not None:
            return structured

        content = getattr(result, "content", None)
        if content is None:
            return str(result)

        output = []
        for item in content:
            text = getattr(item, "text", None)
            if text is not None:
                output.append(text)
            else:
                output.append(str(item))
        return "\n".join(output)

    # ==================================================================
    # Gemini
    # ==================================================================

    async def get_response(
        self,
        prompt: str,
        tools: Optional[list[Tool]] = None,
    ) -> str:
        if not self.started:
            raise RuntimeError("GeminiCore has not been started.")

        tools_to_use = tools if tools is not None else self.gemini_tools

        contents: list[Any] = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)]
            )
        ]

        max_tool_rounds = 20
        for _ in range(max_tool_rounds):
            config = types.GenerateContentConfig(
                temperature=0,
                tools=tools_to_use,  # type: ignore
            )
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )

            function_calls = response.function_calls
            if not function_calls:
                return response.text or ""

            if response.candidates:
                model_content = response.candidates[0].content
                contents.append(model_content)

            function_response_parts = []
            for function_call in function_calls:
                name = function_call.name
                # Gemini should always provide a name; guard against None
                if name is None:
                    raise RuntimeError("Function call name is None")

                arguments = function_call.args or {}
                try:
                    result = await self._call_mcp_tool(name, arguments)
                except Exception as exc:
                    result = {"error": f"{type(exc).__name__}: {exc}"}
                function_response_parts.append(
                    types.Part.from_function_response(
                        name=name,
                        response={"result": result}
                    )
                )

            contents.append(
                types.Content(
                    role="user",
                    parts=function_response_parts,
                )
            )

        raise RuntimeError("Gemini exceeded the maximum number of MCP tool rounds.")

    # ==================================================================
    # Tool discovery (public)
    # ==================================================================

    def list_mcp_tools(self) -> list[dict[str, str]]:
        result = []
        for gemini_name, tool in self.mcp_tools.items():
            description = getattr(tool, "description", None) or f"MCP tool: {getattr(tool, 'name', gemini_name)}"
            result.append({"name": str(gemini_name), "description": str(description)})
        return result

    def get_mcp_status(self) -> list[dict[str, str]]:
        """Return status of connected MCP servers."""
        return [{"server": name, "status": "connected"} for name in self.sessions.keys()]

    # ==================================================================
    # Shutdown
    # ==================================================================

    async def close(self) -> None:
        try:
            await self.exit_stack.aclose()
        finally:
            self.sessions.clear()
            self.gemini_tools.clear()
            self.tool_servers.clear()
            self.mcp_tools.clear()
            await self.client.aio.aclose()
            self.started = False