# gemini_core.py

import json
import os
import re
import shutil

from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Optional

import httpx2

from google import genai
from google.genai import types

from mcp import (
    ClientSession,
    StdioServerParameters,
)

from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client


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

        self.mcp_config_path = Path(
            mcp_config_path
        )

        # --------------------------------------------------------------
        # Gemini
        # --------------------------------------------------------------

        api_key = os.getenv(
            "GEMINI_API_KEY"
        )

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable "
                "is not set."
            )

        self.client = genai.Client(
            api_key=api_key
        )

        # --------------------------------------------------------------
        # MCP
        # --------------------------------------------------------------

        self.exit_stack = AsyncExitStack()

        self.sessions: dict[
            str,
            ClientSession,
        ] = {}

        # Gemini function declarations
        self.gemini_tools: list[
            types.Tool
        ] = []

        # Map Gemini function name -> MCP server
        self.tool_servers: dict[
            str,
            str,
        ] = {}

        # Map Gemini function name -> MCP tool
        self.mcp_tools: dict[
            str,
            Any,
        ] = {}

        self.started = False

    # ==================================================================
    # Environment expansion
    # ==================================================================

    @staticmethod
    def _expand_env(
        value: Any,
    ) -> Any:

        if isinstance(value, str):

            pattern = r"\$\{([^}]+)\}"

            def replace(
                match: re.Match[str],
            ) -> str:

                return os.getenv(
                    match.group(1),
                    "",
                )

            return re.sub(
                pattern,
                replace,
                value,
            )

        if isinstance(value, list):

            return [
                GeminiCore._expand_env(item)
                for item in value
            ]

        if isinstance(value, dict):

            return {
                key: GeminiCore._expand_env(item)
                for key, item in value.items()
            }

        return value

    # ==================================================================
    # Configuration
    # ==================================================================

    def _load_mcp_config(
        self,
    ) -> dict[str, Any]:

        if not self.mcp_config_path.exists():

            raise FileNotFoundError(
                f"MCP config not found: "
                f"{self.mcp_config_path}"
            )

        with self.mcp_config_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            config = json.load(file)

        return self._expand_env(
            config
        )

    # ==================================================================
    # Start
    # ==================================================================

    async def start(
        self,
    ) -> None:

        if self.started:
            return

        config = self._load_mcp_config()

        servers = config.get(
            "mcpServers",
            {},
        )

        for name, server_config in servers.items():

            try:

                await self._connect_server(
                    str(name),
                    server_config,
                )

            except Exception as exc:

                print(
                    f"[MCP] FAILED: {name}"
                )

                print(
                    f"[MCP] "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

        self.started = True

        print(
            f"[MCP] Connected servers: "
            f"{len(self.sessions)}"
        )

        print(
            f"[MCP] Available tools: "
            f"{len(self.gemini_tools)}"
        )

    # ==================================================================
    # MCP connection dispatcher
    # ==================================================================

    async def _connect_server(
        self,
        name: str,
        config: dict[str, Any],
    ) -> None:

        if "command" in config:

            await self._connect_stdio_server(
                name,
                config,
            )

            return

        if "url" in config:

            await self._connect_http_server(
                name,
                config,
            )

            return

        raise ValueError(
            f"MCP server '{name}' must specify "
            f"either 'command' or 'url'."
        )

    # ==================================================================
    # STDIO MCP
    # ==================================================================

    async def _connect_stdio_server(
        self,
        name: str,
        config: dict[str, Any],
    ) -> None:

        command = config.get(
            "command"
        )

        if not isinstance(
            command,
            str,
        ):

            raise ValueError(
                f"Invalid command for "
                f"MCP server '{name}'."
            )

        args = [
            str(arg)
            for arg in config.get(
                "args",
                [],
            )
        ]

        configured_env = {
            str(key): str(value)
            for key, value
            in config.get(
                "env",
                {},
            ).items()
        }

        process_env = os.environ.copy()

        process_env.update(
            configured_env
        )

        executable = shutil.which(
            command
        )

        if executable is None:

            executable = shutil.which(
                f"{command}.cmd"
            )

        if executable is None:

            raise RuntimeError(
                f"Could not find MCP executable "
                f"'{command}'."
            )

        params = StdioServerParameters(
            command=executable,
            args=args,
            env=process_env,
        )

        print(
            f"[MCP] Starting: {name}"
        )

        read, write = (
            await self.exit_stack.enter_async_context(
                stdio_client(params)
            )
        )

        session = (
            await self.exit_stack.enter_async_context(
                ClientSession(
                    read,
                    write,
                )
            )
        )

        await session.initialize()

        self.sessions[name] = session

        print(
            f"[MCP] Connected: {name}"
        )

        await self._register_tools(
            name,
            session,
        )

    # ==================================================================
    # HTTP MCP
    # ==================================================================

    async def _connect_http_server(
        self,
        name: str,
        config: dict[str, Any],
    ) -> None:

        url = config.get(
            "url"
        )

        if not isinstance(
            url,
            str,
        ):

            raise ValueError(
                f"Invalid URL for "
                f"MCP server '{name}'."
            )

        headers = {
            str(key): str(value)
            for key, value
            in config.get(
                "headers",
                {},
            ).items()
        }

        print(
            f"[MCP] Connecting HTTP: {name}"
        )

        print(
            f"[MCP] URL: {url}"
        )

        # --------------------------------------------------------------
        # IMPORTANT:
        # This MCP SDK version uses httpx2.
        # --------------------------------------------------------------

        http_client = httpx2.AsyncClient(
            headers=headers,
            follow_redirects=True,
            timeout=httpx2.Timeout(
                60.0,
                read=300.0,
            ),
        )

        await self.exit_stack.enter_async_context(
            http_client
        )

        # --------------------------------------------------------------
        # Streamable HTTP
        # --------------------------------------------------------------

        read_stream, write_stream = (
            await self.exit_stack.enter_async_context(
                streamable_http_client(
                    url,
                    http_client=http_client,
                )
            )
        )

        session = (
            await self.exit_stack.enter_async_context(
                ClientSession(
                    read_stream,
                    write_stream,
                )
            )
        )

        await session.initialize()

        self.sessions[name] = session

        print(
            f"[MCP] Connected: {name}"
        )

        await self._register_tools(
            name,
            session,
        )

    # ==================================================================
    # Tool registration
    # ==================================================================

    async def _register_tools(
        self,
        server_name: str,
        session: ClientSession,
    ) -> None:

        result = await session.list_tools()

        tools = result.tools

        print(
            f"[MCP] {server_name}: "
            f"{len(tools)} tools"
        )

        for tool in tools:

            tool_name = str(
                tool.name
            )

            # ----------------------------------------------------------
            # Avoid collisions between MCP servers.
            #
            # Gemini function names must be unique.
            # ----------------------------------------------------------

            gemini_name = tool_name

            if gemini_name in self.mcp_tools:

                gemini_name = (
                    f"{server_name}__"
                    f"{tool_name}"
                )

            # ----------------------------------------------------------
            # MCP schema
            # ----------------------------------------------------------

            input_schema = getattr(
                tool,
                "inputSchema",
                None,
            )

            if input_schema is None:

                input_schema = getattr(
                    tool,
                    "input_schema",
                    {},
                )

            if not isinstance(
                input_schema,
                dict,
            ):

                input_schema = {}

            # ----------------------------------------------------------
            # Gemini function declaration
            # ----------------------------------------------------------

            declaration = {
                "name": gemini_name,
                "description": (
                    tool.description
                    or f"MCP tool: {tool_name}"
                ),
                "parameters_json_schema": (
                    input_schema
                ),
            }

            gemini_tool = types.Tool(
                function_declarations=[
                    declaration #type: ignore
                ]
            )

            self.gemini_tools.append(
                gemini_tool
            )

            # ----------------------------------------------------------
            # Remember how to execute this function.
            # ----------------------------------------------------------

            self.tool_servers[
                gemini_name
            ] = server_name

            self.mcp_tools[
                gemini_name
            ] = tool

            print(
                f"    - {gemini_name}"
            )

    # ==================================================================
    # Execute MCP tool
    # ==================================================================

    async def _call_mcp_tool(
        self,
        gemini_tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:

        if gemini_tool_name not in self.tool_servers:

            raise RuntimeError(
                f"Unknown MCP tool: "
                f"{gemini_tool_name}"
            )

        server_name = self.tool_servers[
            gemini_tool_name
        ]

        session = self.sessions[
            server_name
        ]

        mcp_tool = self.mcp_tools[
            gemini_tool_name
        ]

        # --------------------------------------------------------------
        # The name exposed to Gemini may differ from the actual
        # MCP tool name because of collision handling.
        # --------------------------------------------------------------

        actual_tool_name = str(
            mcp_tool.name
        )

        print(
            f"[MCP] Calling: "
            f"{server_name}.{actual_tool_name}"
        )

        print(
            f"[MCP] Arguments: "
            f"{arguments}"
        )

        result = await session.call_tool(
            actual_tool_name,
            arguments,
        )

        return self._serialize_mcp_result(
            result
        )

    # ==================================================================
    # Serialize MCP result
    # ==================================================================

    @staticmethod
    def _serialize_mcp_result(
        result: Any,
    ) -> Any:

        # MCP CallToolResult generally has:
        #
        #     content
        #     structuredContent
        #
        # Prefer structured content when available.

        structured = getattr(
            result,
            "structuredContent",
            None,
        )

        if structured is not None:

            return structured

        structured = getattr(
            result,
            "structured_content",
            None,
        )

        if structured is not None:

            return structured

        content = getattr(
            result,
            "content",
            None,
        )

        if content is None:
            return str(result)

        output = []

        for item in content:

            text = getattr(
                item,
                "text",
                None,
            )

            if text is not None:

                output.append(
                    text
                )

                continue

            output.append(
                str(item)
            )

        return "\n".join(
            output
        )

    # ==================================================================
    # Gemini
    # ==================================================================

    async def get_response(
        self,
        prompt: str,
        tools: Optional[list[types.Tool]] = None,
    ) -> str:

        if not self.started:

            raise RuntimeError(
                "GeminiCore has not been started."
            )

        # --------------------------------------------------------------
        # If the caller passed a filtered tool list, use it.
        # Otherwise expose every registered MCP tool.
        # --------------------------------------------------------------

        tools_to_use = (
            tools if tools is not None else self.gemini_tools
        )

        # --------------------------------------------------------------
        # Conversation for THIS request.
        #
        # Part 2 will eventually provide the actual history.
        # --------------------------------------------------------------

        contents: list[Any] = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=prompt
                    )
                ],
            )
        ]

        # --------------------------------------------------------------
        # Tool loop
        #
        # We manually handle:
        #
        # Gemini
        #   -> function call
        #   -> MCP
        #   -> function response
        #   -> Gemini
        #   -> ...
        #
        # until Gemini produces normal text.
        # --------------------------------------------------------------

        max_tool_rounds = 20

        for _ in range(
            max_tool_rounds
        ):

            config = types.GenerateContentConfig(
                temperature=0,
                tools=tools_to_use, #type: ignore
            )

            response = (
                await self.client.aio.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
            )

            # ----------------------------------------------------------
            # Check for function calls
            # ----------------------------------------------------------

            function_calls = (
                response.function_calls
            )

            if not function_calls:

                return response.text or ""

            # ----------------------------------------------------------
            # Add Gemini's response to conversation
            # ----------------------------------------------------------

            if response.candidates:

                model_content = (
                    response.candidates[0].content
                )

                contents.append(
                    model_content
                )

            # ----------------------------------------------------------
            # Execute every requested tool
            # ----------------------------------------------------------

            function_response_parts = []

            for function_call in function_calls:

                name = function_call.name

                arguments = (
                    function_call.args
                    or {}
                )

                try:

                    result = (
                        await self._call_mcp_tool(
                            name, #type: ignore
                            arguments,
                        )
                    )

                except Exception as exc:

                    result = {
                        "error": (
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        )
                    }

                function_response_parts.append(
                    types.Part.from_function_response(
                        name=name, #type: ignore
                        response={
                            "result": result
                        },
                    )
                )

            # ----------------------------------------------------------
            # Give MCP results back to Gemini
            # ----------------------------------------------------------

            contents.append(
                types.Content(
                    role="user",
                    parts=function_response_parts,
                )
            )

        raise RuntimeError(
            "Gemini exceeded the maximum number "
            "of MCP tool rounds."
        )

    # ==================================================================
    # Tool discovery (public)
    # ==================================================================

    def list_mcp_tools(
        self,
    ) -> list[dict[str, str]]:
        """
        Public, JSON-friendly description of every available tool.

        Returns the *gemini-side* name (which is what callers
        pass back to /chat) and the MCP tool's description.
        """

        result: list[dict[str, str]] = []

        for (
            gemini_name,
            tool,
        ) in self.mcp_tools.items():

            description = (
                getattr(tool, "description", None)
                or f"MCP tool: {getattr(tool, 'name', gemini_name)}"
            )

            result.append(
                {
                    "name": str(gemini_name),
                    "description": str(description),
                }
            )

        return result

    # ==================================================================
    # Shutdown
    # ==================================================================

    async def close(
        self,
    ) -> None:

        try:

            await self.exit_stack.aclose()

        finally:

            self.sessions.clear()

            self.gemini_tools.clear()

            self.tool_servers.clear()

            self.mcp_tools.clear()

            await self.client.aio.aclose()

            self.started = False