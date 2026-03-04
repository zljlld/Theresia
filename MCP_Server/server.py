from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from MCP_Server import tools_impl


mcp = FastMCP("theresia-mcp-demo")


@mcp.tool()
def echo(text: str) -> str:
    """Return the input text directly."""
    return tools_impl.echo(text)


@mcp.tool()
def read_memory_lines(prompt_file: str = "system_prompt.txt") -> list[str]:
    """
    Read memory lines from system_prompt section before ===END_INITIAL_PROMPT===.
    A memory line is identified by starting with '<'.
    """
    return tools_impl.read_memory_lines(prompt_file)


@mcp.tool()
def manage_memories(
    api_key: str,
    prompt_file: str = "system_prompt.txt",
    model: str = "deepseek-chat",
    dry_run: bool = False,
) -> dict:
    """
    Run memory management using prompts in system_prompt.txt.
    Returns summary dict: status/answer/changed_count/written.
    """
    return tools_impl.manage_memories_tool(api_key, prompt_file, model, dry_run)


@mcp.tool()
def web_search(query: str, max_results: int = 5, api_key: str = "") -> list[dict]:
    """
    Search the web and return top results.
    Each result includes title/url/snippet.
    """
    return tools_impl.web_search(query, max_results, api_key=api_key)


if __name__ == "__main__":
    # Stdio transport is the most common for local MCP integration.
    mcp.run(transport="stdio")
