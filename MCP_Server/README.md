# MCP Server 示例

这个目录提供一个最小可运行的 MCP Server（Python + FastMCP）。

## 目录

- `server.py`: MCP 服务入口，包含示例工具
- `requirements.txt`: 依赖

## 安装

在项目根目录执行：

```powershell
python -m pip install -r MCP_Server/requirements.txt
```

## 运行

```powershell
python MCP_Server/server.py
```

服务通过 `stdio` 提供 MCP 协议。

## 示例工具

- `echo(text: str) -> str`
- `read_memory_lines(prompt_file: str = "system_prompt.txt") -> list[str]`
- `manage_memories(api_key: str, prompt_file: str = "system_prompt.txt", model: str = "deepseek-chat", dry_run: bool = false) -> dict`
- `web_search(query: str, max_results: int = 5) -> list[dict]`
  - 优先使用 Tavily（传入 `api_key` 或环境变量 `TAVILY_API_KEY`）
  - 若 Tavily 不可用，自动回退 DuckDuckGo

## 对接到客户端（示例）

如果你的客户端支持 MCP 配置，通常可以添加一个 `stdio` server：

```json
{
  "mcpServers": {
    "theresia-demo": {
      "command": "python",
      "args": ["MCP_Server/server.py"]
    }
  }
}
```

## Tavily 使用

设置环境变量后启动主程序或 MCP server：

```powershell
$env:TAVILY_API_KEY="你的_tavily_key"
```
