from .base import Tool, ToolCall, ToolRegistry, ToolResult, parse_tool_call
from .calculator import CalculatorTool
from .filesystem import FilesystemTool
from .python import PythonTool


def default_tools(fs_root: str | None = None) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(PythonTool())
    registry.register(FilesystemTool(root=fs_root))
    return registry


__all__ = [
    "Tool",
    "ToolCall",
    "ToolResult",
    "ToolRegistry",
    "parse_tool_call",
    "CalculatorTool",
    "PythonTool",
    "FilesystemTool",
    "default_tools",
]
