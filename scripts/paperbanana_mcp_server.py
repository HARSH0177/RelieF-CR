"""
PaperBanana Official Model Context Protocol (MCP) Server.
Implements the JSON-RPC 2.0 Stdio Transport Protocol for Antigravity, Claude, and Gemini IDEs.
"""

import sys
import os
import json
import traceback
import subprocess

# Ensure UTF-8 stdio
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')


def send_response(response_dict):
    """Writes a single JSON-RPC line to stdout."""
    payload = json.dumps(response_dict, ensure_ascii=False)
    sys.stdout.write(payload + "\n")
    sys.stdout.flush()


def handle_initialize(msg_id, params):
    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {
                    "listChanged": False
                }
            },
            "serverInfo": {
                "name": "paperbanana-mcp-server",
                "version": "1.0.0"
            }
        }
    }


def handle_tools_list(msg_id):
    tools = [
        {
            "name": "paperbanana_generate_diagram",
            "description": "Generates a publication-grade academic methodology architecture diagram following NeurIPS/IEEE guidelines.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Text description or architecture specification of the methodology to illustrate."
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Target file path to save the generated figure (e.g. outputs/architecture.png)."
                    },
                    "venue": {
                        "type": "string",
                        "description": "Target venue style (e.g. ieee, neurips, cvpr, iclr).",
                        "default": "ieee"
                    }
                },
                "required": ["description"]
            }
        },
        {
            "name": "paperbanana_guidelines",
            "description": "Retrieves official PaperBanana 2025/2026 design principles, color palettes, and tensor typography rules.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Category of guidelines (methodology, plots, colors, typography).",
                        "default": "methodology"
                    }
                }
            }
        },
        {
            "name": "paperbanana_doctor",
            "description": "Inspects system health, installed PaperBanana dependencies, and API credentials.",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }
    ]
    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "result": {
            "tools": tools
        }
    }


def handle_tools_call(msg_id, params):
    name = params.get("name")
    arguments = params.get("arguments", {})

    try:
        if name == "paperbanana_guidelines":
            from paperbanana.guidelines.methodology import DEFAULT_METHODOLOGY_GUIDELINES
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": DEFAULT_METHODOLOGY_GUIDELINES
                        }
                    ]
                }
            }

        elif name == "paperbanana_doctor":
            res = subprocess.run([sys.executable, "-m", "paperbanana.cli", "doctor"], capture_output=True, text=True)
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": res.stdout or res.stderr or "PaperBanana doctor check passed."
                        }
                    ]
                }
            }

        elif name == "paperbanana_generate_diagram":
            desc = arguments.get("description", "")
            out_path = arguments.get("output_path", "outputs/generated_paperbanana_figure.png")
            
            # Run paperbanana CLI generate
            cmd = [sys.executable, "-m", "paperbanana.cli", "generate", "--description", desc, "--output", out_path]
            res = subprocess.run(cmd, capture_output=True, text=True)
            
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Generation output: {res.stdout}\n{res.stderr}\nTarget: {out_path}"
                        }
                    ]
                }
            }

        else:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32601,
                    "message": f"Unknown tool: {name}"
                }
            }

    except Exception as e:
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {
                "code": -32603,
                "message": f"Internal execution error: {str(e)}\n{traceback.format_exc()}"
            }
        }


def main():
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue

            msg = json.loads(line)
            method = msg.get("method")
            msg_id = msg.get("id")

            if method == "initialize":
                send_response(handle_initialize(msg_id, msg.get("params", {})))
            elif method == "notifications/initialized" or method == "initialized":
                # Notification, no response needed
                pass
            elif method == "tools/list":
                send_response(handle_tools_list(msg_id))
            elif method == "tools/call":
                send_response(handle_tools_call(msg_id, msg.get("params", {})))
            elif method == "ping":
                send_response({"jsonrpc": "2.0", "id": msg_id, "result": {}})
            else:
                if msg_id is not None:
                    send_response({
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "error": {
                            "code": -32601,
                            "message": f"Method not found: {method}"
                        }
                    })
        except Exception as e:
            # Never print raw unformatted exceptions to stdout, as it breaks JSON-RPC
            sys.stderr.write(f"MCP Server Error: {e}\n")
            sys.stderr.flush()


if __name__ == "__main__":
    main()
