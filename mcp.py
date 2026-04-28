"""
mcp.py
Proper MCP (Model Context Protocol) integration for the Backlog Quality Gate.

Implements the MCP protocol correctly:
  - Tool schema definitions (what the server exposes)
  - Tool call dispatch (client sends structured tool_call)
  - Structured tool responses  (server returns structured result)
  - Full audit log of every operation

In production: swap MCPTransport.BASE_URL for a real Jira/Linear MCP endpoint.
The MCPClient interface stays identical.
"""

import json
import copy
import datetime
import pathlib
from typing import Any

# ---------------------------------------------------------------------------
# MCP Transport layer  (mock HTTP — swap for real requests in production)
# ---------------------------------------------------------------------------

class MCPTransport:
    """
    Simulates the HTTP transport layer of an MCP server.
    In production this would be:
        requests.post(f"{BASE_URL}/tool_call", json=payload, headers=auth_headers)
    """
    BASE_URL = "mock://backlog-server/v1"

    def __init__(self):
        # In-memory backlog — the "database" on the MCP server side
        self._store: dict[str, dict] = {}

    def send(self, tool_name: str, arguments: dict) -> dict:
        """Dispatch a tool call and return a structured MCP tool response."""
        handlers = {
            "patch_story":      self._handle_patch_story,
            "create_story":     self._handle_create_story,
            "mark_decomposed":  self._handle_mark_decomposed,
            "get_story":        self._handle_get_story,
            "list_stories":     self._handle_list_stories,
            "load_backlog":     self._handle_load_backlog,
        }
        if tool_name not in handlers:
            return self._error(f"Unknown tool: {tool_name}")
        try:
            return handlers[tool_name](arguments)
        except Exception as exc:
            return self._error(str(exc))

    # -- Handlers ------------------------------------------------------------

    def _handle_load_backlog(self, args: dict) -> dict:
        stories = args.get("stories", [])
        self._store = {s["id"]: copy.deepcopy(s) for s in stories}
        return {"ok": True, "loaded": len(stories)}

    def _handle_patch_story(self, args: dict) -> dict:
        sid   = args["story_id"]
        field = args["field"]
        value = args["value"]
        if sid not in self._store:
            return self._error(f"Story {sid} not found")
        old = self._store[sid].get(field)
        self._store[sid][field] = value
        return {"ok": True, "story_id": sid, "field": field,
                "old_value": old, "new_value": value}

    def _handle_create_story(self, args: dict) -> dict:
        story = args["story"]
        sid   = story["id"]
        if sid in self._store:
            return self._error(f"Story {sid} already exists")
        self._store[sid] = copy.deepcopy(story)
        self._store[sid]["_new"] = True
        return {"ok": True, "created": sid, "story": self._store[sid]}

    def _handle_mark_decomposed(self, args: dict) -> dict:
        sid = args["story_id"]
        if sid not in self._store:
            return self._error(f"Story {sid} not found")
        self._store[sid]["_decomposed"] = True
        return {"ok": True, "story_id": sid, "status": "decomposed"}

    def _handle_get_story(self, args: dict) -> dict:
        sid = args["story_id"]
        if sid not in self._store:
            return self._error(f"Story {sid} not found")
        return {"ok": True, "story": copy.deepcopy(self._store[sid])}

    def _handle_list_stories(self, _args: dict) -> dict:
        return {"ok": True, "stories": list(self._store.values()),
                "count": len(self._store)}

    @staticmethod
    def _error(msg: str) -> dict:
        return {"ok": False, "error": msg}


# ---------------------------------------------------------------------------
# MCP Tool Schema Registry
# ---------------------------------------------------------------------------

MCP_TOOL_SCHEMAS = {
    "patch_story": {
        "name":        "patch_story",
        "description": "Update a single field on an existing backlog story",
        "inputSchema": {
            "type": "object",
            "properties": {
                "story_id": {"type": "string",  "description": "Story ID (e.g. US-001)"},
                "field":    {"type": "string",  "description": "Field to update (acceptance_criteria, points, title, epic)"},
                "value":    {                   "description": "New value for the field"},
            },
            "required": ["story_id", "field", "value"],
        },
    },
    "create_story": {
        "name":        "create_story",
        "description": "Create a new story in the backlog (used for child stories from decomposition)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "story": {
                    "type": "object",
                    "description": "Full story object to create",
                    "properties": {
                        "id":                  {"type": "string"},
                        "title":               {"type": "string"},
                        "points":              {"type": "integer"},
                        "acceptance_criteria": {"type": "string"},
                        "epic":                {"type": "string"},
                        "parent":              {"type": "string"},
                    },
                    "required": ["id", "title", "points"],
                }
            },
            "required": ["story"],
        },
    },
    "mark_decomposed": {
        "name":        "mark_decomposed",
        "description": "Mark a parent story as decomposed after child stories are created",
        "inputSchema": {
            "type": "object",
            "properties": {
                "story_id": {"type": "string", "description": "Parent story ID to mark as decomposed"},
            },
            "required": ["story_id"],
        },
    },
    "get_story": {
        "name":        "get_story",
        "description": "Fetch a single story by ID",
        "inputSchema": {
            "type": "object",
            "properties": {
                "story_id": {"type": "string"},
            },
            "required": ["story_id"],
        },
    },
    "list_stories": {
        "name":        "list_stories",
        "description": "List all stories currently in the backlog",
        "inputSchema": {"type": "object", "properties": {}},
    },
}


# ---------------------------------------------------------------------------
# MCP Client  (what app.py talks to)
# ---------------------------------------------------------------------------

class MCPClient:
    """
    Proper MCP client that:
      1. Sends structured tool_call requests to the transport
      2. Receives structured tool_response results
      3. Logs every request + response for full auditability
    """

    def __init__(self):
        self._transport  = MCPTransport()
        self._log: list[dict] = []
        self._initialized = False

    # -- Lifecycle -----------------------------------------------------------

    def initialize(self, backlog: list[dict]) -> str:
        """Open MCP session and load the initial backlog into the server."""
        resp = self._tool_call("load_backlog", {"stories": backlog})
        msg  = (f"MCP session initialized — {MCPTransport.BASE_URL}  "
                f"({resp.get('loaded', 0)} stories loaded)")
        self._log_info(msg)
        self._initialized = True
        return msg

    # -- Public tool calls ---------------------------------------------------

    def patch_story(self, story_id: str, field: str,
                    old_value: Any, new_value: Any) -> dict:
        """PATCH a single field on a story."""
        self._require_init()
        resp = self._tool_call("patch_story", {
            "story_id": story_id,
            "field":    field,
            "value":    new_value,
        })
        if not resp.get("ok"):
            self._log_error(f"patch_story failed: {resp.get('error')}")
        return resp

    def create_story(self, story: dict) -> dict:
        """POST a new story to the backlog."""
        self._require_init()
        resp = self._tool_call("create_story", {"story": story})
        if not resp.get("ok"):
            self._log_error(f"create_story failed: {resp.get('error')}")
        return resp

    def mark_decomposed(self, story_id: str) -> dict:
        """Mark parent story as decomposed."""
        self._require_init()
        return self._tool_call("mark_decomposed", {"story_id": story_id})

    def get_story(self, story_id: str) -> dict:
        """GET a single story."""
        self._require_init()
        return self._tool_call("get_story", {"story_id": story_id})

    def list_stories(self) -> list[dict]:
        """GET all stories."""
        self._require_init()
        resp = self._tool_call("list_stories", {})
        return resp.get("stories", [])

    # -- Bulk apply ----------------------------------------------------------

    def apply_approved_fixes(self, approved_stories: list[dict]) -> dict:
        """
        Apply all approved fixes via proper MCP tool calls.
        Each fix is a separate tool_call → tool_response cycle.
        """
        self._require_init()
        updates = 0
        creates = 0

        count = len(approved_stories)
        self._log_info(f"Processing {count} approved update{'s' if count != 1 else ''}...")

        for story in approved_stories:
            sid   = story["id"]
            fixes = story.get("_fixes", {})

            # Field patches — one tool_call per field
            for field, change in fixes.get("field_fixes", {}).items():
                resp = self.patch_story(sid, field, change["before"], change["after"])
                if resp.get("ok"):
                    updates += 1

            # Child story creates — one tool_call per child
            for child in fixes.get("children", []):
                resp = self.create_story(child)
                if resp.get("ok"):
                    creates += 1

            # Mark parent as decomposed
            if fixes.get("children"):
                self.mark_decomposed(sid)

        summary = (f"Done. {updates} field update{'s' if updates != 1 else ''}, "
                   f"{creates} child stor{'ies' if creates != 1 else 'y'} created.")
        self._log_info(summary)
        return {"updates": updates, "creates": creates, "message": summary}

    # -- Log -----------------------------------------------------------------

    def get_log(self) -> list[dict]:
        return list(self._log)

    def get_log_lines(self) -> list[str]:
        lines = []
        for e in self._log:
            lines.append(f"[{e['ts']}] [{e['level']}] {e['message']}")
        return lines

    def clear_log(self):
        self._log.clear()

    # -- Export --------------------------------------------------------------

    def export_backlog(self) -> list[dict]:
        """Return clean backlog from MCP server (strips internal _ fields except _new)."""
        stories = self.list_stories()
        result  = []
        for s in stories:
            clean = {k: v for k, v in s.items()
                     if not k.startswith("_") or k == "_new"}
            result.append(clean)
        return result

    def save_log(self, path: pathlib.Path = pathlib.Path("mcp_session_log.json")):
        path.write_text(json.dumps(self._log, indent=2, default=str))

    # -- Internals -----------------------------------------------------------

    def _tool_call(self, tool_name: str, arguments: dict) -> dict:
        """
        Send a structured MCP tool call and log the request + response.
        This is the core MCP protocol interaction.
        """
        schema = MCP_TOOL_SCHEMAS.get(tool_name, {})

        # Build the MCP tool_call payload (as per MCP spec)
        tool_call_payload = {
            "type":      "tool_call",
            "tool_name": tool_name,
            "arguments": arguments,
        }

        # Dispatch to transport
        response = self._transport.send(tool_name, arguments)

        # Build the MCP tool_response
        tool_response = {
            "type":      "tool_response",
            "tool_name": tool_name,
            "result":    response,
        }

        # Log at the appropriate level with a human-readable message
        level, msg = self._format_log_entry(tool_name, arguments, response)
        self._log.append({
            "ts":            self._ts(),
            "level":         level,
            "message":       msg,
            "tool_call":     tool_call_payload,
            "tool_response": tool_response,
        })

        return response

    @staticmethod
    def _format_log_entry(tool_name: str, args: dict, resp: dict) -> tuple[str, str]:
        """Build a human-readable log line for a tool call."""
        if not resp.get("ok"):
            return "ERROR", f"{tool_name} → ERROR: {resp.get('error', 'unknown')}"

        if tool_name == "patch_story":
            old = str(args.get("value", ""))
            old_v = str(resp.get("old_value", ""))
            new_v = str(resp.get("new_value", ""))
            trunc = lambda s: s[:30] + "..." if len(s) > 30 else s
            return "UPDATE", (
                f"PATCH backlog/{args['story_id']} · field:{args['field']}  "
                f"\"{trunc(old_v)}\" → \"{trunc(new_v)}\""
            )

        if tool_name == "create_story":
            story = args.get("story", {})
            return "CREATE", (
                f"POST backlog/stories · id:{story.get('id')}  "
                f"\"{story.get('title', '')[:40]}\" ({story.get('points', '?')}pts)"
            )

        if tool_name == "mark_decomposed":
            return "UPDATE", f"PATCH backlog/{args['story_id']} · status:decomposed"

        if tool_name == "load_backlog":
            return "INFO", f"Loaded {resp.get('loaded', 0)} stories into MCP server"

        if tool_name == "list_stories":
            return "INFO", f"GET backlog → {resp.get('count', 0)} stories"

        if tool_name == "get_story":
            return "INFO", f"GET backlog/{args.get('story_id')} → found"

        return "INFO", f"{tool_name} → ok"

    def _log_info(self, message: str):
        self._log.append({"ts": self._ts(), "level": "INFO", "message": message})

    def _log_error(self, message: str):
        self._log.append({"ts": self._ts(), "level": "ERROR", "message": message})

    def _require_init(self):
        if not self._initialized:
            raise RuntimeError("MCPClient not initialized. Call initialize(backlog) first.")

    @staticmethod
    def _ts() -> str:
        return datetime.datetime.now().strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_session(backlog: list[dict]) -> MCPClient:
    """Create and initialize an MCPClient with the given backlog."""
    client = MCPClient()
    client.initialize(backlog)
    return client


def get_tool_schemas() -> list[dict]:
    """Return all registered MCP tool schemas (for display in UI)."""
    return list(MCP_TOOL_SCHEMAS.values())


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pathlib as pl
    stories = json.loads(pl.Path("sample_backlog.json").read_text())
    client  = create_session(stories)

    client.patch_story("US-001", "acceptance_criteria", "",
                       "Given a user, when they log in with valid credentials, then they are redirected to the dashboard.")
    client.create_story({
        "id": "US-002-A", "title": "Edit profile — basic info", "points": 3,
        "acceptance_criteria": "Given logged in user, when they save, then changes persist.",
        "epic": "Profile", "parent": "US-002",
    })

    for line in client.get_log_lines():
        print(line)

    print("\n--- Tool Schemas ---")
    for schema in get_tool_schemas():
        print(f"  {schema['name']}: {schema['description']}")