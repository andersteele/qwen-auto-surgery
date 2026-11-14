"""Stands in for `pi --mode rpc`: echoes each command and plays a short run for every prompt."""
import json
import sys

out = sys.stdout
for line in sys.stdin:
    cmd = json.loads(line)
    out.write(json.dumps({"type": "echo", "cmd": cmd}) + "\n")
    if cmd["type"] == "prompt" and "streamingBehavior" not in cmd:
        out.write(json.dumps({"type": "response", "id": cmd["id"], "command": "prompt", "success": True}) + "\n")
        out.write(json.dumps({"type": "agent_start"}) + "\n")
        if cmd["message"] != "long":
            out.write(json.dumps({"type": "agent_settled", "aborted": False}) + "\n")
    elif cmd["type"] == "abort":
        out.write(json.dumps({"type": "message_end", "message": {"role": "assistant", "stopReason": "error",
                                                                 "errorMessage": "404 generation thread died"}}) + "\n")
        out.write(json.dumps({"type": "agent_settled", "aborted": True}) + "\n")
    elif cmd["type"] == "prompt":
        out.write(json.dumps({"type": "response", "id": cmd["id"], "command": "prompt", "success": False, "error": "nope"}) + "\n")
    out.flush()
