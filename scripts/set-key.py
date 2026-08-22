#!/usr/bin/env python3
"""Set arcAgent.openRouterKey in VS Code's user settings.

Usage:  python3 scripts/set-key.py <api-key>

The key is read from argv and never stored anywhere but settings.json, so
this file is safe to commit. Reload the window afterwards
(Ctrl+Shift+P -> "Developer: Reload Window") for the extension to pick it up.
"""
import json
import pathlib
import re
import sys

if len(sys.argv) != 2 or not sys.argv[1].strip():
    sys.exit("usage: set-key.py <api-key>")

key = sys.argv[1].strip()

path = pathlib.Path.home() / ".config/Code/User/settings.json"
if not path.exists():
    sys.exit(f"no settings file at {path}")

text = path.read_text()

# settings.json is JSONC — VS Code allows comments and trailing commas, so a
# real settings file usually will not round-trip through `json`. Rewriting the
# whole document would strip every comment the user put there, so replace the
# one value in place instead and leave the rest of the file byte-for-byte.
entry = re.compile(r'("arcAgent\.openRouterKey"\s*:\s*")(?:[^"\\]|\\.)*(")')
escaped = key.replace("\\", "\\\\").replace('"', '\\"')

text, count = entry.subn(lambda m: m.group(1) + escaped + m.group(2), text)
if count == 0:
    # No entry to rewrite, and inserting one means parsing the JSONC we just
    # avoided parsing. The settings UI handles this case fine.
    sys.exit(f'no "arcAgent.openRouterKey" entry in {path}; add it once via Ctrl+, and rerun')
if count > 1:
    sys.exit(f"{path} has {count} arcAgent.openRouterKey entries; fix that by hand first")

path.write_text(text)
print(f"set arcAgent.openRouterKey ({key[:4]}…{key[-4:]}) in {path}")
print("now: Ctrl+Shift+P -> Developer: Reload Window")
