#!/usr/bin/env python3
"""Serve the hand-annotation page and persist every stroke to disk.

    python serve_handlabel.py                          # the 62-page set,  port 8777
    HANDLABEL_SET=handlabel2 python serve_handlabel.py  # the blind 100,      port 8778

The page on its own keeps work in the browser's localStorage, which is not a safe place to put an
hour of drawing: it is silently unavailable in a sandboxed frame, it is wiped by "clear browsing
data", and it does not follow you to another browser. With this server running, every edit is also
written to a real file, so the browser becomes a cache rather than the only copy.

  GET  /labels.json   the labels on disk, so a fresh browser recovers everything
  POST /save          a full label set; written atomically, previous version kept as a backup

Writes go to labels/hand_labels.json via a temp file and os.replace, so an interrupted write
cannot truncate the existing labels. Every save also rotates a timestamped copy into
labels/backups/, capped at the most recent 40.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Two sets are served from this one script, on different ports so both can be open at once:
#   handlabel   the 62 pages the pipeline had already been run on
#   handlabel2  a blind evaluation draw of 100 pages it has never seen, carrying no pipeline output
SETS = {
    "handlabel":  ("handlabel",  "hand_labels.json",       8777),
    "handlabel2": ("handlabel2", "hand_labels_blind.json", 8778),
}
_which = os.environ.get("HANDLABEL_SET", "handlabel")
if _which not in SETS:
    raise SystemExit(f"HANDLABEL_SET must be one of {list(SETS)}")
_dir, _file, _port = SETS[_which]
ROOT = Path("/data/braintree/scan-processed/wp7-panels-sam3-seg/geometry") / _dir
LABELS = HERE / "labels" / _file
BACKUPS = HERE / "labels" / "backups"
KEEP = 40
PORT = int(os.environ.get("HANDLABEL_PORT", str(_port)))


def write_atomic(payload: dict) -> None:
    LABELS.parent.mkdir(parents=True, exist_ok=True)
    BACKUPS.mkdir(parents=True, exist_ok=True)
    if LABELS.exists():
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        shutil.copy2(LABELS, BACKUPS / f"{LABELS.stem}-{stamp}.json")
        old = sorted(BACKUPS.glob(f"{LABELS.stem}-*.json"))
        for stale in old[:-KEEP]:
            stale.unlink()
    tmp = LABELS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, LABELS)                     # atomic on the same filesystem


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def log_message(self, fmt, *args):          # one line per save, not per image
        if "POST" in (args[0] if args else ""):
            sys.stderr.write("%s  %s\n" % (self.log_date_time_string(), fmt % args))

    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.split("?")[0] == "/labels.json":
            if LABELS.exists():
                try:
                    return self._json(200, json.loads(LABELS.read_text(encoding="utf-8")))
                except Exception as e:                            # noqa: BLE001
                    return self._json(200, {"labels": [], "error": str(e)})
            return self._json(200, {"labels": []})
        return super().do_GET()

    def do_POST(self):
        if self.path.split("?")[0] != "/save":
            return self._json(404, {"ok": False})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(n) or b"{}")
            if not isinstance(payload, dict) or "labels" not in payload:
                raise ValueError("expected an object with a 'labels' list")
            write_atomic(payload)
            done = sum(1 for x in payload["labels"] if x.get("status") == "drawn")
            sys.stderr.write(f"  saved {len(payload['labels'])} labels ({done} drawn) "
                             f"-> {LABELS}\n")
            return self._json(200, {"ok": True, "n": len(payload["labels"]),
                                    "at": time.strftime("%H:%M:%S")})
        except Exception as e:                                    # noqa: BLE001
            sys.stderr.write(f"  SAVE FAILED: {e}\n")
            return self._json(500, {"ok": False, "error": str(e)})


def main() -> int:
    if not ROOT.exists():
        print(f"annotation page not found at {ROOT}")
        return 1
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"serving {_which}: {ROOT}\n  open   http://127.0.0.1:{PORT}/\n"
          f"  saving {LABELS}\n  backups in {BACKUPS} (last {KEEP})\n"
          f"Ctrl-C to stop.\n", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped; labels are on disk.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
