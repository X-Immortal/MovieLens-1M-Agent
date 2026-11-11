"""Exercise session deletion and active-job protection over HTTP."""
from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import web_server


class SessionDeletionTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.sessions = Path(self.directory.name)
        for patcher in (patch.object(web_server, "SESSIONS", self.sessions),
                        patch.dict(web_server.JOBS, {}, clear=True),
                        patch.object(web_server.Handler, "log_message", lambda *args: None)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), web_server.Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def stop_server(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()

    def request(self, path, method="DELETE"):
        request = urllib.request.Request(self.base + path, method=method)
        try:
            response = urllib.request.urlopen(request, timeout=3)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.code, json.load(response)

    def test_delete_removes_only_selected_session_and_finished_jobs(self):
        selected, other = "a" * 32, "b" * 32
        for identifier in (selected, other):
            web_server.save_session({"id": identifier, "title": identifier, "messages": []})
        web_server.JOBS.update({"selected": {"session_id": selected, "state": "completed"},
                                "other": {"session_id": other, "state": "completed"}})
        self.assertEqual(self.request(f"/api/sessions/{selected}"), (200, {"deleted": selected}))
        self.assertEqual(self.request(f"/api/sessions/{selected}", "GET")[0], 404)
        self.assertEqual(self.request(f"/api/sessions/{other}", "GET")[0], 200)
        self.assertEqual(set(web_server.JOBS), {"other"})
        self.assertEqual([item["id"] for item in web_server.list_sessions()], [other])

    def test_running_session_is_preserved_until_worker_finishes(self):
        identifier = "c" * 32
        web_server.save_session({"id": identifier, "messages": []})
        web_server.JOBS["running"] = {"session_id": identifier, "state": "running"}
        self.assertEqual(self.request(f"/api/sessions/{identifier}")[0], 409)
        self.assertIsNotNone(web_server.get_session(identifier))
        web_server.JOBS["running"]["state"] = "completed"
        self.assertEqual(self.request(f"/api/sessions/{identifier}")[0], 200)

    def test_missing_or_invalid_target_does_not_delete_other_files(self):
        sentinel = self.sessions / "keep.json"
        sentinel.write_text('{"keep":true}', encoding="utf-8")
        for path in ("/api/sessions/missing", "/api/sessions/" + "d" * 32,
                     "/api/sessions/../keep", "/api/artifacts/keep"):
            self.assertEqual(self.request(path)[0], 404)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), '{"keep":true}')


if __name__ == "__main__":
    unittest.main()
