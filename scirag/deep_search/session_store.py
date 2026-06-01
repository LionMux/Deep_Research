from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict

from .mcp_contracts import FullReportSession, make_new_session_id


@dataclass(frozen=True)
class DiskPaths:
    root_dir: str
    filename_prefix: str = "session_"

    def session_path(self, session_id: str) -> str:
        safe = session_id.replace("/", "_")
        return os.path.join(self.root_dir, f"{self.filename_prefix}{safe}.json")


class SessionStore:
    def get(self, session_id: str) -> FullReportSession:
        raise NotImplementedError

    def put(self, session: FullReportSession) -> None:
        raise NotImplementedError


class InMemorySessionStore(SessionStore):
    def __init__(self) -> None:
        self._sessions: Dict[str, FullReportSession] = {}

    def get(self, session_id: str) -> FullReportSession:
        if session_id not in self._sessions:
            raise KeyError(f"Session not found: {session_id}")
        return self._sessions[session_id]

    def put(self, session: FullReportSession) -> None:
        if not session.get("session_id"):
            raise ValueError("Session must have session_id")
        self._sessions[session["session_id"]] = session


class DiskSessionStore(SessionStore):
    def __init__(self, paths: DiskPaths) -> None:
        self._paths = paths
        os.makedirs(self._paths.root_dir, exist_ok=True)

    def get(self, session_id: str) -> FullReportSession:
        path = self._paths.session_path(session_id)
        if not os.path.exists(path):
            raise KeyError(f"Session not found on disk: {session_id}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def put(self, session: FullReportSession) -> None:
        session_id = session.get("session_id")
        if not session_id:
            raise ValueError("Session must have session_id")
        path = self._paths.session_path(session_id)
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(session, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)


class HybridSessionStore(SessionStore):
    """
    Wraps two stores: writes go to both, reads prefer memory fallback to disk.
    """

    def __init__(self, memory: InMemorySessionStore, disk: DiskSessionStore) -> None:
        self._memory = memory
        self._disk = disk

    def get(self, session_id: str) -> FullReportSession:
        try:
            return self._memory.get(session_id)
        except KeyError:
            return self._disk.get(session_id)

    def put(self, session: FullReportSession) -> None:
        self._memory.put(session)
        self._disk.put(session)


def new_session_id(prefix: str = "sess") -> str:
    return make_new_session_id(prefix=prefix)
