from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

THEME_META_VERSION = 1

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf", ".docx"}


def theme_id(query: str) -> str:
    """
    Deterministic, filesystem-safe theme identifier derived from query.
    """
    q = (query or "").strip()
    if not q:
        q = "empty_query"

    prefix = re.sub(r"[^a-zA-Z0-9_-]+", "-", q).strip("-").lower()
    prefix = prefix[:48] if prefix else "theme"

    digest = hashlib.sha256(q.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _safe_relname(paper_id: str) -> str:
    # paper_id == file_path.stem per ingestion contract.
    p = (paper_id or "").strip()
    if not p:
        p = "unknown"
    p = re.sub(r"[^a-zA-Z0-9._-]+", "_", p)
    return p[:180]


def _iter_doc_ids_in_dir(base_papers_dir: Path) -> List[str]:
    # Remaining pool must match ingestion's ingestion contract: doc_id == file_path.stem.
    seen: set[str] = set()
    doc_ids: List[str] = []

    for fp in base_papers_dir.rglob("*"):
        if not fp.is_file():
            continue
        if fp.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        stem = fp.stem
        if stem and stem not in seen:
            seen.add(stem)
            doc_ids.append(stem)

    doc_ids.sort()
    return doc_ids


def _read_meta(meta_path: Path) -> dict:
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_meta(meta_path: Path, meta: dict) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def build_paper_id_to_path_map(base_papers_dir: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for fp in base_papers_dir.rglob("*"):
        if not fp.is_file():
            continue
        if fp.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        pid = fp.stem
        if pid and pid not in mapping:
            mapping[pid] = fp
    return mapping


@dataclass(frozen=True)
class MaterializePlan:
    used_doc_ids: List[str]
    remaining_doc_ids: List[str]


def compute_materialize_plan(*, base_papers_dir: Path, used_doc_ids: Sequence[str]) -> MaterializePlan:
    base_all = _iter_doc_ids_in_dir(base_papers_dir)
    used_set = {str(x) for x in used_doc_ids if x}
    remaining = [pid for pid in base_all if pid not in used_set]
    used_sorted = sorted(used_set)
    return MaterializePlan(used_doc_ids=used_sorted, remaining_doc_ids=remaining)


def _unlink_if_exists(p: Path) -> None:
    try:
        if p.is_symlink() or p.is_file():
            p.unlink()
        elif p.is_dir():
            shutil.rmtree(p)
    except FileNotFoundError:
        return


def _try_link(src: Path, dst: Path) -> Tuple[bool, str]:
    """
    Prefer symlink, then hardlink, then copy.
    """
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            _unlink_if_exists(dst)
        os.symlink(src, dst)
        return True, "symlink"
    except Exception:
        pass

    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            _unlink_if_exists(dst)
        os.link(src, dst)
        return True, "hardlink"
    except Exception:
        pass

    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            _unlink_if_exists(dst)
        shutil.copy2(src, dst)
        return True, "copy"
    except Exception as e:
        return False, f"copy_failed:{e}"


def materialize_used_remaining(
    *,
    base_papers_dir: str,
    theme: str,
    used_doc_ids: Sequence[str],
    resume: bool = True,
) -> dict:
    """
    Materialize into:
      papers/themes/<theme>/used/<paper_id>/<original_filename>
      papers/themes/<theme>/remaining/<paper_id>/<original_filename>
      papers/themes/<theme>/meta.json

    Idempotent:
      - re-running with same used_doc_ids doesn't break existing links/files
      - follow-up incremental: merges prev used + adds, then prunes remaining
    """
    base_dir = Path(base_papers_dir).expanduser().resolve()
    if not base_dir.exists():
        raise FileNotFoundError(f"base_papers_dir not found: {base_dir}")

    theme_dir = Path("papers") / "themes" / theme
    used_dir = theme_dir / "used"
    remaining_dir = theme_dir / "remaining"
    meta_path = theme_dir / "meta.json"

    prev_meta = _read_meta(meta_path)
    prev_used = set(prev_meta.get("used_doc_ids", []) or [])

    plan = compute_materialize_plan(base_papers_dir=base_dir, used_doc_ids=used_doc_ids)
    used_set = set(plan.used_doc_ids)
    remaining_set = set(plan.remaining_doc_ids)

    pid_to_path = build_paper_id_to_path_map(base_dir)

    used_dir.mkdir(parents=True, exist_ok=True)
    remaining_dir.mkdir(parents=True, exist_ok=True)

    created_used_files = 0
    created_remaining_files = 0
    removed_remaining_pids = 0
    skipped_missing_source = 0

    # Incremental prune remaining: remove remaining/<pid>/ if pid became used
    if resume and prev_used:
        newly_used = used_set - prev_used
    else:
        newly_used = used_set

    for pid in newly_used:
        # only prune if it was previously in remaining; pruning by checking existence is fine
        pid_dir = remaining_dir / _safe_relname(pid)
        if pid_dir.exists():
            _unlink_if_exists(pid_dir)
            removed_remaining_pids += 1

    def _dst_file(target_root: Path, pid: str, src: Path) -> Path:
        # used/<pid>/<original_filename>
        pid_dir = target_root / _safe_relname(pid)
        return pid_dir / src.name

    # Materialize used files
    for pid in plan.used_doc_ids:
        src = pid_to_path.get(pid)
        if not src:
            skipped_missing_source += 1
            continue
        dst = _dst_file(used_dir, pid, src)

        # idempotency best-effort: if dst exists and points to same source, keep
        if dst.exists() or dst.is_symlink():
            try:
                if dst.is_symlink() and os.path.realpath(dst) == os.path.realpath(src):
                    continue
                if dst.is_file() and dst.stat().st_size == src.stat().st_size:
                    continue
            except Exception:
                pass

        ok, _how = _try_link(src, dst)
        if ok:
            created_used_files += 1

    # Materialize remaining files (full correctness)
    for pid in plan.remaining_doc_ids:
        src = pid_to_path.get(pid)
        if not src:
            skipped_missing_source += 1
            continue
        dst = _dst_file(remaining_dir, pid, src)

        if dst.exists() or dst.is_symlink():
            try:
                if dst.is_symlink() and os.path.realpath(dst) == os.path.realpath(src):
                    continue
                if dst.is_file() and dst.stat().st_size == src.stat().st_size:
                    continue
            except Exception:
                pass

        ok, _how = _try_link(src, dst)
        if ok:
            created_remaining_files += 1

    # Remove remaining/<pid>/ directories not in computed remaining set
    remaining_safe_names = {_safe_relname(pid) for pid in plan.remaining_doc_ids}
    if remaining_dir.exists():
        for child in remaining_dir.iterdir():
            if not child.is_dir():
                # tolerate leftover files
                continue
            if child.name not in remaining_safe_names:
                _unlink_if_exists(child)

    meta = {
        "meta_version": THEME_META_VERSION,
        "theme": theme,
        "base_papers_dir": str(base_dir),
        "used_doc_ids_count": len(plan.used_doc_ids),
        "remaining_doc_ids_count": len(plan.remaining_doc_ids),
        "used_doc_ids": plan.used_doc_ids,
        "remaining_doc_ids": plan.remaining_doc_ids,
        "resume": bool(resume),
    }
    _write_meta(meta_path, meta)

    return {
        "status": "materialized",
        "theme": theme,
        "used_dir": str(used_dir),
        "remaining_dir": str(remaining_dir),
        "created_used_files": created_used_files,
        "created_remaining_files": created_remaining_files,
        "removed_remaining_pids": removed_remaining_pids,
        "skipped_missing_source": skipped_missing_source,
    }


def theme_materialize_from_query(
    *,
    base_papers_dir: str,
    query: str,
    used_doc_ids: Sequence[str],
    resume: bool = True,
) -> dict:
    t = theme_id(query)
    res = materialize_used_remaining(
        base_papers_dir=base_papers_dir,
        theme=t,
        used_doc_ids=used_doc_ids,
        resume=resume,
    )
    res["theme_id"] = t
    return res
