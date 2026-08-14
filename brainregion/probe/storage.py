"""模型指纹的 SQLite 存储:基线(probe_baselines) + 运行历史(probe_runs)。

对齐 eval/store 模式:幂等 CREATE TABLE、WAL、append-only 历史不覆盖。
基线可被重建(save_baseline 时旧的 active=0),历史永远只增。
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("brainregion.probe.storage")


def _db_path() -> Path:
    root = os.environ.get("UNITY_PROJECT_ROOT", ".")
    p = Path(root) / ".brain-region" / "probe" / "probe.db"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
    except Exception:  # noqa: BLE001
        pass
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS probe_baselines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_key TEXT NOT NULL,
            kind TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS probe_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_key TEXT NOT NULL,
            kind TEXT NOT NULL,
            mode TEXT NOT NULL,
            verdict TEXT,
            score REAL,
            details_json TEXT,
            cost_usd REAL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_baseline(model_key: str, kind: str, payload: dict) -> int:
    """写入基线并将同 model_key+kind 的旧基线置为 inactive(可重建,不删历史)。"""
    with _connect() as conn:
        conn.execute(
            "UPDATE probe_baselines SET active=0 WHERE model_key=? AND kind=?",
            (model_key, kind),
        )
        cur = conn.execute(
            "INSERT INTO probe_baselines (model_key, kind, payload_json, created_at, active)"
            " VALUES (?, ?, ?, ?, 1)",
            (model_key, kind, json.dumps(payload, ensure_ascii=False), _now()),
        )
        return int(cur.lastrowid)


def load_active_baseline(model_key: str, kind: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT payload_json, created_at FROM probe_baselines"
            " WHERE model_key=? AND kind=? AND active=1 ORDER BY id DESC LIMIT 1",
            (model_key, kind),
        ).fetchone()
    if not row:
        return None
    try:
        payload = json.loads(row["payload_json"])
    except Exception:  # noqa: BLE001
        logger.warning("基线 payload 解析失败 model_key=%s kind=%s", model_key, kind)
        return None
    payload = payload if isinstance(payload, dict) else {}
    payload["baseline_created_at"] = row["created_at"]
    return payload


def append_run(
    model_key: str,
    kind: str,
    mode: str,
    verdict: str | None,
    score: float | None,
    details: dict,
    cost_usd: float | None = None,
) -> int:
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO probe_runs (model_key, kind, mode, verdict, score, details_json,"
            " cost_usd, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                model_key,
                kind,
                mode,
                verdict,
                score,
                json.dumps(details, ensure_ascii=False),
                cost_usd,
                _now(),
            ),
        )
        return int(cur.lastrowid)


def recent_runs(
    model_key: str | None = None, kind: str | None = None, limit: int = 20
) -> list[dict]:
    """最近的探针运行(Inspector/汇总用):不含 details 大 JSON,只留判定摘要。"""
    sql = (
        "SELECT id, model_key, kind, mode, verdict, score, cost_usd, created_at"
        " FROM probe_runs"
    )
    conds, args = [], []
    if model_key:
        conds.append("model_key=?")
        args.append(model_key)
    if kind:
        conds.append("kind=?")
        args.append(kind)
    if conds:
        sql += " WHERE " + " AND ".join(conds)
    sql += " ORDER BY id DESC LIMIT ?"
    args.append(int(limit))
    with _connect() as conn:
        rows = conn.execute(sql, args).fetchall()
    return [dict(r) for r in rows]
