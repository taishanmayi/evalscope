# Copyright (c) Alibaba, Inc. and its affiliates.
"""SQLite database layer for persisting EvalScope task history and results."""

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from evalscope.utils.logger import get_logger

logger = get_logger()

_DB_FILENAME = '.evalscope_tasks.db'

# Schema DDL
_DDL = """
CREATE TABLE IF NOT EXISTS tasks (
    id          TEXT PRIMARY KEY,
    model       TEXT NOT NULL,
    datasets    TEXT NOT NULL,
    config      TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    work_dir    TEXT,
    created_at  TEXT NOT NULL,
    started_at  TEXT,
    completed_at TEXT,
    error       TEXT
);

CREATE TABLE IF NOT EXISTS task_results (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id     TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    model       TEXT NOT NULL,
    dataset     TEXT NOT NULL,
    metric      TEXT NOT NULL,
    score       REAL NOT NULL,
    num         INTEGER,
    details     TEXT,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_task_results_task_id ON task_results(task_id);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
"""


def _now() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')


class TaskDatabase:
    """Thread-safe SQLite-backed store for evaluation task history and results.

    A single shared instance is created via :func:`get_db`.
    """

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._local = threading.local()
        self._init_db()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        """Return a per-thread SQLite connection (created lazily)."""
        conn = getattr(self._local, 'conn', None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute('PRAGMA foreign_keys = ON')
            conn.execute('PRAGMA journal_mode = WAL')
            self._local.conn = conn
        return conn

    def _init_db(self) -> None:
        conn = self._conn()
        conn.executescript(_DDL)
        conn.commit()

    # ------------------------------------------------------------------
    # Task CRUD
    # ------------------------------------------------------------------

    def create_task(
        self,
        task_id: str,
        model: str,
        datasets: List[str],
        config: Dict[str, Any],
        work_dir: Optional[str] = None,
    ) -> None:
        """Insert a new task record with status='running'."""
        conn = self._conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO tasks
                (id, model, datasets, config, status, work_dir, created_at, started_at)
            VALUES (?, ?, ?, ?, 'running', ?, ?, ?)
            """,
            (
                task_id,
                model,
                json.dumps(datasets, ensure_ascii=False),
                json.dumps(config, ensure_ascii=False),
                work_dir,
                _now(),
                _now(),
            ),
        )
        conn.commit()

    def update_task_status(
        self,
        task_id: str,
        status: str,
        error: Optional[str] = None,
    ) -> None:
        """Update the status (and optionally error) of an existing task."""
        conn = self._conn()
        completed_at = _now() if status in ('completed', 'error') else None
        conn.execute(
            """
            UPDATE tasks
            SET status = ?, error = ?, completed_at = ?
            WHERE id = ?
            """,
            (status, error, completed_at, task_id),
        )
        conn.commit()

    def save_task_results(self, task_id: str, results: List[Dict[str, Any]]) -> None:
        """Persist per-dataset metric scores extracted from report files.

        Each element of *results* should have keys:
            model, dataset, metric, score, num (optional), details (optional).
        """
        if not results:
            return
        now = _now()
        conn = self._conn()
        conn.executemany(
            """
            INSERT INTO task_results (task_id, model, dataset, metric, score, num, details, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    task_id,
                    r['model'],
                    r['dataset'],
                    r['metric'],
                    float(r['score']),
                    r.get('num'),
                    json.dumps(r.get('details'), ensure_ascii=False) if r.get('details') else None,
                    now,
                )
                for r in results
            ],
        )
        conn.commit()

    def get_tasks(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Return a list of task records, newest first."""
        conn = self._conn()
        if status and status != '全部':
            rows = conn.execute(
                'SELECT * FROM tasks WHERE status = ? ORDER BY created_at DESC LIMIT ? OFFSET ?',
                (status, limit, offset),
            ).fetchall()
        else:
            rows = conn.execute(
                'SELECT * FROM tasks ORDER BY created_at DESC LIMIT ? OFFSET ?',
                (limit, offset),
            ).fetchall()
        return [self._task_row_to_dict(r) for r in rows]

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Return a single task record or None."""
        conn = self._conn()
        row = conn.execute('SELECT * FROM tasks WHERE id = ?', (task_id,)).fetchone()
        return self._task_row_to_dict(row) if row else None

    def get_task_results(self, task_id: str) -> List[Dict[str, Any]]:
        """Return all metric results for a given task."""
        conn = self._conn()
        rows = conn.execute(
            'SELECT * FROM task_results WHERE task_id = ? ORDER BY dataset, metric',
            (task_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_task(self, task_id: str) -> None:
        """Delete a task record (cascade removes its results)."""
        conn = self._conn()
        conn.execute('DELETE FROM tasks WHERE id = ?', (task_id,))
        conn.commit()

    # ------------------------------------------------------------------
    # Summary / aggregation
    # ------------------------------------------------------------------

    def get_summary(
        self,
        model: Optional[str] = None,
        dataset: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return aggregated (latest) scores grouped by model × dataset × metric.

        Applies optional model/dataset substring filters.
        """
        conn = self._conn()
        clauses: List[str] = []
        params: List[Any] = []

        if model:
            clauses.append('tr.model LIKE ?')
            params.append(f'%{model}%')
        if dataset:
            clauses.append('tr.dataset LIKE ?')
            params.append(f'%{dataset}%')

        where = ('WHERE ' + ' AND '.join(clauses)) if clauses else ''
        sql = f"""
            SELECT tr.model, tr.dataset, tr.metric,
                   AVG(tr.score) AS avg_score,
                   MAX(tr.score) AS max_score,
                   MIN(tr.score) AS min_score,
                   COUNT(*)      AS run_count
            FROM task_results tr
            JOIN tasks t ON t.id = tr.task_id
            {where}
            GROUP BY tr.model, tr.dataset, tr.metric
            ORDER BY tr.model, tr.dataset, tr.metric
        """
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _task_row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        # Deserialize JSON columns
        for col in ('datasets', 'config'):
            if d.get(col):
                try:
                    d[col] = json.loads(d[col])
                except (json.JSONDecodeError, TypeError):
                    pass
        return d


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_db_instance: Optional[TaskDatabase] = None
_db_lock = threading.Lock()


def get_db(output_dir: Optional[str] = None) -> TaskDatabase:
    """Return the module-level :class:`TaskDatabase` singleton.

    The database file is placed inside *output_dir* (defaults to the value of
    ``OUTPUT_DIR`` from :mod:`evalscope.service.utils.log`).
    """
    global _db_instance
    if _db_instance is None:
        with _db_lock:
            if _db_instance is None:
                if output_dir is None:
                    from evalscope.service.utils.log import OUTPUT_DIR
                    output_dir = OUTPUT_DIR
                os.makedirs(output_dir, exist_ok=True)
                db_path = os.path.join(output_dir, _DB_FILENAME)
                logger.info(f'[TaskDatabase] Initialising database at {db_path}')
                _db_instance = TaskDatabase(db_path)
    return _db_instance
