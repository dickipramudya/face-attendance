"""SQLite storage: people, their face embeddings, and attendance events."""
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS people (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    emp_no TEXT UNIQUE,
    name TEXT NOT NULL,
    dept TEXT DEFAULT '',
    phone TEXT DEFAULT '',
    consent_at REAL NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    embedding BLOB NOT NULL,
    image TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    camera_id TEXT NOT NULL,
    kind TEXT NOT NULL,          -- 'in' or 'out'
    ts REAL NOT NULL,
    day TEXT NOT NULL,           -- YYYY-MM-DD, local time
    score REAL NOT NULL,
    snapshot TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS events_day ON events(day, person_id);
"""


class Database:
    def __init__(self, path: Path):
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.executescript(SCHEMA)
        self.lock = threading.Lock()

    def q(self, sql, args=()):
        with self.lock:
            return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    def x(self, sql, args=()):
        with self.lock:
            cur = self.conn.execute(sql, args)
            self.conn.commit()
            return cur.lastrowid

    # ── people ──────────────────────────────────────────────────────────
    def add_person(self, emp_no, name, dept, phone):
        now = time.time()
        return self.x(
            "INSERT INTO people(emp_no, name, dept, phone, consent_at, created_at) VALUES (?,?,?,?,?,?)",
            (emp_no or None, name, dept, phone, now, now),
        )

    def add_face(self, person_id, emb: np.ndarray, image: str):
        return self.x("INSERT INTO faces(person_id, embedding, image) VALUES (?,?,?)",
                      (person_id, emb.astype(np.float32).tobytes(), image))

    def people(self):
        return self.q("""
            SELECT p.*, COUNT(f.id) AS faces, MIN(f.image) AS photo
            FROM people p LEFT JOIN faces f ON f.person_id = p.id
            GROUP BY p.id ORDER BY p.name COLLATE NOCASE""")

    def person(self, pid):
        rows = self.q("SELECT * FROM people WHERE id = ?", (pid,))
        return rows[0] if rows else None

    def delete_person(self, pid):
        self.x("DELETE FROM people WHERE id = ?", (pid,))

    def all_embeddings(self):
        with self.lock:
            rows = self.conn.execute("SELECT person_id, embedding FROM faces").fetchall()
        return [(r[0], np.frombuffer(r[1], dtype=np.float32)) for r in rows]

    # ── events ──────────────────────────────────────────────────────────
    def add_event(self, person_id, camera_id, kind, ts, day, score, snapshot):
        return self.x(
            "INSERT INTO events(person_id, camera_id, kind, ts, day, score, snapshot) VALUES (?,?,?,?,?,?,?)",
            (person_id, camera_id, kind, ts, day, score, snapshot),
        )

    def events_on(self, day, limit=None):
        sql = """SELECT e.*, p.name, p.emp_no, p.dept FROM events e JOIN people p ON p.id = e.person_id
                 WHERE e.day = ? ORDER BY e.ts DESC"""
        if limit:
            sql += f" LIMIT {int(limit)}"
        return self.q(sql, (day,))

    def events_between(self, day_from, day_to):
        return self.q("""SELECT e.*, p.name, p.emp_no, p.dept FROM events e JOIN people p ON p.id = e.person_id
                         WHERE e.day BETWEEN ? AND ? ORDER BY e.ts""", (day_from, day_to))
