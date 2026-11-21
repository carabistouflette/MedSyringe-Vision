"""Explainable scene reports with append-only human review."""
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import uuid


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def inventory(objects, expected, field):
    """Compare a count, or a complete reference inventory over one visual field."""
    if field == "count":
        expected_total = expected["syringe"]
        delta = len(objects)-expected_total
        return dict(mode=field, expected=expected, observed={"syringe": len(objects)},
                    missing={"syringe": max(0, -delta)}, extra={"syringe": max(0, delta)},
                    evidence={"syringe": [obj["id"] for obj in objects]},
                    unresolved=[], minimum_missing=max(0, -delta))
    observed = Counter(obj.get(field) for obj in objects)
    unresolved = [obj["id"] for obj in objects if obj.get(field) is None]
    keys = sorted(set(expected) | {value for value in observed if value is not None})
    deficits = {key: max(0, expected.get(key, 0) - observed[key]) for key in keys}
    return dict(mode=field, expected=expected, observed={key: observed[key] for key in keys},
                missing={key: max(0, value-len(unresolved)) for key, value in deficits.items()},
                possible_missing=deficits, minimum_missing=max(0, sum(deficits.values())-len(unresolved)),
                extra={key: max(0, observed[key]-expected.get(key, 0)) for key in keys},
                evidence={key: [obj["id"] for obj in objects if obj.get(field) == key] for key in keys},
                unresolved=unresolved)


def compare(previous, current):
    previous_objects, current_objects = previous["objects"], current["objects"]
    labels = {}
    for field in ("scale", "color"):
        before = Counter(obj.get(field) or "unknown" for obj in previous_objects)
        after = Counter(obj.get(field) or "unknown" for obj in current_objects)
        labels[field] = {name: after[name]-before[name] for name in sorted(before.keys() | after.keys())}
    return dict(previous_id=previous["id"], current_id=current["id"],
                previous_revision=previous["revision"], current_revision=current["revision"],
                count_before=len(previous_objects), count_after=len(current_objects),
                count_change=len(current_objects)-len(previous_objects), attribute_count_changes=labels,
                observations={"previous": [r["id"] for r in previous_objects],
                              "current": [r["id"] for r in current_objects]},
                identity_status="unresolved",
                reason="Separate still images do not establish additions, removals or movement of physical objects.")


class Conflict(Exception):
    pass


class Store:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.database = self.directory / "observations.sqlite3"
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS observation (
                    id TEXT PRIMARY KEY, original TEXT NOT NULL, image BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS review (
                    id TEXT NOT NULL REFERENCES observation(id), revision INTEGER NOT NULL,
                    event TEXT NOT NULL, PRIMARY KEY (id, revision)
                );
                CREATE TABLE IF NOT EXISTS source_asset (
                    id TEXT PRIMARY KEY REFERENCES observation(id), payload BLOB NOT NULL
                );
            """)

    def connect(self):
        db = sqlite3.connect(self.database)
        db.execute("PRAGMA foreign_keys = ON")
        return db

    def create(self, original, image, identifier=None, *, source=None):
        identifier = identifier or uuid.uuid4().hex
        original = {**original, "id": identifier, "created_at": utc_now()}
        with self.connect() as db:
            inserted = db.execute("INSERT OR IGNORE INTO observation VALUES (?, ?, ?)",
                                  (identifier, json.dumps(original), image))
            if inserted.rowcount == 1 and source is not None:
                db.execute("INSERT INTO source_asset VALUES (?, ?)", (identifier,source))
        return self.report(identifier)

    def summaries(self):
        with self.connect() as db:
            rows = db.execute("SELECT original FROM observation ORDER BY rowid DESC LIMIT 300").fetchall()
        keys = ("id", "name", "created_at", "kind")
        return [{k: original[k] for k in keys} for original in map(lambda r: json.loads(r[0]), rows)]

    def image(self, identifier):
        with self.connect() as db:
            row = db.execute("SELECT image FROM observation WHERE id=?", (identifier,)).fetchone()
        if not row:
            raise KeyError(identifier)
        return row[0]

    def source(self, identifier):
        with self.connect() as db:
            row = db.execute("SELECT payload FROM source_asset WHERE id=?", (identifier,)).fetchone()
        return row[0] if row is not None else None

    def report(self, identifier):
        with self.connect() as db:
            db.execute("BEGIN")
            source = db.execute("SELECT original FROM observation WHERE id=?", (identifier,)).fetchone()
            if source is None:
                raise KeyError(identifier)
            records = db.execute("SELECT event FROM review WHERE id=? ORDER BY revision",
                                 (identifier,)).fetchall()
        original = json.loads(source[0])
        events = [json.loads(r[0]) for r in records]
        last = events[-1] if events else original
        return dict(id=identifier, name=original["name"], kind=original["kind"],
                    revision=len(events), original=original, reviews=events,
                    objects=last["objects"], review_status=last.get("review_status", "unreviewed"))

    def correct(self, identifier, revision, objects, reviewer, note, status):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM observation WHERE id=?", (identifier,)).fetchone() is None:
                raise KeyError(identifier)
            current = db.execute("SELECT count(*) FROM review WHERE id=?", (identifier,)).fetchone()[0]
            if revision != current:
                raise Conflict("This observation has a newer revision. Reload it before saving.")
            event = dict(revision=revision+1, reviewer=reviewer, note=note, created_at=utc_now(),
                         objects=objects, review_status=status)
            db.execute("INSERT INTO review VALUES (?, ?, ?)",
                       (identifier, revision+1, json.dumps(event)))
        return self.report(identifier)
