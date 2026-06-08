import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import List, Dict, Any

from config.constants import *

from utils.logger import get_logger

logger = get_logger(__name__)

# Sentinel so callers can distinguish "don't touch this column" from "set it to NULL".
_UNSET = object()


class DatabaseManager:
    """SQLite database operations"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_db()

    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def init_db(self) -> None:
        """Initialize database schema"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # grouped table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS grouped (
                    group_id TEXT PRIMARY KEY,
                    broadcast_date TEXT NOT NULL,
                    broadcast_time TEXT NOT NULL,
                    program_title TEXT NOT NULL,
                    channel_name TEXT NOT NULL,
                    plain_text TEXT NOT NULL,
                    program_id TEXT,
                    status TEXT DEFAULT 'PENDING'
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS person_bio (
                    row_id TEXT PRIMARY KEY,
                    group_id TEXT NOT NULL,
                    program_id TEXT NOT NULL,
                    broadcast_time TEXT NOT NULL,
                    plain_text TEXT NOT NULL,
                    person TEXT,
                    role TEXT,
                    context_of_mention TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dialogues (
                    dialogue_id TEXT PRIMARY KEY,
                    group_id TEXT NOT NULL,
                    program_id TEXT NOT NULL,
                    broadcast_time TEXT NOT NULL,
                    dialogue TEXT NOT NULL,
                    speaker TEXT,
                    role TEXT,
                    confidence_score REAL,
                    verification_score REAL
                )
            """)

    def insert_grouped(self, transcript_data: List[Dict[str, str]]) -> None:
        """Insert grouped data"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for data in transcript_data:
                try:
                    cursor.execute(
                        "INSERT INTO grouped (group_id, broadcast_date, broadcast_time, program_title, channel_name, plain_text, program_id) VALUES (?,?,?,?,?,?,?)",
                        (data['group_id'], data['broadcast_date'], data['broadcast_time'], data['program_title'], data['channel_name'], data['plain_text'], data['program_id'])
                    )
                except sqlite3.IntegrityError:
                    logger.warning(f"Group ID already exists: {data['group_id']}")
            conn.commit()
    
    def insert_person_bio(self, rows: List[Dict[str, Any]]) -> None:
        """Insert person/role/context rows into person_bio"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for row in rows:
                try:
                    cursor.execute(
                        "INSERT INTO person_bio (row_id, group_id, program_id, broadcast_time, plain_text, person, role, context_of_mention) VALUES (?,?,?,?,?,?,?,?)",
                        (row['row_id'], row['group_id'], row['program_id'], row['broadcast_time'], row['plain_text'], row['person'], row.get('role'), row.get('context_of_mention'))
                    )
                except sqlite3.IntegrityError:
                    logger.warning(f"Row ID already exists in person_bio: {row['row_id']}")
            conn.commit()

    def get_row_by_status(self, status: str, table: str) -> List[Dict]:
        """Get rows from grouped table by status"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM {table} WHERE status = ?", (status,))
            return [dict(row) for row in cursor.fetchall()]

    def update_punctuated(self, group_id: str, punctuated_text: str) -> None:
        """Overwrite plain_text with punctuated text and mark row PUNCTUATED"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE grouped SET plain_text = ? WHERE group_id = ?",
                (punctuated_text, group_id)
            )
            conn.commit()

    def get_plain_text_by_group_ids(self, group_ids: List[str]) -> Dict[str, str]:
        """Map {group_id: plain_text} for the given group_ids (the Full Text per minute)."""
        if not group_ids:
            return {}
        placeholders = ",".join("?" for _ in group_ids)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT group_id, plain_text FROM grouped WHERE group_id IN ({placeholders})",
                tuple(group_ids)
            )
            return {row["group_id"]: row["plain_text"] for row in cursor.fetchall()}

    def get_person_bio_by_group_id(self, group_id: str) -> List[Dict]:
        """Return the person_bio candidate speakers (person/role/context) for one group."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT person, role, context_of_mention FROM person_bio WHERE group_id = ?",
                (group_id,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def insert_dialogues(self, rows: List[Dict[str, Any]]) -> None:
        """Insert diarized speaker turns into the dialogues table."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for row in rows:
                try:
                    cursor.execute(
                        "INSERT INTO dialogues (dialogue_id, group_id, program_id, broadcast_time, dialogue, speaker, role, confidence_score) VALUES (?,?,?,?,?,?,?,?)",
                        (row['dialogue_id'], row['group_id'], row['program_id'], row['broadcast_time'], row['dialogue'], row.get('speaker'), row.get('role'), row.get('confidence_score'))
                    )
                except sqlite3.IntegrityError:
                    logger.warning(f"Dialogue ID already exists: {row['dialogue_id']}")
            conn.commit()

    def get_low_confidence_dialogues(self, threshold: float) -> List[Dict]:
        """Return dialogues needing verification — confidence below `threshold` or NULL —
        joined with their group's full plain_text for context.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    d.dialogue_id,
                    d.group_id,
                    d.dialogue,
                    d.speaker,
                    d.confidence_score,
                    g.plain_text
                FROM dialogues d
                JOIN grouped g ON d.group_id = g.group_id
                WHERE (d.confidence_score < ? OR d.confidence_score IS NULL)
                  AND d.verification_score IS NULL
            """, (threshold,))
            return [dict(row) for row in cursor.fetchall()]

    def update_dialogue_verification(self, dialogue_id: str, speaker, verification_score,
                                     role=_UNSET) -> None:
        """Set the (possibly corrected) speaker and verification_score for one dialogue.

        The `role` column is only rewritten when an explicit `role` is passed (i.e. the speaker
        changed and the verifier assigned a new role). When `role` is omitted, the role populated
        earlier by diarization is left untouched.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if role is _UNSET:
                cursor.execute(
                    "UPDATE dialogues SET speaker = ?, verification_score = ? WHERE dialogue_id = ?",
                    (speaker, verification_score, dialogue_id)
                )
            else:
                cursor.execute(
                    "UPDATE dialogues SET speaker = ?, verification_score = ?, role = ? WHERE dialogue_id = ?",
                    (speaker, verification_score, role, dialogue_id)
                )
            conn.commit()

    def get_dialogues_export(self) -> List[Dict]:
        """Join dialogues with grouped to produce export-ready rows.

        Returns one dict per dialogue with:
        broadcast_date, broadcast_time, program_title, channel_name,
        dialogue, speaker, role, confidence_score, verification_score — sorted chronologically.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    g.broadcast_date     AS "Broadcast Date",
                    d.broadcast_time     AS "Broadcast Time",
                    g.program_title      AS "Program Title",
                    g.channel_name       AS "Channel Name",
                    d.dialogue           AS "Dialogue",
                    d.speaker            AS "Speaker",
                    d.role               AS "Role",
                    d.confidence_score   AS "Confidence Score",
                    d.verification_score AS "Verification Score"
                FROM dialogues d
                JOIN grouped g ON d.group_id = g.group_id
                ORDER BY g.broadcast_date, d.broadcast_time
            """)
            return [dict(row) for row in cursor.fetchall()]

    def update_llm_classification(self, pair_id: str, label, confidence, status: str) -> None:
        """Set llm_label, llm_confidence_score, and status for a sentence_pairwise row."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE sentence_pairwise SET llm_label = ?, llm_confidence_score = ?, status = ? WHERE pair_id = ?",
                (label, confidence, status, pair_id)
            )
            conn.commit()

    def update_bert_classification(self, pair_id: str, label, confidence, status: str) -> None:
        """Set bert_label, bert_confidence_score, and status for a sentence_pairwise row."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE sentence_pairwise SET bert_label = ?, bert_confidence_score = ?, status = ? WHERE pair_id = ?",
                (label, confidence, status, pair_id)
            )
            conn.commit()

    def update_status(self, table: str, row_id: str, status: str) -> None:
        """Updates the status in a specified table"""
        allowed_tables = {"grouped", "person_bio", "sentence_pairwise", "dialogues"}
        if table not in allowed_tables:
            raise ValueError(f"Invalid table: {table}")

        pk_column = {
            "grouped": "group_id",
            "person_bio": "row_id",
            "sentence_pairwise": "pair_id",
            "dialogues": "dialogue_id"
        }[table]

        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = f"UPDATE {table} SET status = ? WHERE {pk_column} = ?"
            cursor.execute(query, (status, row_id))
            conn.commit()