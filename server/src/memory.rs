#![forbid(unsafe_code)]

use crate::Result;
use rusqlite::Connection;
use std::path::Path;

pub struct EpisodeMemory {
    conn: Connection,
}

impl EpisodeMemory {
    pub fn open<P: AsRef<Path>>(db_path: P) -> Result<Self> {
        let conn = Connection::open(db_path)?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                avg_surprise REAL NOT NULL,
                max_cycles INTEGER NOT NULL,
                user_feedback INTEGER DEFAULT 0,
                is_consolidated BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_episodes_unconsolidated 
            ON episodes (is_consolidated, user_feedback, avg_surprise DESC);",
        )?;

        Ok(Self { conn })
    }

    pub fn conn(&self) -> &Connection {
        &self.conn
    }
}
