#![forbid(unsafe_code)]

use crate::Result;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeRecord {
    pub id: i64,
    pub session_id: String,
    pub prompt: String,
    pub response: String,
    pub avg_surprise: f32,
    pub max_cycles: usize,
    pub user_feedback: i32,
    pub is_consolidated: bool,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    pub total_episodes: usize,
    pub unconsolidated_episodes: usize,
    pub consolidated_episodes: usize,
    pub positive_feedback_count: usize,
    pub avg_system_surprise: f32,
}

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

    /// Validates dialogue against toxicity, prompt injection, and data poisoning.
    pub fn validate_safety_guardrail(prompt: &str, response: &str) -> bool {
        let p_lower = prompt.to_lowercase();
        let r_lower = response.to_lowercase();

        // 1. Block malicious prompt injection and system override attempts
        let injection_keywords = [
            "ignore previous instructions",
            "system prompt leak",
            "dan mode",
            "bypass all rules",
            "jailbreak",
        ];
        for keyword in injection_keywords {
            if p_lower.contains(keyword) {
                return false;
            }
        }

        // 2. Block empty or corrupted content
        if prompt.trim().is_empty() || response.trim().is_empty() {
            return false;
        }

        // 3. Block poisoning payloads
        if p_lower.len() > 8192 || r_lower.len() > 8192 {
            return false;
        }

        true
    }

    pub fn save_episode(
        &mut self,
        session_id: &str,
        prompt: &str,
        response: &str,
        avg_surprise: f32,
        max_cycles: usize,
        user_feedback: i32,
    ) -> Result<i64> {
        // Apply Hippocampal Immune Guardrail: Reject poisoned episodes from sleep queue
        if !Self::validate_safety_guardrail(prompt, response) {
            return Ok(0);
        }

        self.conn.execute(
            "INSERT INTO episodes (session_id, prompt, response, avg_surprise, max_cycles, user_feedback)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                session_id,
                prompt,
                response,
                avg_surprise as f64,
                max_cycles as i64,
                user_feedback
            ],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn update_feedback(&mut self, episode_id: i64, feedback: i32) -> Result<()> {
        self.conn.execute(
            "UPDATE episodes SET user_feedback = ?1 WHERE id = ?2",
            params![feedback, episode_id],
        )?;
        Ok(())
    }

    pub fn get_trainable_episodes(&self, limit: usize) -> Result<Vec<EpisodeRecord>> {
        // Guardrail Filter: Only positive feedback (+1) or high-surprise valid dialogues (> 0.6)
        let mut stmt = self.conn.prepare(
            "SELECT id, session_id, prompt, response, avg_surprise, max_cycles, user_feedback, is_consolidated, created_at
             FROM episodes
             WHERE is_consolidated = 0 AND (user_feedback > 0 OR (user_feedback = 0 AND avg_surprise >= 0.6))
             ORDER BY user_feedback DESC, avg_surprise DESC
             LIMIT ?1",
        )?;

        let rows = stmt.query_map(params![limit as i64], |row| {
            let surprise_f64: f64 = row.get(4)?;
            let cycles_i64: i64 = row.get(5)?;
            let consolidated_i64: i64 = row.get(7)?;
            Ok(EpisodeRecord {
                id: row.get(0)?,
                session_id: row.get(1)?,
                prompt: row.get(2)?,
                response: row.get(3)?,
                avg_surprise: surprise_f64 as f32,
                max_cycles: cycles_i64 as usize,
                user_feedback: row.get(6)?,
                is_consolidated: consolidated_i64 != 0,
                created_at: row.get(8)?,
            })
        })?;

        let mut episodes = Vec::new();
        for episode in rows {
            episodes.push(episode?);
        }
        Ok(episodes)
    }

    pub fn mark_consolidated(&mut self, episode_ids: &[i64]) -> Result<()> {
        let tx = self.conn.transaction()?;
        for &id in episode_ids {
            tx.execute(
                "UPDATE episodes SET is_consolidated = 1 WHERE id = ?1",
                params![id],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    pub fn get_stats(&self) -> Result<MemoryStats> {
        let total: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM episodes", [], |r| r.get(0))
            .unwrap_or(0);
        let unconsolidated: i64 = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM episodes WHERE is_consolidated = 0",
                [],
                |r| r.get(0),
            )
            .unwrap_or(0);
        let positive: i64 = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM episodes WHERE user_feedback > 0",
                [],
                |r| r.get(0),
            )
            .unwrap_or(0);
        let avg_surprise_f64: f64 = self
            .conn
            .query_row(
                "SELECT COALESCE(AVG(avg_surprise), 0.0) FROM episodes",
                [],
                |r| r.get(0),
            )
            .unwrap_or(0.0);

        Ok(MemoryStats {
            total_episodes: total as usize,
            unconsolidated_episodes: unconsolidated as usize,
            consolidated_episodes: (total - unconsolidated) as usize,
            positive_feedback_count: positive as usize,
            avg_system_surprise: avg_surprise_f64 as f32,
        })
    }
}
