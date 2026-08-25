#![forbid(unsafe_code)]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

pub struct SleepLearner {
    is_preempted: Arc<AtomicBool>,
}

impl SleepLearner {
    pub fn new() -> Self {
        Self {
            is_preempted: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn trigger_preemption(&self) {
        self.is_preempted.store(true, Ordering::SeqCst);
    }

    pub fn is_interrupted(&self) -> bool {
        self.is_preempted.load(Ordering::SeqCst)
    }
}

impl Default for SleepLearner {
    fn default() -> Self {
        Self::new()
    }
}
