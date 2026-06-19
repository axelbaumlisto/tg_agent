//! Per-(chat,thread) async lock registry for Telegram dispatch.
//!
//! The registry serializes only the short dispatch-decision window for one
//! Telegram conversation key. It deliberately returns an owned mutex guard so
//! callers can drop the registry map guards before awaiting the per-key mutex.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::{Mutex, OwnedMutexGuard, RwLock};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChatThreadKey {
    pub chat_id: i64,
    pub thread_id: Option<i32>,
}

impl ChatThreadKey {
    pub fn new(chat_id: i64, thread_id: Option<i32>) -> Self {
        Self { chat_id, thread_id }
    }
}

#[derive(Debug)]
pub struct PerChatLockGuard {
    _guard: OwnedMutexGuard<()>,
    waited: bool,
}

impl PerChatLockGuard {
    pub fn waited(&self) -> bool {
        self.waited
    }
}

#[derive(Debug, Default)]
pub struct PerChatLocks {
    locks: RwLock<HashMap<ChatThreadKey, Arc<Mutex<()>>>>,
}

impl PerChatLocks {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn lock(&self, key: ChatThreadKey) -> PerChatLockGuard {
        let mutex = self.mutex_for(key).await;
        match mutex.clone().try_lock_owned() {
            Ok(guard) => PerChatLockGuard {
                _guard: guard,
                waited: false,
            },
            Err(_) => PerChatLockGuard {
                _guard: mutex.lock_owned().await,
                waited: true,
            },
        }
    }

    async fn mutex_for(&self, key: ChatThreadKey) -> Arc<Mutex<()>> {
        if let Some(existing) = self.locks.read().await.get(&key).cloned() {
            return existing;
        }

        let mut locks = self.locks.write().await;
        locks
            .entry(key)
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::sync::Barrier;
    use tokio::time::{Duration, sleep, timeout};

    fn update_max(max: &AtomicUsize, candidate: usize) {
        let mut current = max.load(Ordering::Relaxed);
        while candidate > current {
            match max.compare_exchange(current, candidate, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(next) => current = next,
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn same_key_dispatches_are_serial() {
        let locks = Arc::new(PerChatLocks::new());
        let key = ChatThreadKey::new(42, Some(7));
        let start = Arc::new(Barrier::new(8));
        let in_flight = Arc::new(AtomicUsize::new(0));
        let max_in_flight = Arc::new(AtomicUsize::new(0));

        let mut tasks = Vec::new();
        for _ in 0..8 {
            let locks = locks.clone();
            let start = start.clone();
            let in_flight = in_flight.clone();
            let max_in_flight = max_in_flight.clone();
            tasks.push(tokio::spawn(async move {
                start.wait().await;
                let _guard = locks.lock(key).await;
                let previous = in_flight.fetch_add(1, Ordering::SeqCst);
                assert_eq!(previous, 0, "same key entered concurrently");
                update_max(&max_in_flight, previous + 1);
                sleep(Duration::from_millis(10)).await;
                in_flight.fetch_sub(1, Ordering::SeqCst);
            }));
        }

        for task in tasks {
            task.await.expect("task panicked");
        }
        assert_eq!(max_in_flight.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn different_keys_overlap() {
        let locks = Arc::new(PerChatLocks::new());
        let barrier = Arc::new(Barrier::new(2));
        let in_flight = Arc::new(AtomicUsize::new(0));
        let max_in_flight = Arc::new(AtomicUsize::new(0));

        let mut tasks = Vec::new();
        for key in [ChatThreadKey::new(1, None), ChatThreadKey::new(2, None)] {
            let locks = locks.clone();
            let barrier = barrier.clone();
            let in_flight = in_flight.clone();
            let max_in_flight = max_in_flight.clone();
            tasks.push(tokio::spawn(async move {
                let _guard = locks.lock(key).await;
                let now = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                update_max(&max_in_flight, now);
                barrier.wait().await;
                in_flight.fetch_sub(1, Ordering::SeqCst);
            }));
        }

        timeout(Duration::from_secs(1), async {
            for task in tasks {
                task.await.expect("task panicked");
            }
        })
        .await
        .expect("different keys should overlap instead of deadlocking at the barrier");
        assert_eq!(max_in_flight.load(Ordering::SeqCst), 2);
    }
}
