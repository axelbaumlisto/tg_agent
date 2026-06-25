//! Process-wide fff picker registry for the fast-index backend.
//!
//! The registry owns fff's long-lived [`SharedFilePicker`] instances so the
//! Telegram daemon gets warm content indexes, mmap cache, and background
//! watcher state across turns. Tools only borrow handles from here; they never
//! decide watcher lifecycle or workspace caps themselves.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
#[cfg(test)]
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::{AtomicU64, Ordering};

use fff_search::file_picker::{FFFMode, FilePicker, FilePickerOptions};
use fff_search::{ContentCacheBudget, SharedFilePicker, SharedFrecency};
use parking_lot::Mutex;

use crate::config::Config;
use crate::types::{
    FFF_PICKER_REGISTRY_CAP_FALLBACK_COUNT, FFF_PICKER_REGISTRY_CREATED_COUNT,
    FFF_PICKER_REGISTRY_REUSED_COUNT,
};

const DEFAULT_FAST_INDEX_MAX_FILES: usize = 30_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FffPickerBackend {
    FastIndex,
    Fallback,
}

#[derive(Debug, Clone)]
pub struct FffPickerHandle {
    pub backend: FffPickerBackend,
    pub picker: SharedFilePicker,
}

#[derive(Debug, Clone, Copy)]
pub struct FffRegistryConfig {
    pub enabled: bool,
    pub max_workspaces: usize,
    pub cache_max_bytes: u64,
}

impl FffRegistryConfig {
    pub fn from_config(config: &Config) -> Self {
        Self {
            enabled: config.fff_fast_index_enabled,
            max_workspaces: config.fff_fast_index_max_workspaces,
            cache_max_bytes: config.fff_fast_index_cache_max_bytes,
        }
    }
}

#[derive(Debug)]
enum RegistryEntry {
    /// A caller reserved this workspace slot and is building the fast picker
    /// outside the registry lock. Other callers fall back instead of building a
    /// duplicate picker for the same key.
    Pending,
    Ready {
        picker: SharedFilePicker,
    },
}

#[derive(Debug, Default)]
struct RegistryState {
    entries: HashMap<PathBuf, RegistryEntry>,
}

#[cfg(test)]
type TestFastFactory = Arc<dyn Fn(&Path, u64) -> Result<SharedFilePicker, String> + Send + Sync>;

/// Long-lived fff picker registry keyed by canonical workspace path.
#[derive(Default)]
pub struct FffPickerRegistry {
    state: Mutex<RegistryState>,
    creation_count: AtomicU64,
    #[cfg(test)]
    test_fast_factory: Mutex<Option<TestFastFactory>>,
}

impl std::fmt::Debug for FffPickerRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FffPickerRegistry")
            .field("state", &self.state)
            .field("creation_count", &self.creation_count)
            .finish()
    }
}

impl FffPickerRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_or_fallback(
        &self,
        workspace: &Path,
        cfg: FffRegistryConfig,
    ) -> Result<FffPickerHandle, String> {
        if !cfg.enabled {
            return Ok(FffPickerHandle {
                backend: FffPickerBackend::Fallback,
                picker: make_fallback_picker(workspace)?,
            });
        }
        if cfg.max_workspaces == 0 {
            FFF_PICKER_REGISTRY_CAP_FALLBACK_COUNT.fetch_add(1, Ordering::Relaxed);
            return Ok(FffPickerHandle {
                backend: FffPickerBackend::Fallback,
                picker: make_fallback_picker(workspace)?,
            });
        }

        let key = canonical_workspace(workspace);
        {
            let mut state = self.state.lock();
            if let Some(entry) = state.entries.get(&key) {
                match entry {
                    RegistryEntry::Ready { picker } => {
                        FFF_PICKER_REGISTRY_REUSED_COUNT.fetch_add(1, Ordering::Relaxed);
                        return Ok(FffPickerHandle {
                            backend: FffPickerBackend::FastIndex,
                            picker: picker.clone(),
                        });
                    }
                    RegistryEntry::Pending => {
                        drop(state);
                        return Ok(FffPickerHandle {
                            backend: FffPickerBackend::Fallback,
                            picker: make_fallback_picker(workspace)?,
                        });
                    }
                }
            }
            if state.entries.len() >= cfg.max_workspaces {
                FFF_PICKER_REGISTRY_CAP_FALLBACK_COUNT.fetch_add(1, Ordering::Relaxed);
                drop(state);
                return Ok(FffPickerHandle {
                    backend: FffPickerBackend::Fallback,
                    picker: make_fallback_picker(workspace)?,
                });
            }
            state.entries.insert(key.clone(), RegistryEntry::Pending);
        }

        let picker = match self.make_fast_picker_for_registry(&key, cfg.cache_max_bytes) {
            Ok(picker) => picker,
            Err(err) => {
                self.state.lock().entries.remove(&key);
                return Err(err);
            }
        };

        let mut state = self.state.lock();
        match state.entries.get(&key) {
            Some(RegistryEntry::Pending) => {
                state.entries.insert(
                    key,
                    RegistryEntry::Ready {
                        picker: picker.clone(),
                    },
                );
                self.creation_count.fetch_add(1, Ordering::Relaxed);
                FFF_PICKER_REGISTRY_CREATED_COUNT.fetch_add(1, Ordering::Relaxed);
                Ok(FffPickerHandle {
                    backend: FffPickerBackend::FastIndex,
                    picker,
                })
            }
            Some(RegistryEntry::Ready { picker }) => {
                FFF_PICKER_REGISTRY_REUSED_COUNT.fetch_add(1, Ordering::Relaxed);
                Ok(FffPickerHandle {
                    backend: FffPickerBackend::FastIndex,
                    picker: picker.clone(),
                })
            }
            None => Ok(FffPickerHandle {
                backend: FffPickerBackend::Fallback,
                picker: make_fallback_picker(workspace)?,
            }),
        }
    }

    pub fn indexed_workspaces(&self) -> usize {
        self.state.lock().entries.len()
    }

    pub fn creation_count(&self) -> u64 {
        self.creation_count.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    fn set_test_fast_factory(&self, factory: TestFastFactory) {
        *self.test_fast_factory.lock() = Some(factory);
    }

    fn make_fast_picker_for_registry(
        &self,
        workspace: &Path,
        cache_max_bytes: u64,
    ) -> Result<SharedFilePicker, String> {
        let started = std::time::Instant::now();
        #[cfg(test)]
        let result = if let Some(factory) = self.test_fast_factory.lock().clone() {
            factory(workspace, cache_max_bytes)
        } else {
            make_fast_picker(workspace, cache_max_bytes)
        };
        #[cfg(not(test))]
        let result = make_fast_picker(workspace, cache_max_bytes);
        if result.is_ok() {
            crate::metrics_hist::record_fff_cold_build_duration(
                started.elapsed().as_millis() as u64
            );
        }
        result
    }
}

pub fn same_shared_picker(a: &SharedFilePicker, b: &SharedFilePicker) -> bool {
    match (a.read(), b.read()) {
        (Ok(a_guard), Ok(b_guard)) => match (a_guard.as_ref(), b_guard.as_ref()) {
            (Some(a_picker), Some(b_picker)) => std::ptr::eq(a_picker, b_picker),
            (None, None) => true,
            _ => false,
        },
        _ => false,
    }
}

pub fn canonical_workspace(workspace: &Path) -> PathBuf {
    workspace
        .canonicalize()
        .unwrap_or_else(|_| workspace.to_path_buf())
}

fn make_fallback_picker(workspace: &Path) -> Result<SharedFilePicker, String> {
    let shared_picker = SharedFilePicker::default();
    let picker = FilePicker::new(FilePickerOptions {
        base_path: workspace.to_string_lossy().into_owned(),
        mode: FFFMode::Ai,
        enable_mmap_cache: false,
        enable_content_indexing: false,
        watch: false,
        cache_budget: None,
        ..Default::default()
    })
    .map_err(|e| format!("fff fallback init failed: {e}"))?;
    let mut guard = shared_picker
        .write()
        .map_err(|e| format!("fff fallback lock failed: {e}"))?;
    *guard = Some(picker);
    drop(guard);
    Ok(shared_picker)
}

fn make_fast_picker(workspace: &Path, cache_max_bytes: u64) -> Result<SharedFilePicker, String> {
    let shared_picker = SharedFilePicker::default();
    let cache_budget = ContentCacheBudget::from_overrides(
        DEFAULT_FAST_INDEX_MAX_FILES,
        cache_max_bytes,
        fff_search::grep::MAX_FFFILE_SIZE,
    );
    FilePicker::new_with_shared_state(
        shared_picker.clone(),
        SharedFrecency::noop(),
        FilePickerOptions {
            base_path: workspace.to_string_lossy().into_owned(),
            mode: FFFMode::Ai,
            enable_mmap_cache: true,
            enable_content_indexing: true,
            watch: true,
            cache_budget,
            follow_symlinks: false,
            enable_fs_root_scanning: false,
            enable_home_dir_scanning: false,
        },
    )
    .map_err(|e| format!("fff fast-index init failed: {e}"))?;
    Ok(shared_picker)
}

impl Drop for FffPickerRegistry {
    fn drop(&mut self) {
        let entries: Vec<SharedFilePicker> = self
            .state
            .lock()
            .entries
            .values()
            .filter_map(|entry| match entry {
                RegistryEntry::Ready { picker } => Some(picker.clone()),
                RegistryEntry::Pending => None,
            })
            .collect();
        for picker in entries {
            if let Ok(mut guard) = picker.write()
                && let Some(ref mut picker) = *guard
            {
                picker.stop_background_monitor();
            }
        }
    }
}

pub type SharedFffPickerRegistry = Arc<FffPickerRegistry>;

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(max_workspaces: usize) -> FffRegistryConfig {
        FffRegistryConfig {
            enabled: true,
            max_workspaces,
            cache_max_bytes: 8 * 1024 * 1024,
        }
    }

    #[test]
    fn fff_registry_reuses_picker_for_same_canonical_workspace() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("a.txt"), "needle\n").unwrap();
        let registry = FffPickerRegistry::new();

        let first = registry.get_or_fallback(tmp.path(), cfg(4)).unwrap();
        let second = registry.get_or_fallback(tmp.path(), cfg(4)).unwrap();

        assert_eq!(first.backend, FffPickerBackend::FastIndex);
        assert_eq!(second.backend, FffPickerBackend::FastIndex);
        assert!(same_shared_picker(&first.picker, &second.picker));
        assert_eq!(registry.creation_count(), 1);
        assert_eq!(registry.indexed_workspaces(), 1);
    }

    #[test]
    fn fff_registry_enforces_max_workspaces_and_falls_back() {
        let a = tempfile::tempdir().unwrap();
        let b = tempfile::tempdir().unwrap();
        std::fs::write(a.path().join("a.txt"), "alpha\n").unwrap();
        std::fs::write(b.path().join("b.txt"), "beta\n").unwrap();
        let registry = FffPickerRegistry::new();

        let first = registry.get_or_fallback(a.path(), cfg(1)).unwrap();
        let second = registry.get_or_fallback(b.path(), cfg(1)).unwrap();

        assert_eq!(first.backend, FffPickerBackend::FastIndex);
        assert_eq!(second.backend, FffPickerBackend::Fallback);
        assert_eq!(registry.creation_count(), 1);
        assert_eq!(registry.indexed_workspaces(), 1);
    }

    #[test]
    fn fff_registry_cap_race_does_not_overspawn() {
        let a = tempfile::tempdir().unwrap();
        let b = tempfile::tempdir().unwrap();
        std::fs::write(a.path().join("a.txt"), "alpha\n").unwrap();
        std::fs::write(b.path().join("b.txt"), "beta\n").unwrap();

        let registry = Arc::new(FffPickerRegistry::new());
        let fast_factory_calls = Arc::new(AtomicUsize::new(0));
        let (builder_started_tx, builder_started_rx) = std::sync::mpsc::channel();
        let release_builder = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));

        registry.set_test_fast_factory(Arc::new({
            let fast_factory_calls = Arc::clone(&fast_factory_calls);
            let release_builder = Arc::clone(&release_builder);
            move |workspace, cache_max_bytes| {
                fast_factory_calls.fetch_add(1, Ordering::SeqCst);
                builder_started_tx.send(()).unwrap();

                let (lock, cvar) = &*release_builder;
                let mut released = lock.lock().unwrap();
                while !*released {
                    released = cvar.wait(released).unwrap();
                }
                drop(released);

                make_fast_picker(workspace, cache_max_bytes)
            }
        }));

        let start = Arc::new(std::sync::Barrier::new(3));
        let (result_tx, result_rx) = std::sync::mpsc::channel();
        let paths = [a.path().to_path_buf(), b.path().to_path_buf()];
        let mut threads = Vec::new();
        for path in paths {
            let registry = Arc::clone(&registry);
            let start = Arc::clone(&start);
            let result_tx = result_tx.clone();
            threads.push(std::thread::spawn(move || {
                start.wait();
                let handle = registry.get_or_fallback(&path, cfg(1)).unwrap();
                result_tx.send(handle.backend).unwrap();
            }));
        }
        drop(result_tx);

        start.wait();
        builder_started_rx.recv().unwrap();

        let first_completed = result_rx.recv().unwrap();
        assert_eq!(first_completed, FffPickerBackend::Fallback);
        assert_eq!(fast_factory_calls.load(Ordering::SeqCst), 1);

        {
            let (lock, cvar) = &*release_builder;
            *lock.lock().unwrap() = true;
            cvar.notify_one();
        }

        let second_completed = result_rx.recv().unwrap();
        assert_eq!(second_completed, FffPickerBackend::FastIndex);
        for thread in threads {
            thread.join().unwrap();
        }

        assert_eq!(fast_factory_calls.load(Ordering::SeqCst), 1);
        assert_eq!(registry.creation_count(), 1);
        assert_eq!(registry.indexed_workspaces(), 1);
    }

    #[test]
    fn fff_frecency_noop_no_lmdb_open() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("a.txt"), "needle\n").unwrap();
        let registry = FffPickerRegistry::new();
        let handle = registry.get_or_fallback(tmp.path(), cfg(4)).unwrap();
        assert_eq!(handle.backend, FffPickerBackend::FastIndex);

        let lmdb_like = ["data.mdb", "lock.mdb", "frecency.mdb", "frecency"];
        for name in lmdb_like {
            assert!(
                !tmp.path().join(name).exists(),
                "SharedFrecency::noop() must not create LMDB file {name}"
            );
        }
    }
}
