use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use tokio::fs::{self, File};
use tokio::io::AsyncWriteExt;

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Durably replace `path` with `contents` using a same-directory temporary file.
///
/// The temp file is created with `create_new(true)` to avoid clobbering a
/// concurrent writer's temp. Any error before the final rename, and any rename
/// error, removes the temp file best-effort before returning the original error.
pub(crate) async fn atomic_replace_file(path: &Path, contents: &[u8]) -> io::Result<()> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(Path::to_path_buf);
    let (tmp, mut tmp_file) = create_unique_temp_file(path).await?;

    let write_result = async {
        tmp_file.write_all(contents).await?;
        tmp_file.flush().await?;
        if let Err(e) = tmp_file.sync_data().await {
            tracing::warn!(path = %tmp.display(), error = %e, "temp file sync_data failed; continuing atomic replace");
        }
        drop(tmp_file);
        fs::rename(&tmp, path).await?;
        if let Some(parent) = parent {
            fsync_parent_dir_best_effort(parent).await;
        }
        io::Result::Ok(())
    }
    .await;

    if let Err(e) = write_result {
        let _ = fs::remove_file(&tmp).await;
        return Err(e);
    }

    Ok(())
}

async fn create_unique_temp_file(path: &Path) -> io::Result<(PathBuf, File)> {
    let mut last_error = None;
    for _ in 0..16 {
        let tmp = unique_sibling_path(path);
        match fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)
            .await
        {
            Ok(file) => return Ok((tmp, file)),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {
                last_error = Some(e);
            }
            Err(e) => return Err(e),
        }
    }
    Err(last_error.unwrap_or_else(|| {
        io::Error::new(
            io::ErrorKind::AlreadyExists,
            "failed to allocate unique temporary file name",
        )
    }))
}

fn unique_sibling_path(path: &Path) -> PathBuf {
    let now_nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let pid = std::process::id();
    let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("file");
    path.with_file_name(format!(".{file_name}.tmp.{pid}.{counter}.{now_nanos}"))
}

#[cfg(unix)]
async fn fsync_parent_dir_best_effort(parent: PathBuf) {
    let display_path = parent.display().to_string();
    let started = std::time::Instant::now();
    match tokio::task::spawn_blocking(move || {
        #[cfg(test)]
        run_parent_fsync_test_hook(&parent);
        std::fs::File::open(&parent).and_then(|dir| dir.sync_all())
    })
    .await
    {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            tracing::warn!(path = %display_path, error = %e, "parent directory fsync failed; continuing atomic replace");
        }
        Err(e) => {
            tracing::warn!(path = %display_path, error = %e, "parent directory fsync task failed; continuing atomic replace");
        }
    }
    crate::metrics_hist::record_parent_fsync_duration(started.elapsed().as_millis() as u64);
}

#[cfg(not(unix))]
async fn fsync_parent_dir_best_effort(_parent: PathBuf) {}

#[cfg(all(test, unix))]
type ParentFsyncTestHook = std::sync::Arc<dyn Fn(&Path) + Send + Sync>;

#[cfg(all(test, unix))]
type ParentFsyncTestHookSlot = std::sync::Mutex<Option<ParentFsyncTestHook>>;

#[cfg(all(test, unix))]
static PARENT_FSYNC_TEST_HOOK: ParentFsyncTestHookSlot = std::sync::Mutex::new(None);

#[cfg(all(test, unix))]
fn run_parent_fsync_test_hook(parent: &Path) {
    let hook = PARENT_FSYNC_TEST_HOOK
        .lock()
        .expect("parent fsync test hook mutex poisoned")
        .clone();
    if let Some(hook) = hook {
        hook(parent);
    }
}

#[cfg(all(test, unix))]
fn set_parent_fsync_test_hook(hook: Option<ParentFsyncTestHook>) {
    *PARENT_FSYNC_TEST_HOOK
        .lock()
        .expect("parent fsync test hook mutex poisoned") = hook;
}

#[cfg(test)]
mod tests {
    #[cfg(unix)]
    #[tokio::test]
    async fn atomic_replace_awaits_parent_fsync_after_rename() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("atomic.txt");
        std::fs::write(&path, b"old bytes").unwrap();

        let entered = std::sync::Arc::new(std::sync::Barrier::new(2));
        let release = std::sync::Arc::new(std::sync::Barrier::new(2));
        let expected_parent = dir.path().to_path_buf();
        let hook_entered = std::sync::Arc::clone(&entered);
        let hook_release = std::sync::Arc::clone(&release);

        super::set_parent_fsync_test_hook(Some(std::sync::Arc::new(move |parent| {
            if parent == expected_parent.as_path() {
                hook_entered.wait();
                hook_release.wait();
            }
        })));

        let write_path = path.clone();
        let write_task = tokio::spawn(async move {
            super::atomic_replace_file(&write_path, b"new complete bytes").await
        });

        tokio::task::spawn_blocking(move || entered.wait())
            .await
            .unwrap();

        assert_eq!(std::fs::read(&path).unwrap(), b"new complete bytes");
        tokio::task::yield_now().await;
        assert!(
            !write_task.is_finished(),
            "atomic_replace_file returned before awaited parent fsync completed"
        );

        tokio::task::spawn_blocking(move || release.wait())
            .await
            .unwrap();
        write_task.await.unwrap().unwrap();
        super::set_parent_fsync_test_hook(None);
    }
}
