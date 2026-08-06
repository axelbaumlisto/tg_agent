use std::sync::atomic::{AtomicU64, Ordering};

static TURN_UNDER_1S: AtomicU64 = AtomicU64::new(0);
static TURN_1S_TO_10S: AtomicU64 = AtomicU64::new(0);
static TURN_10S_TO_60S: AtomicU64 = AtomicU64::new(0);
static TURN_OVER_60S: AtomicU64 = AtomicU64::new(0);
static TURN_SUM_MS: AtomicU64 = AtomicU64::new(0);
static TURN_COUNT: AtomicU64 = AtomicU64::new(0);

static TTFT_UNDER_500MS: AtomicU64 = AtomicU64::new(0);
static TTFT_500MS_TO_2S: AtomicU64 = AtomicU64::new(0);
static TTFT_2S_TO_10S: AtomicU64 = AtomicU64::new(0);
static TTFT_OVER_10S: AtomicU64 = AtomicU64::new(0);
static TTFT_SUM_MS: AtomicU64 = AtomicU64::new(0);
static TTFT_COUNT: AtomicU64 = AtomicU64::new(0);

static PROVIDER_UNDER_500MS: AtomicU64 = AtomicU64::new(0);
static PROVIDER_500MS_TO_2S: AtomicU64 = AtomicU64::new(0);
static PROVIDER_2S_TO_10S: AtomicU64 = AtomicU64::new(0);
static PROVIDER_OVER_10S: AtomicU64 = AtomicU64::new(0);
static PROVIDER_SUM_MS: AtomicU64 = AtomicU64::new(0);
static PROVIDER_COUNT: AtomicU64 = AtomicU64::new(0);

static TOOL_GREP: DurationAtomics = DurationAtomics::new();
static TOOL_READ: DurationAtomics = DurationAtomics::new();
static TOOL_EDIT: DurationAtomics = DurationAtomics::new();
static TOOL_WRITE: DurationAtomics = DurationAtomics::new();
static TOOL_BASH: DurationAtomics = DurationAtomics::new();
static TOOL_APPLY_PATCH: DurationAtomics = DurationAtomics::new();
static TOOL_OTHER: DurationAtomics = DurationAtomics::new();
static FFF_COLD_BUILD: DurationAtomics = DurationAtomics::new();
static PARENT_FSYNC: DurationAtomics = DurationAtomics::new();

const TURN_EDGES_MS: [u64; 3] = [1_000, 10_000, 60_000];
const SHORT_EDGES_MS: [u64; 3] = [500, 2_000, 10_000];
pub const TOOL_EDGES_MS: [u64; 3] = [10, 100, 1_000];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LatencySnapshot {
    pub turn_under_1s: u64,
    pub turn_1s_to_10s: u64,
    pub turn_10s_to_60s: u64,
    pub turn_over_60s: u64,
    pub turn_sum_ms: u64,
    pub turn_count: u64,
    pub ttft_under_500ms: u64,
    pub ttft_500ms_to_2s: u64,
    pub ttft_2s_to_10s: u64,
    pub ttft_over_10s: u64,
    pub ttft_sum_ms: u64,
    pub ttft_count: u64,
    pub provider_under_500ms: u64,
    pub provider_500ms_to_2s: u64,
    pub provider_2s_to_10s: u64,
    pub provider_over_10s: u64,
    pub provider_sum_ms: u64,
    pub provider_count: u64,
    pub tool_grep: DurationHistogramSnapshot,
    pub tool_read: DurationHistogramSnapshot,
    pub tool_edit: DurationHistogramSnapshot,
    pub tool_write: DurationHistogramSnapshot,
    pub tool_bash: DurationHistogramSnapshot,
    pub tool_apply_patch: DurationHistogramSnapshot,
    pub tool_other: DurationHistogramSnapshot,
    pub fff_cold_build: DurationHistogramSnapshot,
    pub parent_fsync: DurationHistogramSnapshot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DurationHistogramSnapshot {
    pub under_10ms: u64,
    pub ms_10_to_100: u64,
    pub ms_100_to_1s: u64,
    pub over_1s: u64,
    pub sum_ms: u64,
    pub count: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolClass {
    Grep,
    Read,
    Edit,
    Write,
    Bash,
    ApplyPatch,
    Other,
}

impl ToolClass {
    pub const ALL: [ToolClass; 7] = [
        ToolClass::Grep,
        ToolClass::Read,
        ToolClass::Edit,
        ToolClass::Write,
        ToolClass::Bash,
        ToolClass::ApplyPatch,
        ToolClass::Other,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            ToolClass::Grep => "grep",
            ToolClass::Read => "read",
            ToolClass::Edit => "edit",
            ToolClass::Write => "write",
            ToolClass::Bash => "bash",
            ToolClass::ApplyPatch => "apply_patch",
            ToolClass::Other => "other",
        }
    }
}

struct DurationAtomics {
    under_10ms: AtomicU64,
    ms_10_to_100: AtomicU64,
    ms_100_to_1s: AtomicU64,
    over_1s: AtomicU64,
    sum_ms: AtomicU64,
    count: AtomicU64,
}

impl DurationAtomics {
    const fn new() -> Self {
        Self {
            under_10ms: AtomicU64::new(0),
            ms_10_to_100: AtomicU64::new(0),
            ms_100_to_1s: AtomicU64::new(0),
            over_1s: AtomicU64::new(0),
            sum_ms: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }

    fn record(&self, ms: u64) {
        bump(
            [
                &self.under_10ms,
                &self.ms_10_to_100,
                &self.ms_100_to_1s,
                &self.over_1s,
            ],
            &self.sum_ms,
            &self.count,
            ms,
            TOOL_EDGES_MS,
        );
    }

    fn snapshot(&self) -> DurationHistogramSnapshot {
        DurationHistogramSnapshot {
            under_10ms: self.under_10ms.load(Ordering::Relaxed),
            ms_10_to_100: self.ms_10_to_100.load(Ordering::Relaxed),
            ms_100_to_1s: self.ms_100_to_1s.load(Ordering::Relaxed),
            over_1s: self.over_1s.load(Ordering::Relaxed),
            sum_ms: self.sum_ms.load(Ordering::Relaxed),
            count: self.count.load(Ordering::Relaxed),
        }
    }
}

pub fn classify_tool_name(name: &str) -> ToolClass {
    match name {
        "grep" | "grep_search" => ToolClass::Grep,
        "read" | "read_file" | "file_snapshot" => ToolClass::Read,
        "edit" | "edit_file" => ToolClass::Edit,
        "write" | "write_file" => ToolClass::Write,
        "bash" => ToolClass::Bash,
        "apply_patch" => ToolClass::ApplyPatch,
        _ => ToolClass::Other,
    }
}

pub fn record_turn_duration(ms: u64) {
    bump(
        [
            &TURN_UNDER_1S,
            &TURN_1S_TO_10S,
            &TURN_10S_TO_60S,
            &TURN_OVER_60S,
        ],
        &TURN_SUM_MS,
        &TURN_COUNT,
        ms,
        TURN_EDGES_MS,
    );
}

pub fn record_ttft(ms: u64) {
    bump(
        [
            &TTFT_UNDER_500MS,
            &TTFT_500MS_TO_2S,
            &TTFT_2S_TO_10S,
            &TTFT_OVER_10S,
        ],
        &TTFT_SUM_MS,
        &TTFT_COUNT,
        ms,
        SHORT_EDGES_MS,
    );
}

pub fn record_provider_stream_open(ms: u64) {
    bump(
        [
            &PROVIDER_UNDER_500MS,
            &PROVIDER_500MS_TO_2S,
            &PROVIDER_2S_TO_10S,
            &PROVIDER_OVER_10S,
        ],
        &PROVIDER_SUM_MS,
        &PROVIDER_COUNT,
        ms,
        SHORT_EDGES_MS,
    );
}

pub fn record_tool_duration(tool_name: &str, ms: u64) {
    duration_atomics_for_tool(classify_tool_name(tool_name)).record(ms);
}

pub fn record_fff_cold_build_duration(ms: u64) {
    FFF_COLD_BUILD.record(ms);
}

pub fn record_parent_fsync_duration(ms: u64) {
    PARENT_FSYNC.record(ms);
}

pub fn tool_snapshot(snapshot: &LatencySnapshot, class: ToolClass) -> DurationHistogramSnapshot {
    match class {
        ToolClass::Grep => snapshot.tool_grep,
        ToolClass::Read => snapshot.tool_read,
        ToolClass::Edit => snapshot.tool_edit,
        ToolClass::Write => snapshot.tool_write,
        ToolClass::Bash => snapshot.tool_bash,
        ToolClass::ApplyPatch => snapshot.tool_apply_patch,
        ToolClass::Other => snapshot.tool_other,
    }
}

pub fn snapshot() -> LatencySnapshot {
    LatencySnapshot {
        turn_under_1s: TURN_UNDER_1S.load(Ordering::Relaxed),
        turn_1s_to_10s: TURN_1S_TO_10S.load(Ordering::Relaxed),
        turn_10s_to_60s: TURN_10S_TO_60S.load(Ordering::Relaxed),
        turn_over_60s: TURN_OVER_60S.load(Ordering::Relaxed),
        turn_sum_ms: TURN_SUM_MS.load(Ordering::Relaxed),
        turn_count: TURN_COUNT.load(Ordering::Relaxed),
        ttft_under_500ms: TTFT_UNDER_500MS.load(Ordering::Relaxed),
        ttft_500ms_to_2s: TTFT_500MS_TO_2S.load(Ordering::Relaxed),
        ttft_2s_to_10s: TTFT_2S_TO_10S.load(Ordering::Relaxed),
        ttft_over_10s: TTFT_OVER_10S.load(Ordering::Relaxed),
        ttft_sum_ms: TTFT_SUM_MS.load(Ordering::Relaxed),
        ttft_count: TTFT_COUNT.load(Ordering::Relaxed),
        provider_under_500ms: PROVIDER_UNDER_500MS.load(Ordering::Relaxed),
        provider_500ms_to_2s: PROVIDER_500MS_TO_2S.load(Ordering::Relaxed),
        provider_2s_to_10s: PROVIDER_2S_TO_10S.load(Ordering::Relaxed),
        provider_over_10s: PROVIDER_OVER_10S.load(Ordering::Relaxed),
        provider_sum_ms: PROVIDER_SUM_MS.load(Ordering::Relaxed),
        provider_count: PROVIDER_COUNT.load(Ordering::Relaxed),
        tool_grep: TOOL_GREP.snapshot(),
        tool_read: TOOL_READ.snapshot(),
        tool_edit: TOOL_EDIT.snapshot(),
        tool_write: TOOL_WRITE.snapshot(),
        tool_bash: TOOL_BASH.snapshot(),
        tool_apply_patch: TOOL_APPLY_PATCH.snapshot(),
        tool_other: TOOL_OTHER.snapshot(),
        fff_cold_build: FFF_COLD_BUILD.snapshot(),
        parent_fsync: PARENT_FSYNC.snapshot(),
    }
}

fn duration_atomics_for_tool(class: ToolClass) -> &'static DurationAtomics {
    match class {
        ToolClass::Grep => &TOOL_GREP,
        ToolClass::Read => &TOOL_READ,
        ToolClass::Edit => &TOOL_EDIT,
        ToolClass::Write => &TOOL_WRITE,
        ToolClass::Bash => &TOOL_BASH,
        ToolClass::ApplyPatch => &TOOL_APPLY_PATCH,
        ToolClass::Other => &TOOL_OTHER,
    }
}

fn bump(buckets: [&AtomicU64; 4], sum: &AtomicU64, count: &AtomicU64, ms: u64, edges: [u64; 3]) {
    let bucket_idx = if ms < edges[0] {
        0
    } else if ms < edges[1] {
        1
    } else if ms < edges[2] {
        2
    } else {
        3
    };

    buckets[bucket_idx].fetch_add(1, Ordering::Relaxed);
    sum.fetch_add(ms, Ordering::Relaxed);
    count.fetch_add(1, Ordering::Relaxed);
}

#[cfg(test)]
pub(crate) fn test_guard() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
pub(crate) async fn async_test_guard() -> tokio::sync::MutexGuard<'static, ()> {
    static LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
    LOCK.lock().await
}

#[cfg(test)]
pub(crate) fn reset_for_test() {
    for atomic in all_atomics() {
        atomic.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
fn all_atomics() -> Vec<&'static AtomicU64> {
    let mut atomics = vec![
        &TURN_UNDER_1S,
        &TURN_1S_TO_10S,
        &TURN_10S_TO_60S,
        &TURN_OVER_60S,
        &TURN_SUM_MS,
        &TURN_COUNT,
        &TTFT_UNDER_500MS,
        &TTFT_500MS_TO_2S,
        &TTFT_2S_TO_10S,
        &TTFT_OVER_10S,
        &TTFT_SUM_MS,
        &TTFT_COUNT,
        &PROVIDER_UNDER_500MS,
        &PROVIDER_500MS_TO_2S,
        &PROVIDER_2S_TO_10S,
        &PROVIDER_OVER_10S,
        &PROVIDER_SUM_MS,
        &PROVIDER_COUNT,
    ];
    for hist in [
        &TOOL_GREP,
        &TOOL_READ,
        &TOOL_EDIT,
        &TOOL_WRITE,
        &TOOL_BASH,
        &TOOL_APPLY_PATCH,
        &TOOL_OTHER,
        &FFF_COLD_BUILD,
        &PARENT_FSYNC,
    ] {
        atomics.extend([
            &hist.under_10ms,
            &hist.ms_10_to_100,
            &hist.ms_100_to_1s,
            &hist.over_1s,
            &hist.sum_ms,
            &hist.count,
        ]);
    }
    atomics
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turn_duration_buckets_all_edges_and_sum_count() {
        let _guard = test_guard();
        reset_for_test();

        for ms in [
            999, 1000, 1001, 9999, 10_000, 10_001, 59_999, 60_000, 60_001,
        ] {
            record_turn_duration(ms);
        }

        let snapshot = snapshot();
        assert_eq!(snapshot.turn_under_1s, 1);
        assert_eq!(snapshot.turn_1s_to_10s, 3);
        assert_eq!(snapshot.turn_10s_to_60s, 3);
        assert_eq!(snapshot.turn_over_60s, 2);
        assert_eq!(snapshot.turn_sum_ms, 213_000);
        assert_eq!(snapshot.turn_count, 9);
    }

    #[test]
    fn ttft_buckets_all_edges_and_sum_count() {
        let _guard = test_guard();
        reset_for_test();

        for ms in [499, 500, 501, 1999, 2000, 2001, 9999, 10_000, 10_001] {
            record_ttft(ms);
        }

        let snapshot = snapshot();
        assert_eq!(snapshot.ttft_under_500ms, 1);
        assert_eq!(snapshot.ttft_500ms_to_2s, 3);
        assert_eq!(snapshot.ttft_2s_to_10s, 3);
        assert_eq!(snapshot.ttft_over_10s, 2);
        assert_eq!(snapshot.ttft_sum_ms, 37_500);
        assert_eq!(snapshot.ttft_count, 9);
    }

    #[test]
    fn provider_stream_open_buckets_all_edges_and_sum_count() {
        let _guard = test_guard();
        reset_for_test();

        for ms in [499, 500, 501, 1999, 2000, 2001, 9999, 10_000, 10_001] {
            record_provider_stream_open(ms);
        }

        let snapshot = snapshot();
        assert_eq!(snapshot.provider_under_500ms, 1);
        assert_eq!(snapshot.provider_500ms_to_2s, 3);
        assert_eq!(snapshot.provider_2s_to_10s, 3);
        assert_eq!(snapshot.provider_over_10s, 2);
        assert_eq!(snapshot.provider_sum_ms, 37_500);
        assert_eq!(snapshot.provider_count, 9);
    }

    #[test]
    fn tool_duration_buckets_all_edges_and_sum_count() {
        let _guard = test_guard();
        reset_for_test();

        for ms in [9, 10, 11, 99, 100, 101, 999, 1000, 1001] {
            record_tool_duration("grep_search", ms);
        }

        let hist = snapshot().tool_grep;
        assert_eq!(hist.under_10ms, 1);
        assert_eq!(hist.ms_10_to_100, 3);
        assert_eq!(hist.ms_100_to_1s, 3);
        assert_eq!(hist.over_1s, 2);
        assert_eq!(hist.sum_ms, 3330);
        assert_eq!(hist.count, 9);
    }

    #[test]
    fn fff_cold_build_buckets_all_edges_and_sum_count() {
        let _guard = test_guard();
        reset_for_test();

        for ms in [9, 10, 99, 100, 999, 1000] {
            record_fff_cold_build_duration(ms);
        }

        let hist = snapshot().fff_cold_build;
        assert_eq!(hist.under_10ms, 1);
        assert_eq!(hist.ms_10_to_100, 2);
        assert_eq!(hist.ms_100_to_1s, 2);
        assert_eq!(hist.over_1s, 1);
        assert_eq!(hist.sum_ms, 2217);
        assert_eq!(hist.count, 6);
    }

    #[test]
    fn parent_fsync_buckets_all_edges_and_sum_count() {
        // The other six bucket tests take this guard; this one did not, so it
        // zeroed the process-global counters underneath whichever of them was
        // running in parallel and then asserted exact totals. Rare but real
        // (observed once in a full-suite run).
        let _guard = test_guard();
        reset_for_test();

        for ms in [0, 9, 10, 99, 100, 999, 1000, 1001] {
            record_parent_fsync_duration(ms);
        }

        let hist = snapshot().parent_fsync;
        assert_eq!(hist.under_10ms, 2);
        assert_eq!(hist.ms_10_to_100, 2);
        assert_eq!(hist.ms_100_to_1s, 2);
        assert_eq!(hist.over_1s, 2);
        assert_eq!(hist.sum_ms, 3218);
        assert_eq!(hist.count, 8);
    }

    #[test]
    fn tool_name_classification_is_bounded() {
        assert_eq!(classify_tool_name("grep_search"), ToolClass::Grep);
        assert_eq!(classify_tool_name("read_file"), ToolClass::Read);
        assert_eq!(classify_tool_name("file_snapshot"), ToolClass::Read);
        assert_eq!(classify_tool_name("edit_file"), ToolClass::Edit);
        assert_eq!(classify_tool_name("write_file"), ToolClass::Write);
        assert_eq!(classify_tool_name("bash"), ToolClass::Bash);
        assert_eq!(classify_tool_name("apply_patch"), ToolClass::ApplyPatch);
        assert_eq!(classify_tool_name("some_weird_tool"), ToolClass::Other);
    }
}
