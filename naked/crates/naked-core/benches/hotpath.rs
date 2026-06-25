use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use naked_core::tool::Tool;
use naked_core::tool::fff_tools::{FffFindTool, FffGrepTool, FffState};
use naked_core::tool::file_ops::{EditFileTool, ReadFileTool};
use serde_json::json;
use tempfile::TempDir;
use tokio::runtime::Runtime;

const SMALL_FILE: &str = "src/module_07.rs";
const LARGE_FILE: &str = "logs/large_fixture.txt";
const EDIT_SINGLE_FILE: &str = "edit/single.txt";
const EDIT_MULTI_FILE: &str = "edit/multi.txt";

struct HotpathFixture {
    _tempdir: TempDir,
    root: PathBuf,
    edit_single_original: String,
    edit_multi_original: String,
    next_edit_id: AtomicU64,
}

impl HotpathFixture {
    fn new() -> Self {
        let tempdir = tempfile::tempdir().expect("create benchmark tempdir");
        let root = tempdir.path().to_path_buf();

        write_fixture_tree(&root);

        let edit_single_original =
            std::fs::read_to_string(root.join(EDIT_SINGLE_FILE)).expect("read single edit fixture");
        let edit_multi_original =
            std::fs::read_to_string(root.join(EDIT_MULTI_FILE)).expect("read multi edit fixture");

        Self {
            _tempdir: tempdir,
            root,
            edit_single_original,
            edit_multi_original,
            next_edit_id: AtomicU64::new(0),
        }
    }

    fn fresh_single_edit_file(&self) -> String {
        let relative_path = self.fresh_edit_path("single");
        std::fs::write(self.root.join(&relative_path), &self.edit_single_original)
            .expect("write fresh single edit fixture");
        relative_path
    }

    fn fresh_multi_edit_file(&self) -> String {
        let relative_path = self.fresh_edit_path("multi");
        std::fs::write(self.root.join(&relative_path), &self.edit_multi_original)
            .expect("write fresh multi edit fixture");
        relative_path
    }

    fn fresh_edit_path(&self, stem: &str) -> String {
        let id = self.next_edit_id.fetch_add(1, Ordering::Relaxed);
        format!("edit/{stem}_{id}.txt")
    }
}

fn write_fixture_tree(root: &Path) {
    std::fs::create_dir_all(root.join("src")).expect("create src dir");
    std::fs::create_dir_all(root.join("notes")).expect("create notes dir");
    std::fs::create_dir_all(root.join("logs")).expect("create logs dir");
    std::fs::create_dir_all(root.join("edit")).expect("create edit dir");

    for i in 0..48 {
        let module = format!(
            "pub fn fixture_fn_{i:02}() -> &'static str {{\n    \"needle_alpha module_{i:02}\"\n}}\n\n#[cfg(test)]\nmod tests {{\n    #[test]\n    fn smoke_{i:02}() {{\n        assert!(super::fixture_fn_{i:02}().contains(\"needle_alpha\"));\n    }}\n}}\n"
        );
        std::fs::write(root.join(format!("src/module_{i:02}.rs")), module)
            .expect("write rust module fixture");

        let note = format!(
            "title: deterministic note {i:02}\nbody: beta_token_{i:02} and common_search_term\n"
        );
        std::fs::write(root.join(format!("notes/note_{i:02}.txt")), note)
            .expect("write note fixture");
    }

    let mut large = String::new();
    for i in 0..4_096 {
        large.push_str(&format!(
            "{i:04}: large fixture row with needle_alpha and stable payload for read benchmarks\n"
        ));
    }
    std::fs::write(root.join(LARGE_FILE), large).expect("write large fixture");

    std::fs::write(
        root.join(EDIT_SINGLE_FILE),
        "alpha = 1\nbeta = 2\ngamma = 3\n",
    )
    .expect("write single edit fixture");

    std::fs::write(
        root.join(EDIT_MULTI_FILE),
        "first = red\nsecond = green\nthird = blue\nfourth = yellow\n",
    )
    .expect("write multi edit fixture");
}

fn assert_tool_ok(result: naked_core::types::ToolResult) {
    assert!(!result.is_error, "tool returned error: {}", result.output);
    assert!(!result.output.is_empty(), "tool returned empty output");
}

fn bench_read_file(c: &mut Criterion) {
    let runtime = Runtime::new().expect("create tokio runtime");
    let fixture = HotpathFixture::new();
    let tool = ReadFileTool::default();
    let cwd = fixture.root.as_path();
    let tool_ref: &dyn Tool = &tool;

    let mut group = c.benchmark_group("read_file");
    group.bench_function("small", |b| {
        b.to_async(&runtime).iter(|| async {
            assert_tool_ok(
                tool_ref
                    .execute(json!({ "file_path": SMALL_FILE }), cwd)
                    .await,
            );
        });
    });
    group.bench_function("large", |b| {
        b.to_async(&runtime).iter(|| async {
            assert_tool_ok(
                tool_ref
                    .execute(json!({ "file_path": LARGE_FILE }), cwd)
                    .await,
            );
        });
    });
    group.finish();
}

fn bench_edit_file(c: &mut Criterion) {
    let runtime = Runtime::new().expect("create tokio runtime");
    let fixture = HotpathFixture::new();
    let tool = EditFileTool::default();
    let cwd = fixture.root.as_path();
    let tool_ref: &dyn Tool = &tool;

    let mut group = c.benchmark_group("edit_file");
    group.bench_function("single", |b| {
        b.to_async(&runtime).iter_batched(
            || fixture.fresh_single_edit_file(),
            |file_path| async move {
                assert_tool_ok(
                    tool_ref
                        .execute(
                            json!({
                                "file_path": file_path,
                                "old_string": "beta = 2",
                                "new_string": "beta = 20"
                            }),
                            cwd,
                        )
                        .await,
                );
            },
            BatchSize::SmallInput,
        );
    });
    group.bench_function("multi", |b| {
        b.to_async(&runtime).iter_batched(
            || fixture.fresh_multi_edit_file(),
            |file_path| async move {
                assert_tool_ok(
                    tool_ref
                        .execute(
                            json!({
                                "file_path": file_path,
                                "edits": [
                                    { "old_string": "first = red", "new_string": "first = crimson" },
                                    { "old_string": "third = blue", "new_string": "third = azure" }
                                ]
                            }),
                            cwd,
                        )
                        .await,
                );
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn bench_grep(c: &mut Criterion) {
    let runtime = Runtime::new().expect("create tokio runtime");
    let fixture = HotpathFixture::new();
    let state = FffState::new(&fixture.root);
    let tool = FffGrepTool::new(&state);
    let tool_ref: &dyn Tool = &tool;
    let cwd = fixture.root.as_path();

    let mut group = c.benchmark_group("grep");
    group.bench_function("fff_registered_default", |b| {
        b.to_async(&runtime).iter(|| async {
            assert_tool_ok(
                tool_ref
                    .execute(json!({ "query": "needle_alpha", "max_results": 30 }), cwd)
                    .await,
            );
        });
    });
    group.finish();
}

fn bench_find(c: &mut Criterion) {
    let runtime = Runtime::new().expect("create tokio runtime");
    let fixture = HotpathFixture::new();
    let state = FffState::new(&fixture.root);
    let tool = FffFindTool::new(&state);
    let tool_ref: &dyn Tool = &tool;
    let cwd = fixture.root.as_path();

    let mut group = c.benchmark_group("find");
    group.bench_function("fff_registered_default", |b| {
        b.to_async(&runtime).iter(|| async {
            assert_tool_ok(
                tool_ref
                    .execute(json!({ "query": "module_07", "max_results": 20 }), cwd)
                    .await,
            );
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_read_file,
    bench_edit_file,
    bench_grep,
    bench_find
);
criterion_main!(benches);
