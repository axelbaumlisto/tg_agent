//! Standalone smoke-test for the multi-engine search wiring.
//!
//! Reads `~/.naked/secrets/search_pool.alive.json`, builds the same
//! [`MultiEngineSearch`] the bot uses, and runs a single query against
//! every wired engine. Prints per-engine hit counts and the first hit.
//!
//! Run:
//!     cargo run -p naked-core --example search_smoke -- "cho thuê mặt bằng An Thượng"

use std::sync::Arc;
use std::time::Duration;

use naked_core::keys::KeyProvider;
use naked_core::keys::fs::FilesystemKeyProvider;
use naked_core::keys::pool::KeyPool;
use naked_core::search::SearchEngine;
use naked_core::search::ddg::DdgEngine;
use naked_core::search::exa::ExaEngine;
use naked_core::search::multi::MultiEngineSearch;
use naked_core::search::serpapi::SerpApiEngine;
use naked_core::search::snippet::SnippetExtractor;
use naked_core::search::tavily::TavilyEngine;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,naked_core=debug")),
        )
        .with_writer(std::io::stderr)
        .init();

    let query = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "cho thuê mặt bằng An Thượng Đà Nẵng".into());

    let fs_provider: Arc<dyn KeyProvider> = Arc::new(FilesystemKeyProvider::new(
        FilesystemKeyProvider::default_path(),
    ));
    let ttl = Duration::from_secs(3600);

    let exa_pool = Arc::new(KeyPool::new(vec![fs_provider.clone()], "exa", ttl));
    let tavily_pool = Arc::new(KeyPool::new(vec![fs_provider.clone()], "tavily", ttl));
    let serpapi_pool = Arc::new(KeyPool::new(vec![fs_provider.clone()], "serpapi", ttl));

    println!("== pool sizes ==");
    println!("  exa:     {}", exa_pool.size());
    println!("  tavily:  {}", tavily_pool.size());
    println!("  serpapi: {}", serpapi_pool.size());

    let mut engines: Vec<Box<dyn SearchEngine>> = Vec::new();
    if exa_pool.size() > 0 {
        engines.push(Box::new(ExaEngine::new(exa_pool)));
    }
    if tavily_pool.size() > 0 {
        engines.push(Box::new(TavilyEngine::new(tavily_pool)));
    }
    if serpapi_pool.size() > 0 {
        engines.push(Box::new(SerpApiEngine::new(serpapi_pool)));
    }
    engines.push(Box::new(DdgEngine::new()));

    let names: Vec<_> = engines.iter().map(|e| e.name()).collect();
    println!("\n== engines wired: {names:?} ==");
    println!("== query: {query:?} ==\n");

    let multi = MultiEngineSearch::new(engines);
    let t0 = std::time::Instant::now();
    let hits = multi.search(&query, 5).await;
    let elapsed = t0.elapsed();

    let mut by_engine: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for h in &hits {
        *by_engine.entry(h.source_engine).or_default() += 1;
    }

    println!("== {} hits in {:?} ==", hits.len(), elapsed);
    for (eng, n) in &by_engine {
        println!("  {eng}: {n}");
    }

    if let Some(first) = hits.first() {
        println!(
            "\n== first hit ({}) ==\n  url:     {}\n  title:   {}\n  snippet: {}",
            first.source_engine,
            first.url,
            first.title.chars().take(80).collect::<String>(),
            first.snippet.chars().take(160).collect::<String>(),
        );
    }

    println!("\n== snippet fast-path scan ==");
    let extractor = SnippetExtractor::new();
    let mut fast = 0usize;
    for h in &hits {
        let combined = format!("{} {}", h.title, h.snippet);
        if let Some(f) = extractor.extract(&combined) {
            fast += 1;
            println!(
                "  ⚡ {} | {} VND | {} m² | {} | phone={}",
                h.source_engine,
                f.price_vnd_per_month,
                f.area_m2,
                f.district,
                f.phone.as_deref().unwrap_or("—"),
            );
        }
    }
    println!(
        "  {fast}/{} hits fully-extractable from snippet",
        hits.len()
    );
}
