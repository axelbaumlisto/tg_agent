mod commands;
mod research;

use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("naked=info".parse().expect("static directive")),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();
    commands::run(args).await
}
