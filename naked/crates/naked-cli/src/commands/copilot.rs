//! `naked copilot-login` subcommand — GitHub Copilot OAuth device login.

use std::io::{self, BufRead, Write};

use anyhow::Result;

pub(crate) async fn copilot_login_cmd() -> Result<()> {
    use naked_core::provider::copilot;

    eprintln!("GitHub Copilot — OAuth Device Login\n");

    if let Some(token) = copilot::load_copilot_token() {
        let masked = if token.len() > 8 {
            format!("{}…{}", &token[..4], &token[token.len() - 4..])
        } else {
            "****".to_string()
        };
        eprintln!("Existing token found: {masked}");
        eprint!("Re-authenticate? [y/N]: ");
        io::stderr().flush()?;
        let mut answer = String::new();
        let _ = io::stdin().lock().read_line(&mut answer);
        if !matches!(answer.trim().to_lowercase().as_str(), "y" | "yes") {
            eprintln!("Keeping existing token.");

            eprintln!("\nFetching available models...");
            match copilot::fetch_copilot_models(&token).await {
                Ok(models) if !models.is_empty() => {
                    eprintln!("Available Copilot models:");
                    for m in &models {
                        eprintln!("  • {m}");
                    }
                }
                Ok(_) => eprintln!("No models returned (token may be expired)."),
                Err(e) => eprintln!("Could not fetch models: {e}"),
            }
            return Ok(());
        }
    }

    let token = copilot::copilot_device_login().await?;

    eprintln!("\nFetching available models...");
    match copilot::fetch_copilot_models(&token).await {
        Ok(models) if !models.is_empty() => {
            eprintln!("Available Copilot models:");
            for m in &models {
                eprintln!("  • {m}");
            }
        }
        Ok(_) => eprintln!("No models returned yet — try again shortly."),
        Err(e) => eprintln!("Could not fetch models: {e}"),
    }

    Ok(())
}
