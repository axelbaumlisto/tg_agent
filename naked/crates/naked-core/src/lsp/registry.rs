//! Language detection + default server commands.

use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Language {
    Rust,
    Python,
    TypeScript,
    JavaScript,
    Go,
    C,
    Cpp,
}

impl Language {
    /// LSP `textDocument.languageId` value. Per LSP spec.
    #[must_use]
    pub fn language_id(self) -> &'static str {
        match self {
            Self::Rust => "rust",
            Self::Python => "python",
            Self::TypeScript => "typescript",
            Self::JavaScript => "javascript",
            Self::Go => "go",
            Self::C => "c",
            Self::Cpp => "cpp",
        }
    }
}

/// Map a file path to a [`Language`] via its extension.
#[must_use]
pub fn detect_language(path: &Path) -> Option<Language> {
    let ext = path.extension().and_then(|s| s.to_str())?;
    Some(match ext {
        "rs" => Language::Rust,
        "py" => Language::Python,
        "ts" | "tsx" => Language::TypeScript,
        "js" | "jsx" | "mjs" | "cjs" => Language::JavaScript,
        "go" => Language::Go,
        "c" | "h" => Language::C,
        "cpp" | "cc" | "cxx" | "hpp" | "hh" => Language::Cpp,
        _ => return None,
    })
}

/// Default server command + args for a language. Returns `None`
/// when no built-in default is known. Per-language overrides should
/// be loaded from `[lsp.servers]` in `naked.json` before falling
/// back to this map.
#[must_use]
pub fn server_for_extension(lang: Language) -> Option<(&'static str, &'static [&'static str])> {
    Some(match lang {
        Language::Rust => ("rust-analyzer", &[]),
        Language::Python => ("pyright-langserver", &["--stdio"]),
        Language::TypeScript | Language::JavaScript => ("typescript-language-server", &["--stdio"]),
        Language::Go => ("gopls", &["serve"]),
        Language::C | Language::Cpp => ("clangd", &[]),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_extensions() {
        assert_eq!(detect_language(Path::new("foo.rs")), Some(Language::Rust));
        assert_eq!(
            detect_language(Path::new("a/b/foo.py")),
            Some(Language::Python)
        );
        assert_eq!(
            detect_language(Path::new("comp.tsx")),
            Some(Language::TypeScript)
        );
        assert_eq!(detect_language(Path::new("Main.go")), Some(Language::Go));
        assert_eq!(detect_language(Path::new("a.cpp")), Some(Language::Cpp));
    }

    #[test]
    fn unknown_extension_returns_none() {
        assert_eq!(detect_language(Path::new("readme.md")), None);
        assert_eq!(detect_language(Path::new("noext")), None);
    }

    #[test]
    fn server_for_each_language_has_default() {
        for lang in [
            Language::Rust,
            Language::Python,
            Language::TypeScript,
            Language::JavaScript,
            Language::Go,
            Language::C,
            Language::Cpp,
        ] {
            let (cmd, _args) = server_for_extension(lang).unwrap();
            assert!(!cmd.is_empty(), "server cmd missing for {lang:?}");
        }
    }
}
