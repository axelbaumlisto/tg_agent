//! Schema migration framework for persisted session/config records.
//!
//! Each domain (session, config) has a version. On load, if version < current,
//! forward migrations run in sequence. Backup created before mutating.

use serde_json::Value;

/// Migration function: takes mutable JSON value, upgrades in-place.
pub type MigrationFn = fn(&mut Value) -> Result<(), String>;

/// Trait for versioned persistence domains.
pub trait SchemaMigration {
    const CURRENT_VERSION: u32;
    const DOMAIN: &'static str;
    /// Migrations indexed by `[0] = v1→v2, [1] = v2→v3`, etc.
    const MIGRATIONS: &'static [MigrationFn];

    /// Run migrations from `from_version` up to `CURRENT_VERSION`.
    fn migrate(value: &mut Value, from_version: u32) -> Result<u32, String> {
        if from_version >= Self::CURRENT_VERSION {
            return Ok(from_version);
        }
        if from_version == 0 {
            return Err(format!("{}: version 0 is invalid", Self::DOMAIN));
        }
        let start = (from_version - 1) as usize;
        let end = (Self::CURRENT_VERSION - 1) as usize;
        if end > Self::MIGRATIONS.len() {
            return Err(format!(
                "{}: need migration to v{}, but only {} migrations defined",
                Self::DOMAIN,
                Self::CURRENT_VERSION,
                Self::MIGRATIONS.len()
            ));
        }
        for i in start..end {
            Self::MIGRATIONS[i](value)?;
        }
        // Stamp the new version:
        if let Some(obj) = value.as_object_mut() {
            obj.insert(
                "schema_version".into(),
                Value::Number(Self::CURRENT_VERSION.into()),
            );
        }
        Ok(Self::CURRENT_VERSION)
    }

    /// Check if record needs migration.
    fn needs_migration(version: u32) -> bool {
        version < Self::CURRENT_VERSION
    }
}

/// Create a backup of the file before migrating.
pub fn backup_before_migrate(path: &std::path::Path, domain: &str) -> std::io::Result<()> {
    let backup = path.with_extension(format!("{domain}.bak"));
    if path.exists() {
        std::fs::copy(path, &backup)?;
    }
    Ok(())
}

// ── Example: Session schema ────────────────────────────────────────────────

/// Session persistence migration.
pub struct SessionMigration;

impl SchemaMigration for SessionMigration {
    const CURRENT_VERSION: u32 = 1;
    const DOMAIN: &'static str = "session";
    const MIGRATIONS: &'static [MigrationFn] = &[];
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // Test domain with 2 versions:
    struct TestDomain;
    fn migrate_v1_to_v2(value: &mut Value) -> Result<(), String> {
        if let Some(obj) = value.as_object_mut() {
            obj.insert("new_field".into(), Value::String("default".into()));
        }
        Ok(())
    }
    impl SchemaMigration for TestDomain {
        const CURRENT_VERSION: u32 = 2;
        const DOMAIN: &'static str = "test";
        const MIGRATIONS: &'static [MigrationFn] = &[migrate_v1_to_v2];
    }

    #[test]
    fn no_migration_needed() {
        let mut v = json!({"schema_version": 2, "data": "ok"});
        assert_eq!(TestDomain::migrate(&mut v, 2).unwrap(), 2);
        assert!(!TestDomain::needs_migration(2));
    }

    #[test]
    fn migrates_v1_to_v2() {
        let mut v = json!({"schema_version": 1, "data": "old"});
        assert!(TestDomain::needs_migration(1));
        let result = TestDomain::migrate(&mut v, 1).unwrap();
        assert_eq!(result, 2);
        assert_eq!(v["new_field"], "default");
        assert_eq!(v["schema_version"], 2);
    }

    #[test]
    fn version_zero_error() {
        let mut v = json!({});
        assert!(TestDomain::migrate(&mut v, 0).is_err());
    }

    #[test]
    fn backup_works() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("test.json");
        std::fs::write(&path, "data").unwrap();
        backup_before_migrate(&path, "test").unwrap();
        assert!(tmp.path().join("test.test.bak").exists());
    }

    #[test]
    fn session_current_no_migration() {
        assert!(!SessionMigration::needs_migration(1));
    }
}
