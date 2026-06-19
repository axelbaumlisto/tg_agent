//! Pure boot-audit target de-duplication helpers (B63).
//!
//! The boot audit must probe each physical credential at most once per
//! process boot. These helpers operate only on already-resolved key strings
//! and expose non-secret fingerprints; they never log or store raw key values.

use std::collections::HashSet;

/// Non-secret stable fingerprint of a resolved API key.
///
/// This is intentionally opaque: callers can compare/hash it, but should not
/// treat it as a credential or print raw key material alongside it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KeyFingerprint(String);

impl KeyFingerprint {
    /// Short stable hash string suitable for tests/diagnostics without key
    /// preimage disclosure.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Fingerprint one resolved key value.
///
/// Blank strings and unresolved `$ENV` placeholders are ignored. Production
/// callers should pass values from `ProviderConfig::resolved_all_keys()`;
/// this guard keeps the pure helper safe for tests and future callers.
pub fn key_fingerprint(resolved_key: &str) -> Option<KeyFingerprint> {
    let key = resolved_key.trim();
    if key.is_empty() || key.starts_with('$') {
        return None;
    }

    // FNV-1a 64-bit: tiny, deterministic, no dependency or randomized state.
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in key.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    Some(KeyFingerprint(format!("{hash:016x}")))
}

/// One physical credential selected for boot probing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DedupAuditTarget {
    pub provider_name: String,
    pub key_index: usize,
    pub fingerprint: KeyFingerprint,
}

/// Deterministic de-duplication result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DedupAuditPlan {
    pub providers_seen: usize,
    pub targets: Vec<DedupAuditTarget>,
    pub targets_skipped_dup: usize,
}

/// Build a boot-audit target list by first occurrence of physical key.
///
/// Input order is significant and deterministic: the first provider/key index
/// that owns a fingerprint is kept, later duplicates are skipped. Blank keys
/// and unresolved `$ENV` placeholders are not emitted.
pub fn dedup_audit_targets(providers: &[(String, Vec<String>)]) -> DedupAuditPlan {
    let mut seen = HashSet::new();
    let mut targets = Vec::new();
    let mut targets_skipped_dup = 0usize;

    for (provider_name, keys) in providers {
        for (key_index, key) in keys.iter().enumerate() {
            let Some(fingerprint) = key_fingerprint(key) else {
                continue;
            };
            if !seen.insert(fingerprint.clone()) {
                targets_skipped_dup += 1;
                continue;
            }
            targets.push(DedupAuditTarget {
                provider_name: provider_name.clone(),
                key_index,
                fingerprint,
            });
        }
    }

    DedupAuditPlan {
        providers_seen: providers.len(),
        targets,
        targets_skipped_dup,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fp(key: &str) -> KeyFingerprint {
        key_fingerprint(key).expect("test key should fingerprint")
    }

    #[test]
    fn dedup_audit_targets_shared_fallback_once() {
        let providers = vec![
            (
                "a".to_string(),
                vec!["a-primary".into(), "shared-fallback".into()],
            ),
            (
                "b".to_string(),
                vec!["b-primary".into(), "shared-fallback".into()],
            ),
            (
                "c".to_string(),
                vec!["c-primary".into(), "shared-fallback".into()],
            ),
        ];

        let plan = dedup_audit_targets(&providers);

        assert_eq!(plan.providers_seen, 3);
        assert_eq!(plan.targets_skipped_dup, 2);
        assert_eq!(
            plan.targets
                .iter()
                .filter(|t| t.fingerprint == fp("shared-fallback"))
                .count(),
            1
        );
    }

    #[test]
    fn dedup_audit_targets_primary_in_a_beats_fallback_in_b() {
        let providers = vec![
            ("a".to_string(), vec!["same-physical-key".into()]),
            (
                "b".to_string(),
                vec!["b-primary".into(), "same-physical-key".into()],
            ),
        ];

        let plan = dedup_audit_targets(&providers);

        let same = fp("same-physical-key");
        let kept = plan
            .targets
            .iter()
            .find(|target| target.fingerprint == same)
            .expect("shared key should be kept once");
        assert_eq!(kept.provider_name, "a");
        assert_eq!(kept.key_index, 0);
        assert_eq!(plan.targets_skipped_dup, 1);
    }

    #[test]
    fn dedup_audit_targets_multi_own_key_partial_dup() {
        let providers = vec![
            (
                "a".to_string(),
                vec!["a0".into(), "dup".into(), "a2".into()],
            ),
            (
                "b".to_string(),
                vec!["b0".into(), "dup".into(), "b2".into()],
            ),
        ];

        let plan = dedup_audit_targets(&providers);

        let kept: Vec<(&str, usize)> = plan
            .targets
            .iter()
            .map(|target| (target.provider_name.as_str(), target.key_index))
            .collect();
        assert_eq!(kept, vec![("a", 0), ("a", 1), ("a", 2), ("b", 0), ("b", 2)]);
        assert_eq!(plan.targets.len(), 5);
        assert_eq!(plan.targets_skipped_dup, 1);
    }

    #[test]
    fn dedup_audit_targets_blank_and_env_placeholders_skipped() {
        let providers = vec![(
            "a".to_string(),
            vec![
                "".into(),
                "   ".into(),
                "$MISSING_KEY".into(),
                "real-key".into(),
            ],
        )];

        let plan = dedup_audit_targets(&providers);

        assert_eq!(plan.targets.len(), 1);
        assert_eq!(plan.targets[0].provider_name, "a");
        assert_eq!(plan.targets[0].key_index, 3);
        assert_eq!(plan.targets[0].fingerprint, fp("real-key"));
        assert_eq!(plan.targets_skipped_dup, 0);
    }
}
