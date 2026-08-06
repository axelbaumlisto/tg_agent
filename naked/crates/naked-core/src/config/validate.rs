//! Config validation — warn about common misconfigurations at load time.

use super::{Config, ProviderConfig};

fn warn_capability_mismatch(
    provider_name: &str,
    model: &str,
    task: crate::model_catalog::TaskKind,
    site: &'static str,
    pc: &ProviderConfig,
) {
    use crate::model_catalog::ModelStatus;

    // Only walk the alias chain if we actually have a caps block — no
    // sense hiding "unknown" pairs behind the permissive default.
    let resolved_model = pc.resolve_model_alias(model);
    let caps = match pc
        .capabilities
        .get(model)
        .or_else(|| pc.capabilities.get(resolved_model))
    {
        Some(c) => c,
        None => return, // No caps block → permissive unknown(), stay silent.
    };

    match caps.status {
        ModelStatus::Deprecated => {
            tracing::warn!(
                site = %site,
                provider = %provider_name,
                model = %model,
                task = %task,
                status = "deprecated",
                known_failure_modes = ?caps.known_failure_modes,
                notes = ?caps.notes,
                "model is deprecated; selector will refuse once enforce_model_capabilities=true"
            );
            return;
        }
        ModelStatus::Experimental => {
            tracing::warn!(
                site = %site,
                provider = %provider_name,
                model = %model,
                task = %task,
                status = "experimental",
                "model is marked experimental; pin it explicitly in new sessions only"
            );
            return;
        }
        ModelStatus::Degraded | ModelStatus::Active => {}
    }

    if !caps.fits(task) {
        tracing::warn!(
            site = %site,
            provider = %provider_name,
            model = %model,
            task = %task,
            task_fit = ?caps.task_fit,
            "model is not marked fit for this task in the capability catalog"
        );
    }

    if matches!(caps.status, ModelStatus::Degraded)
        && matches!(
            task,
            crate::model_catalog::TaskKind::Research | crate::model_catalog::TaskKind::Coding
        )
    {
        tracing::warn!(
            site = %site,
            provider = %provider_name,
            model = %model,
            task = %task,
            known_failure_modes = ?caps.known_failure_modes,
            "model is Degraded but referenced from a high-stakes task; \
             consider swapping to an Active alternative"
        );
    }
}

impl Config {
    pub fn validate_and_warn(&self) {
        if !self.default_provider.is_empty() && !self.providers.contains_key(&self.default_provider)
        {
            tracing::warn!(
                provider = %self.default_provider,
                known = ?self.providers.keys().collect::<Vec<_>>(),
                "config.default_provider is not in `providers` — requests will fail"
            );
        }

        if let Some(pc) = self.providers.get(&self.default_provider)
            && !self.default_model.is_empty()
            && !pc.models.is_empty()
            && !pc.serves_model(&self.default_model)
        {
            tracing::warn!(
                model = %self.default_model,
                provider = %self.default_provider,
                known = ?pc.models,
                "config.default_model is not listed under provider's `models` — may be rejected"
            );
        }

        for (name, pc) in &self.providers {
            if pc.api_key.is_empty() && pc.provider_type != "copilot" {
                tracing::warn!(
                    provider = %name,
                    "provider has empty api_key and is not copilot (auto-login); \
                     requests will 401 when routed here"
                );
            }
        }

        // Research subsystem: warn if the override points at a provider or model
        // that won't actually work at runtime. Typos here silently fell back to
        // the main provider before v5 — which made research burn the very
        // tokens the user was trying to save.
        if self.research.enabled {
            if let Some(p) = self.research.provider.as_deref()
                && !p.is_empty()
                && !self.providers.contains_key(p)
            {
                tracing::warn!(
                    research_provider = %p,
                    known = ?self.providers.keys().collect::<Vec<_>>(),
                    "config.research.provider is not in `providers` — research will fall back to default"
                );
            }
            if let Some(p) = self.research.provider.as_deref()
                && let Some(pc) = self.providers.get(p)
                && let Some(m) = self.research.model.as_deref()
                && !m.is_empty()
                && !pc.models.is_empty()
                && !pc.serves_model(m)
            {
                tracing::warn!(
                    research_model = %m,
                    research_provider = %p,
                    known = ?pc.models,
                    "config.research.model is not listed under its provider — may be rejected"
                );
            }
        }

        // Capability-catalog checks. Soft (warn-only) under the default
        // `enforce_model_capabilities=false`; phase 2 wires them into the
        // selector so warnings become hard filters. We still emit the
        // warning in enforce mode so operators see why a model was
        // filtered out.
        self.validate_capabilities();
    }

    /// Walk every selection site and compare the (provider, model) pair
    /// against the structured capability catalog
    /// ([`crate::model_catalog::ModelCapabilities`]).
    ///
    /// Three classes of warning are emitted:
    /// 1. `status=deprecated` — the selector will eventually refuse to
    ///    pick this; fix the config now.
    /// 2. `status=degraded` with a known-bad task fit — fine for
    ///    operator-pinned chat, but a lurking surprise for research.
    /// 3. `fits(task)=false` — the task isn't in the model's `task_fit`
    ///    list; structured selectors will skip it.
    ///
    /// Pairs with no capability block default to
    /// [`crate::model_catalog::ModelCapabilities::unknown`] and pass
    /// through silently (back-compat with legacy configs that haven't
    /// been seeded yet).
    fn validate_capabilities(&self) {
        use crate::model_catalog::{ModelStatus, TaskKind};

        // --- default chat pair ---------------------------------------
        if !self.default_provider.is_empty()
            && !self.default_model.is_empty()
            && let Some(pc) = self.providers.get(&self.default_provider)
        {
            warn_capability_mismatch(
                &self.default_provider,
                &self.default_model,
                TaskKind::Chat,
                "config.default_{provider,model}",
                pc,
            );
        }

        // --- global fallback chain -----------------------------------
        // Entries are "provider/model" strings; the chat turn consumes
        // them when the primary provider fails (see
        // `AgentConfig::fallback_providers`).
        for entry in &self.fallback {
            let Some((p, m)) = crate::research::parse_provider_model_pair(entry) else {
                continue;
            };
            if let Some(pc) = self.providers.get(&p) {
                warn_capability_mismatch(&p, &m, TaskKind::Chat, "config.fallback[]", pc);
            }
        }

        // --- research primary ---------------------------------------
        if self.research.enabled
            && let Some(p) = self
                .research
                .provider
                .as_deref()
                .filter(|s| !s.is_empty())
                .or({
                    if self.default_provider.is_empty() {
                        None
                    } else {
                        Some(self.default_provider.as_str())
                    }
                })
            && let Some(pc) = self.providers.get(p)
            && let Some(m) = self
                .research
                .model
                .as_deref()
                .filter(|s| !s.is_empty())
                .or({
                    if self.default_model.is_empty() {
                        None
                    } else {
                        Some(self.default_model.as_str())
                    }
                })
        {
            warn_capability_mismatch(
                p,
                m,
                TaskKind::Research,
                "config.research.{provider,model}",
                pc,
            );
        }

        // --- research fallback chain ---------------------------------
        if self.research.enabled {
            for entry in &self.research.fallback_models {
                // Entries may be bare model ids (resolved against
                // research.provider / default_provider) or
                // "provider/model" pairs.
                let (p, m) = match crate::research::parse_provider_model_pair(entry) {
                    Some(pair) => pair,
                    None => {
                        // Bare model — pin to research.provider or default.
                        let p = self
                            .research
                            .provider
                            .clone()
                            .filter(|s| !s.is_empty())
                            .unwrap_or_else(|| self.default_provider.clone());
                        if p.is_empty() {
                            continue;
                        }
                        (p, entry.clone())
                    }
                };
                if let Some(pc) = self.providers.get(&p) {
                    warn_capability_mismatch(
                        &p,
                        &m,
                        TaskKind::Research,
                        "config.research.fallback_models[]",
                        pc,
                    );
                }
            }
        }

        // --- memory digest provider ---------------------------------
        // The memory module parses `memory.digest_provider` as either
        // "<model>" (inherits default provider) or "<provider>/<model>".
        if let Some(entry) = self.memory.digest_provider.as_deref()
            && !entry.is_empty()
        {
            let (p, m) = match crate::research::parse_provider_model_pair(entry) {
                Some(pair) => pair,
                None => {
                    if self.default_provider.is_empty() {
                        return;
                    }
                    (self.default_provider.clone(), entry.to_string())
                }
            };
            if let Some(pc) = self.providers.get(&p) {
                warn_capability_mismatch(
                    &p,
                    &m,
                    TaskKind::Digest,
                    "config.memory.digest_provider",
                    pc,
                );
            }
        }

        // --- hard-deprecated model final sweep -----------------------
        // Also walk every explicitly-catalogued (provider, model) pair
        // and shout about `status=deprecated` ones — useful for pairs
        // that aren't referenced by any selection site yet but are
        // still in `providers[x].models[]`.
        for (pname, pc) in &self.providers {
            for model in &pc.models {
                if let Some(caps) = pc.capabilities.get(model)
                    && matches!(caps.status, ModelStatus::Deprecated)
                {
                    tracing::warn!(
                        provider = %pname,
                        model = %model,
                        "declared model is marked status=deprecated in capabilities; \
                         remove it from `providers.{}.models` or flip its status"
                         , pname
                    );
                }
            }
        }
    }
}
