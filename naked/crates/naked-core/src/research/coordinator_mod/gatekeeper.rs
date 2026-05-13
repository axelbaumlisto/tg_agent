//! Gatekeeper / verification logic for research findings.

use super::DeadFinding;
use super::FuzzyFingerprint;
use super::GatekeeperVerdict;
use super::ResearchCoordinator;
use crate::error::Result;
use crate::research::tool::output::parse_listing_date;
use chrono::Utc;
use std::time::Duration;

impl ResearchCoordinator {
    /// Check every finding against the rules defined in `GatekeeperConfig`.
    pub(crate) async fn verify_findings(&self, spec_id: &str) -> GatekeeperVerdict {
        let gk = &self.config.gatekeeper;
        let findings = self
            .store
            .list_findings(spec_id, None)
            .await
            .unwrap_or_default();

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(gk.url_check_timeout_secs))
            .redirect(reqwest::redirect::Policy::limited(5))
            .build()
            .unwrap_or_default();

        let mut verdict = GatekeeperVerdict::default();

        for f in &findings {
            let mut finding_issues = Vec::new();
            let mut needs_remediation = false;

            // 1. URL liveness
            let url_live = match client.head(&f.url).send().await {
                Ok(resp) => {
                    let status = resp.status().as_u16();
                    if status < 400 {
                        true
                    } else {
                        finding_issues.push(format!("URL {} → HTTP {status}", f.url));
                        false
                    }
                }
                Err(e) => {
                    finding_issues.push(format!("URL {} → error: {e}", f.url));
                    false
                }
            };

            if url_live {
                verdict.live_urls += 1;
            } else {
                verdict.dead_urls += 1;
                verdict.dead_hashes.insert(f.dedup_hash.clone());
                verdict.dead_details.push(DeadFinding {
                    url: f.url.clone(),
                    title: f.title.clone().unwrap_or_default(),
                });
            }

            // 2. Required fields (title, price, excerpt)
            let missing: Vec<&str> = [
                f.title.is_none().then_some("title"),
                f.price.is_none().then_some("price"),
                f.excerpt.is_none().then_some("excerpt"),
            ]
            .into_iter()
            .flatten()
            .collect();

            if !missing.is_empty() {
                verdict.missing_fields += 1;
                finding_issues.push(format!("{}: missing {}", f.url, missing.join(", ")));
                needs_remediation = true;
            }

            // 3. Excerpt quality — configurable minimum length
            if let Some(ref ex) = f.excerpt
                && ex.len() < gk.min_excerpt_chars
            {
                verdict.short_excerpts += 1;
                finding_issues.push(format!(
                    "{}: excerpt too short ({} chars, need ≥{})",
                    f.url,
                    ex.len(),
                    gk.min_excerpt_chars
                ));
                needs_remediation = true;
            }

            // 4. listing_date presence (if required by config)
            if gk.require_listing_date {
                let has_date = matches!(
                    &f.listing_date,
                    Some(d) if d.trim() != "unknown" && !d.trim().is_empty()
                );
                if !has_date {
                    verdict.missing_dates += 1;
                    finding_issues.push(format!("{}: missing listing_date", f.url));
                    needs_remediation = true;
                }
            }

            // 5. Date staleness — configurable max age
            if let Some(ref d) = f.listing_date
                && let Some(parsed) = parse_listing_date(d)
            {
                let age = (Utc::now().date_naive() - parsed).num_days();
                if age > gk.max_listing_age_days {
                    verdict.stale_dates += 1;
                    verdict.dead_hashes.insert(f.dedup_hash.clone());
                    finding_issues.push(format!("{}: stale date {d} ({age} days old)", f.url));
                }
            }

            // 6. source_content presence (if required by config)
            if gk.require_source_content {
                let has_source = f
                    .source_content
                    .as_ref()
                    .is_some_and(|s| s.len() >= gk.min_source_content_chars);
                if !has_source {
                    verdict.missing_source_content += 1;
                    finding_issues.push(format!(
                        "{}: missing or too short source_content (need ≥{})",
                        f.url, gk.min_source_content_chars
                    ));
                    needs_remediation = true;
                }
            }

            if needs_remediation && url_live {
                verdict.remediation_urls.push(f.url.clone());
            }

            verdict.issues.extend(finding_issues);
        }

        // 7. Semantic duplicate detection (if enabled in config)
        if gk.detect_semantic_duplicates {
            let mut seen_signatures: std::collections::HashMap<
                String,
                &crate::research::spec::Finding,
            > = std::collections::HashMap::new();
            // Kept survivors with cached fuzzy fingerprints for the second pass.
            let mut kept: Vec<(&crate::research::spec::Finding, FuzzyFingerprint)> = Vec::new();
            for f in &findings {
                if verdict.dead_hashes.contains(&f.dedup_hash) {
                    continue;
                }
                let title = f.title.as_deref().unwrap_or("").trim().to_lowercase();
                let price = f.price.as_deref().unwrap_or("").trim().to_lowercase();
                if title.is_empty() || title.len() < 10 {
                    continue;
                }
                let sig = format!("{title}||{price}");
                if let Some(earlier) = seen_signatures.get(&sig) {
                    verdict.semantic_dupes += 1;
                    verdict.dead_hashes.insert(f.dedup_hash.clone());
                    verdict.dead_details.push(DeadFinding {
                        url: f.url.clone(),
                        title: f.title.clone().unwrap_or_default(),
                    });
                    verdict.issues.push(format!(
                        "{}: semantic duplicate of {} (same title+price)",
                        f.url, earlier.url,
                    ));
                    continue;
                }

                // Fuzzy pass: collapse near-duplicates that the exact pass missed
                // (e.g. "70m² có bếp+PN" vs "70m²" at the same price).
                let fp = FuzzyFingerprint::new(f.title.as_deref(), f.price.as_deref());
                if let Some(fp) = fp {
                    if let Some((earlier, _)) =
                        kept.iter().find(|(_, other)| fp.is_duplicate_of(other))
                    {
                        verdict.semantic_dupes += 1;
                        verdict.dead_hashes.insert(f.dedup_hash.clone());
                        verdict.dead_details.push(DeadFinding {
                            url: f.url.clone(),
                            title: f.title.clone().unwrap_or_default(),
                        });
                        verdict.issues.push(format!(
                            "{}: fuzzy semantic duplicate of {} (overlapping title tokens + same price)",
                            f.url, earlier.url,
                        ));
                        continue;
                    }
                    seen_signatures.insert(sig, f);
                    kept.push((f, fp));
                } else {
                    seen_signatures.insert(sig, f);
                }
            }
        }

        verdict
    }

    /// Build a feedback prompt from configurable templates telling the agent
    /// what went wrong and asking for replacements and quality fixes.
    pub(crate) async fn build_feedback_prompt(
        &self,
        spec_id: &str,
        verdict: &GatekeeperVerdict,
    ) -> Result<String> {
        let gk = &self.config.gatekeeper;
        let spec = self.store.load_spec(spec_id).await?;
        let remaining = self.store.list_findings(spec_id, Some(50)).await?;
        let today = Utc::now().format("%d/%m/%Y").to_string();

        let dedup_list = remaining
            .iter()
            .rev()
            .take(50)
            .map(|f| {
                let title = f.title.as_deref().unwrap_or("(untitled)");
                format!("  - {} | {}", title, f.url)
            })
            .collect::<Vec<_>>()
            .join("\n");

        let dead_list = verdict
            .dead_details
            .iter()
            .map(|d| format!("  - REMOVED: {} (was: {})", d.url, d.title))
            .collect::<Vec<_>>()
            .join("\n");

        let missing_date_list: Vec<String> = verdict
            .issues
            .iter()
            .filter(|i| i.contains("missing listing_date"))
            .cloned()
            .collect();
        let missing_source_list: Vec<String> = verdict
            .issues
            .iter()
            .filter(|i| i.contains("source_content"))
            .cloned()
            .collect();
        let short_excerpt_list: Vec<String> = verdict
            .issues
            .iter()
            .filter(|i| i.contains("excerpt too short"))
            .cloned()
            .collect();
        let other_missing: Vec<String> = verdict
            .issues
            .iter()
            .filter(|i| {
                i.contains("missing title")
                    || i.contains("missing price")
                    || i.contains("missing excerpt")
            })
            .cloned()
            .collect();

        let remediation_section = if verdict.remediation_urls.is_empty() {
            String::new()
        } else {
            let urls = verdict
                .remediation_urls
                .iter()
                .map(|u| format!("  - {u}"))
                .collect::<Vec<_>>()
                .join("\n");
            format!(
                "## URLs to re-visit and update\n\
                 The following {} URLs are LIVE but have incomplete data. \
                 Re-fetch each page and call `research_save` again with the SAME URL \
                 but with ALL required fields filled in.\n{urls}\n\n",
                verdict.remediation_urls.len()
            )
        };

        // Build quality sections from configurable templates
        let mut quality_sections = String::new();
        if !missing_date_list.is_empty() {
            quality_sections.push_str(
                &gk.missing_date_section
                    .replace("{count}", &missing_date_list.len().to_string())
                    .replace("{today}", &today)
                    .replace("{list}", &missing_date_list.join("\n")),
            );
        }
        if !missing_source_list.is_empty() {
            quality_sections.push_str(
                &gk.missing_source_section
                    .replace("{count}", &missing_source_list.len().to_string())
                    .replace("{min_source}", &gk.min_source_content_chars.to_string())
                    .replace("{list}", &missing_source_list.join("\n")),
            );
        }
        if !short_excerpt_list.is_empty() {
            quality_sections.push_str(
                &gk.short_excerpt_section
                    .replace("{count}", &short_excerpt_list.len().to_string())
                    .replace("{min_excerpt}", &gk.min_excerpt_chars.to_string())
                    .replace("{list}", &short_excerpt_list.join("\n")),
            );
        }
        if !other_missing.is_empty() {
            quality_sections.push_str(&format!(
                "### Missing required fields\n{}\n\n",
                other_missing.join("\n")
            ));
        }

        // Render the main feedback prompt from the configurable header template
        let prompt = gk
            .feedback_prompt_header
            .replace("{id}", &spec.id)
            .replace("{topic}", &spec.topic)
            .replace("{today}", &today)
            .replace(
                "{dead_count}",
                &(verdict.dead_urls + verdict.stale_dates + verdict.semantic_dupes).to_string(),
            )
            .replace(
                "{remediation_count}",
                &verdict.remediation_urls.len().to_string(),
            )
            .replace(
                "{dead_list}",
                if dead_list.is_empty() {
                    "(none)"
                } else {
                    &dead_list
                },
            )
            .replace("{remediation_section}", &remediation_section)
            .replace(
                "{quality_sections}",
                if quality_sections.is_empty() {
                    "(none)\n"
                } else {
                    &quality_sections
                },
            )
            .replace("{dedup_list}", &dedup_list)
            .replace("{min_excerpt}", &gk.min_excerpt_chars.to_string())
            .replace("{min_source}", &gk.min_source_content_chars.to_string());

        Ok(prompt)
    }
}

#[cfg(test)]
mod tests {
    use super::super::FuzzyFingerprint;

    #[test]
    fn fuzzy_detects_duplicates_same_price() {
        let a = FuzzyFingerprint::new(
            Some("apartment two bedroom vinhomes grand park district nine"),
            Some("350000"),
        )
        .unwrap();
        let b = FuzzyFingerprint::new(
            Some("apartment two bedroom vinhomes grand park district nine sale"),
            Some("350000"),
        )
        .unwrap();
        assert!(
            a.is_duplicate_of(&b),
            "overlapping titles + same price = dupe"
        );
    }

    #[test]
    fn fuzzy_different_prices_not_duplicate() {
        let a =
            FuzzyFingerprint::new(Some("Nice apartment downtown area"), Some("500 USD")).unwrap();
        let b =
            FuzzyFingerprint::new(Some("Nice apartment downtown area"), Some("600 USD")).unwrap();
        assert!(!a.is_duplicate_of(&b));
    }

    #[test]
    fn fuzzy_short_title_returns_none() {
        assert!(FuzzyFingerprint::new(Some("Hi"), Some("100")).is_none());
    }

    #[test]
    fn fuzzy_no_price_not_duplicate() {
        let a = FuzzyFingerprint::new(Some("Large villa with pool in Bali Ubud"), None).unwrap();
        let b = FuzzyFingerprint::new(Some("Large villa with pool in Bali Ubud"), None).unwrap();
        assert!(!a.is_duplicate_of(&b), "missing price should not be dupe");
    }
}
