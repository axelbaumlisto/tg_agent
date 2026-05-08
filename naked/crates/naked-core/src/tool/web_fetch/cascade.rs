//! Multi-tier fallback cascade for `web_fetch`.
//!
//! Tier order:
//!   Tier-1  reqwest direct (handled in mod.rs::execute before calling here)
//!   Tier-2  URL-prefix CORS proxy
//!   Tier-3  TLS impersonation subprocess (curl_cffi)
//!   Tier-3.5 Cloud-scrape (ScrapingBee / Firecrawl)
//!   Tier-4  Wayback snapshot

use crate::scrape::host_policy::{Outcome as TierOutcome, Tier};

use super::WebFetchTool;
use super::backends::{fetch_via_tls_subprocess, fetch_wayback_snapshot, format_wayback_ts};
use super::error::detect_block;
use super::http::{FetchOutcome, is_already_prefixed, resolve_url_prefix};

impl WebFetchTool {
    /// Fallback cascade: URL-prefix → TLS → cloud-scrape → Wayback.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn try_fallback_cascade(
        &self,
        primary: &mut FetchOutcome,
        url: &str,
        start_tier: Tier,
        skip_fallback: bool,
        max_chars: usize,
        include_links: bool,
        cascade_notes: &mut Vec<String>,
    ) -> Option<crate::types::ToolResult> {
        // Tier-2 fallback: retry once via the CORS-prefix pool only if the
        // primary path clearly failed (transport error, anti-bot wall, or
        // non-success HTTP status). Successful 200-with-body is returned
        // immediately — no point burning a second round-trip.
        if primary.is_degraded()
            && start_tier <= Tier::UrlPrefix
            && let Some(prefix) = resolve_url_prefix()
            && !is_already_prefixed(url, &prefix)
        {
            let wrapped = format!("{prefix}/{url}");
            tracing::info!(
                target = %url,
                via = %prefix,
                primary_status = primary.status,
                primary_error = %primary.transport_error.as_deref().unwrap_or(""),
                "web_fetch: primary degraded, retrying via URL-prefix proxy",
            );
            match self.fetch_once(&wrapped).await {
                Ok(mut fallback) => {
                    // The CORS proxy may itself return 5xx when the
                    // upstream target is bad; in that case we still
                    // prefer the primary result (it's at least
                    // authoritative about what the origin said).
                    if !fallback.is_degraded() {
                        // Rewrite final_url back to the original
                        // target so downstream dedup/memory doesn't
                        // treat `ws-xxx.onrender.com/...` as a new
                        // canonical URL for this content.
                        fallback.final_url = url.to_string();
                        *primary = fallback;
                        self.host_policy
                            .record(url, Tier::UrlPrefix, TierOutcome::Ok);
                    } else {
                        tracing::info!(
                            fallback_status = fallback.status,
                            fallback_error = %fallback.transport_error.as_deref().unwrap_or(""),
                            "web_fetch: URL-prefix fallback also degraded; keeping primary result",
                        );
                        self.host_policy
                            .record(url, Tier::UrlPrefix, TierOutcome::Blocked);
                    }
                }
                Err(msg) => {
                    tracing::warn!(error = %msg, "web_fetch: URL-prefix fallback failed");
                    self.host_policy
                        .record(url, Tier::UrlPrefix, TierOutcome::Blocked);
                }
            }
        }

        // --- Tier-3: TLS impersonation via curl_cffi subprocess ---------
        // Reqwest+URL-prefix both reached dead ends. Before giving up,
        // try the JA3-impersonation path — it's the only thing that
        // empirically cracks batdongsan/chotot/alonhadat CF walls.
        let is_blocked_primary = primary.transport_error.is_some()
            || detect_block(primary.status, &primary.text).is_some()
            || !(200..400).contains(&primary.status);
        if is_blocked_primary && start_tier <= Tier::Reqwest {
            cascade_notes.push(match primary.transport_error.as_deref() {
                Some(e) => format!("reqwest: transport error ({e})"),
                None => format!(
                    "reqwest: HTTP {} {}",
                    primary.status,
                    detect_block(primary.status, &primary.text)
                        .map(|k| format!("({:?})", k))
                        .unwrap_or_default()
                ),
            });
        }

        if is_blocked_primary && !skip_fallback && start_tier <= Tier::Tls {
            tracing::debug!(
                url = %url,
                "web_fetch: primary blocked/failed, escalating to TLS impersonation",
            );
            match fetch_via_tls_subprocess(url).await {
                Ok(tls_body) => {
                    if let Some(kind) = detect_block(tls_body.status, &tls_body.text) {
                        cascade_notes.push(format!("tls: HTTP {} ({:?})", tls_body.status, kind));
                        self.host_policy
                            .record(url, Tier::Tls, TierOutcome::Blocked);
                    } else if !(200..400).contains(&tls_body.status) {
                        cascade_notes.push(format!("tls: HTTP {}", tls_body.status));
                        self.host_policy
                            .record(url, Tier::Tls, TierOutcome::Blocked);
                    } else {
                        *primary = tls_body;
                        cascade_notes.push("tls: OK (used)".into());
                        self.host_policy.record(url, Tier::Tls, TierOutcome::Ok);
                    }
                }
                Err(msg) => {
                    cascade_notes.push(format!("tls: {msg}"));
                    self.host_policy
                        .record(url, Tier::Tls, TierOutcome::Blocked);
                }
            }
        }

        // --- Tier-3.5: Cloud-scrape cascade (ScrapingBee / Firecrawl) ---
        // Reqwest, URL-prefix, AND TLS impersonation all reached blocks.
        // Burn a paid request through ScrapingBee/Firecrawl — they bring
        // residential IPs + headless browsers and crack the remaining
        // ~30% of CF-protected VN portals (chotot listing detail pages,
        // batdongsan ads with phone-reveal walls, dotproperty SPA).
        let mid_blocked = primary.transport_error.is_some()
            || detect_block(primary.status, &primary.text).is_some()
            || !(200..400).contains(&primary.status);
        if mid_blocked
            && !skip_fallback
            && start_tier <= Tier::Cloud
            && let Some(cloud) = self.cloud.as_ref()
            && cloud.is_active()
        {
            tracing::debug!(
                url = %url,
                engines = %cloud.engine_summary(),
                "web_fetch: escalating to cloud-scrape cascade",
            );
            match cloud.scrape(url).await {
                Ok(cloud_body) => {
                    if let Some(kind) = detect_block(cloud_body.status, &cloud_body.body) {
                        cascade_notes.push(format!(
                            "{}: HTTP {} ({:?})",
                            cloud_body.provider, cloud_body.status, kind
                        ));
                        self.host_policy
                            .record(url, Tier::Cloud, TierOutcome::Blocked);
                    } else if !(200..400).contains(&cloud_body.status) {
                        cascade_notes.push(format!(
                            "{}: HTTP {}",
                            cloud_body.provider, cloud_body.status
                        ));
                        self.host_policy
                            .record(url, Tier::Cloud, TierOutcome::Blocked);
                    } else {
                        *primary = FetchOutcome {
                            status: cloud_body.status,
                            final_url: cloud_body.final_url,
                            content_type: cloud_body.content_type,
                            text: cloud_body.body,
                            transport_error: None,
                        };
                        cascade_notes.push(format!("{}: OK (used)", cloud_body.provider));
                        self.host_policy.record(url, Tier::Cloud, TierOutcome::Ok);
                    }
                }
                Err(msg) => {
                    cascade_notes.push(format!("cloud-scrape: {msg}"));
                    self.host_policy
                        .record(url, Tier::Cloud, TierOutcome::Blocked);
                }
            }
        }

        // --- Tier-4: Wayback snapshot -----------------------------------
        // Last resort. Serves stale content but at least gives the agent
        // *something* to reason about. Labelled clearly in the header so
        // the agent knows to cite the archive timestamp.
        let still_blocked = primary.transport_error.is_some()
            || detect_block(primary.status, &primary.text).is_some()
            || !(200..400).contains(&primary.status);
        if still_blocked && !skip_fallback && start_tier <= Tier::Wayback {
            tracing::info!(url = %url, "web_fetch: escalating to Wayback snapshot");
            match fetch_wayback_snapshot(&self.client, url).await {
                Ok((wb_body, wb_timestamp, wb_snapshot_url)) => {
                    cascade_notes.push(format!("wayback: OK (ts={wb_timestamp}, used)"));
                    self.host_policy.record(url, Tier::Wayback, TierOutcome::Ok);
                    let cascade_summary = cascade_notes.join(" → ");
                    let header = format!(
                        "Wayback snapshot (timestamp {}): {}\nOriginal URL: {url}\nFallback cascade: {cascade_summary}\n\n",
                        format_wayback_ts(&wb_timestamp),
                        wb_snapshot_url,
                    );
                    return Some(super::super::fetch_common::format_fetch_output(
                        &wb_body,
                        &header,
                        include_links,
                        max_chars,
                        true,
                    ));
                }
                Err(msg) => {
                    cascade_notes.push(format!("wayback: {msg}"));
                    self.host_policy
                        .record(url, Tier::Wayback, TierOutcome::Blocked);
                }
            }
        }

        None
    }
}
