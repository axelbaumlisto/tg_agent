//! Prompt building, report generation, agent briefs, run records.

use super::ResearchCoordinator;
use super::ResearchSpec;
use super::RunRecord;
use super::RunReport;
use super::StopReason;
use super::VerificationSummary;
use crate::error::Result;
use chrono::Utc;

impl ResearchCoordinator {
    /// Compose the agent-facing prompt. Short but dense: spec topic, source
    /// seeds, last-N known-finding URLs for dedup, cursor JSON, and a strict
    /// rule block describing the save/list/done tool contract.
    pub(crate) async fn build_prompt(&self, spec: &ResearchSpec) -> Result<String> {
        let known = self.store.list_findings(&spec.id, Some(50)).await?;
        let cursor = self.store.load_cursor(&spec.id).await?;
        let agent_brief = self.store.read_agent_brief(&spec.id).await.unwrap_or(None);

        let dedup_list = if known.is_empty() {
            "  (none yet — first pass)".to_string()
        } else {
            known
                .iter()
                .rev()
                .take(50)
                .map(|f| {
                    let title = f.title.as_deref().unwrap_or("(untitled)");
                    let price = f.price.as_deref().unwrap_or("?");
                    let date = f.listing_date.as_deref().unwrap_or("no date");
                    format!("  - {title} | {price} | {date} | {}", f.url)
                })
                .collect::<Vec<_>>()
                .join("\n")
        };

        let cursor_json = if cursor.data.is_empty() {
            "(empty)".to_string()
        } else {
            serde_json::to_string(&cursor.data).unwrap_or_else(|_| "(unparseable)".to_string())
        };

        let sources = if spec.sources.is_empty() {
            "(no seeds — use web_search or your best guesses)".to_string()
        } else {
            spec.sources
                .iter()
                .map(|s| format!("  - {s}"))
                .collect::<Vec<_>>()
                .join("\n")
        };

        let max_nav = spec
            .max_iterations
            .unwrap_or(self.config.default_max_iterations);

        let today = Utc::now().format("%d/%m/%Y").to_string();

        let brief_section = match &agent_brief {
            Some(b) => format!(
                "# Brief from previous run\n\
                 The following is a structured summary of the previous run. Use it to \
                 focus on NEW listings, deeper pages, or sources that were not yet explored.\n\
                 \n{b}\n"
            ),
            None => String::new(),
        };

        Ok(format!(
            "You are an autonomous researcher. Your research id is `{id}`.\n\
             \n\
             # BEFORE YOU DO ANYTHING\n\
             Call `Skill(name=\"research-playbook\")` once, right now. It is \
             the master playbook for this entire pipeline — multi-engine \
             search, snippet fast-path, the 5-tier fetch cascade with \
             adaptive per-host policy, the VN source allowlist \
             (mogi/homedy/nhadat24h open, batdongsan/chotot/dotproperty \
             need tier 3+), dedup contract, and the gatekeeper's report \
             requirements. Skipping this skill is the single largest \
             source of wasted iterations in prior runs.\n\
             \n\
             # Topic\n{topic}\n\
             \n\
             # Seed sources\n{sources}\n\
             \n\
             {brief_section}\
             # Fetch-tool cascade (what you see before pivoting to the browser)\n\
             `web_fetch` now AUTO-ESCALATES through five tiers before it ever \
             returns a `BLOCKED:` line:\n\
             \n\
             1. reqwest direct (+ NAKED_PROXY tunnel if configured)\n\
             2. URL-prefix CORS proxy fallback\n\
             3. TLS impersonation via `web_fetch_tls` (curl_cffi with chrome120/\
                chrome124/firefox133 ladder — empirically cracks batdongsan.com.vn, \
                chotot.com, alonhadat.com.vn)\n\
             4. Cloud-scrape cascade (ScrapingBee → Firecrawl, when keys are \
                provisioned) — residential-IP + headless-Chromium, paid per \
                request but cracks the remaining ~30% of CF walls including \
                phone-reveal SPAs on chotot/dotproperty/batdongsan.\n\
             5. `web_fetch_wayback` archived snapshot (stale but usable)\n\
             \n\
             The cascade is **adaptive**: a per-host policy tracks which tiers \
             have failed for each domain. After 3 consecutive blocks at a tier \
             for one host, the cascade silently skips that tier on the next \
             fetch (you may see `host-policy: skip → start at tls` in the \
             cascade summary). A single success at a higher tier resets the \
             counter so cheaper tiers get retried. This shaves ~10 s per \
             fetch on known-protected hosts (batdongsan/chotot/dotproperty).\n\
             \n\
             You will only see `BLOCKED:` after ALL FIVE tiers failed. When that \
             happens, prefer switching to an **alternative source first** \
             (`mogi.vn`, `homedy.com`, `nhadat24h.net` — all empirically open) \
             before paying the browser-MCP cost. You can also call the specific \
             tier explicitly: `web_fetch_tls` for a JS-less CF bypass, or \
             `web_fetch_wayback` for a stale-but-present snapshot.\n\
             \n\
             # MANDATORY captcha / anti-bot procedure (non-negotiable)\n\
             Whenever `web_fetch` replies with a line starting `BLOCKED:`, or \
             whenever you hit any of these JS-gated / Cloudflare-protected hosts — \
             `alonhadat.com.vn`, `batdongsan.com.vn`, `nhatot.com`, `chotot.com`, \
             `nhadat24h.net`, `dotproperty.com.vn`, `mogi.vn`, `nhaban.com` — \
             you MUST pivot to the browser in THIS exact order:\n\
             \n\
             1. `Skill(name=\"web-browser-playbook\")` — load the playbook first, \
                before any browser call. The skill lists per-host selectors for \
                the click-to-reveal phone recipe.\n\
             2. `browser_navigate {{ url: \"<final_url>\" }}` — open the page.\n\
             3. `browser_snapshot {{ take_screenshot_afterwards: true }}` — extract \
                the real DOM. Use the snapshot refs (not the screenshot) for any \
                click/hover that follows.\n\
             \n\
             DO NOT call `research_save` with the body of a Cloudflare / \
             `Just a moment…` / `Vui lòng xác minh` page — the save will be \
             rejected at the tool boundary (`captcha_stub`) and the iteration \
             wasted. The only acceptable content for a save is text extracted \
             from a successful `browser_snapshot` or a clean `web_fetch` \
             response (i.e. not prefixed with `BLOCKED:`).\n\
             \n\
             # Strategy\n\
             1. **Try every seed source** — use `web_fetch` first. If it returns a \
                captcha page, 403, or empty content, switch to `browser_navigate` + \
                `browser_snapshot` for that site.\n\
             2. **Search broadly** — after exhausting seeds, use `web_search` \
                (multi-engine: exa+tavily+serpapi+ddg in parallel, deduped) \
                with **at least 3-5 different query variations** derived from the topic. \
                Try different keywords, synonyms, location names, site-specific queries. \
                Cast a wide net. More variations = more unique results.\n\
             3. **Snippet fast-path (cost saver)** — `web_search` results may include \
                an `⚡ extracted:` line with `price`, `area`, `district`, `phone` already \
                parsed from the snippet. When ALL FOUR are present AND the URL is a real \
                listing page (not a category index), you MAY call `research_save` \
                directly using the snippet contents — no `web_fetch` needed. \
                For partial extractions (missing phone or district), still open the page \
                to confirm. Document in the excerpt: `extracted from search snippet`.\n\
             4. **Follow links into individual listing pages** — when the snippet \
                fast-path does NOT apply, you MUST open each individual listing URL \
                with `web_fetch` or `browser_navigate` to extract full details BEFORE \
                saving. Never save a finding from a partial search snippet — always \
                verify the page loads and contains the data.\n\
             5. **Verify before saving** — after fetching a page, confirm it shows \
                a real listing (not a 404, captcha, or redirect). Only then call \
                `research_save`. If `web_fetch` returns a captcha or error page, \
                try `browser_navigate` instead. If both fail, skip that URL.\n\
             6. **Prefer sources with open contacts** — prioritize sites that show \
                phone numbers, Zalo, WhatsApp without requiring login. Sites like \
                nhaban.com, chotot.com, facebook marketplace often show contacts \
                directly. Extract real phone numbers whenever visible.\n\
                For Vietnamese aggregators that hide contacts behind a JS \
                click-to-reveal button (alonhadat.com.vn, nhadat24h.net, \
                batdongsan.com.vn, dotproperty.com.vn, mogi.vn, homedy.com) — \
                load the `web-browser-playbook` skill via the `Skill` tool \
                BEFORE fetching the first such URL. The playbook lists the \
                exact button texts and CSS selectors per host so you can click \
                and re-snapshot to extract the real number.\n\
                \n\
                NOTE: `propertyguru.com.vn` has been rebranded to \
                `dotproperty.com.vn` (the `.com.vn` zone no longer has an A \
                record — DNS lookup will fail). When you see `propertyguru.com.vn` \
                in a search snippet, use the `dotproperty.com.vn` equivalent \
                URL instead.\n\
             7. **Save findings** — for every qualifying item whose URL is NOT in \
                the dedup list below, call `research_save` with complete data. \
                You MUST actually invoke the tool, not just describe what you'd do.\n\
             \n\
             # Hard rules around contacts (always inline, never override)\n\
             - Never invent a phone number, Zalo handle, email, or contact line. \
               If the page does not show a real digit string, the contact is \
               UNKNOWN — do not paraphrase it as `Liên hệ qua <site>` or \
               `Contact via website`.\n\
             - If you tried the click-to-reveal recipe from `web-browser-playbook` \
               and the contact is STILL hidden (real captcha, login wall, A/B \
               test variant), the excerpt for that finding MUST start with the \
               literal string `Contacts hidden behind site captcha — visit URL`. \
               The gatekeeper checks for this exact phrase.\n\
             - For unknown JS-heavy hosts not in the playbook, default to the \
               same workflow: `Skill(skill=\"web-browser-playbook\")` first, then \
               apply the same patterns to the new host.\n\
             \n\
             # Required fields for every finding\n\
             \n\
             ## `listing_date` (mandatory)\n\
             The publication or last-update date shown on the page.\n\
             - Look for: date labels (`posted`, `updated`, `ngày đăng`, `cập nhật`, \
               `đăng ngày`, `дата публикации`, `опубликовано`), breadcrumbs, sidebar \
               metadata, page footer near listing ID.\n\
             - Convert relative dates to absolute DD/MM/YYYY: `today`/`hôm nay` → \
               {today}, `yesterday`/`hôm qua` → yesterday, `N days ago`/`N ngày trước` \
               → today minus N days.\n\
             - **SKIP listings older than 90 days** — do NOT save them.\n\
             - If the page truly has no date after checking all locations, \
               use `\"listing_date\": \"unknown\"`.\n\
             \n\
             ## `excerpt` (mandatory, ≥300 chars, up to 2000)\n\
             This is the MOST VALUABLE field. It must let the user act on the \
             finding WITHOUT visiting the URL. Extract EVERYTHING:\n\
             - **CONTACTS**: phone number, name, Zalo, WhatsApp, email, agency. \
               If contacts are hidden behind login/registration, write \
               \"Contacts hidden — requires site registration\".\n\
             - **SPECS**: area m², dimensions, floors, rooms, condition, \
               furnishing.\n\
             - **TERMS**: price, deposit, contract length, payment schedule.\n\
             - **LOCATION**: full address, nearby landmarks, district.\n\
             - **EXTRAS**: photos count, available date, special features.\n\
             Use the full 2000 char budget. Short excerpts are rejected.\n\
             - **Do NOT** include source/site attribution (`Nguồn: ...`, \
               `Source: ...`, `Posted by ...`, `đăng N ngày trước`, \
               `Cập nhật N giờ trước`). The `url` and `listing_date` fields \
               already capture provenance — repeating it wastes the excerpt budget. \
               The excerpt is for actionable detail only.\n\
             \n\
             ## `source_content` (mandatory, ≥100 chars, up to 8000)\n\
             Paste the condensed text of the page stripped of navigation, ads, \
             and JS boilerplate. This lets us verify claims later.\n\
             \n\
             ## `title` and `price` (mandatory)\n\
             Title must accurately describe the listing. Price as shown on page.\n\
             \n\
             # Quality rules\n\
             - **Verify first, save second** — ALWAYS open the listing URL before \
               saving. If the page is a 404, captcha, or error, do NOT save it. \
               Dead URLs waste everyone's time.\n\
             - **Real URLs only** — never fabricate. The `title` must match the \
               actual content at the URL.\n\
             - **No near-duplicates** — if two listings have the same title AND \
               price, save only one.\n\
             - **Extract real contacts** — if a phone number is visible on the page, \
               it MUST appear in the excerpt. Do not write \"contact via website\" if \
               the actual phone number is shown on the page.\n\
             - **Batch saves** — you may call research_save multiple times in \
               parallel for different listings. This is faster.\n\
             \n\
             # Tool-call template\n\
             ```\n\
             research_save({{\n\
               \"url\": \"https://example.com/listing/123\",\n\
               \"title\": \"2BR apartment, District 7 — $500/mo\",\n\
               \"price\": \"500 USD/month\",\n\
               \"listing_date\": \"15/04/2026\",\n\
               \"excerpt\": \"2-bedroom apartment, 65m², 10th floor, fully furnished. \
             Building: Sunrise City, Nguyen Huu Tho, District 7. Amenities: pool, gym, \
             24/7 security, parking. Condition: newly renovated, move-in ready. Deposit: \
             2 months. Contract: minimum 1 year. CONTACT: Ms. Lan, 0912-345-678 \
             (Zalo/WhatsApp). Available from May 1.\",\n\
               \"source_content\": \"(full page text stripped of navigation and ads, up to 8000 chars)\"\n\
             }})\n\
             ```\n\
             \n\
             If the topic is an imperative (\"save X as a finding with title Y\"), \
             execute it literally with a single `research_save` call.\n\
             \n\
             # Budget\n\
             At most {max_nav} tool invocations. Stop early if the topic is \
             exhausted. Do not keep browsing after you've saved everything.\n\
             \n\
             # Known findings (do NOT save duplicates)\n{dedup_list}\n\
             \n\
             # Cursor from last run\n{cursor_json}\n",
            id = spec.id,
            topic = spec.topic,
            sources = sources,
            brief_section = brief_section,
            dedup_list = dedup_list,
            cursor_json = cursor_json,
            max_nav = max_nav,
            today = today,
        ))
    }

    pub(crate) async fn regenerate_report(&self, spec: &ResearchSpec) -> Result<()> {
        let findings = self.store.list_findings(&spec.id, None).await?;
        let runs = self.store.list_runs(&spec.id, Some(10)).await?;
        let mut md = String::new();
        md.push_str(&format!("# {}\n\n", spec.topic));
        md.push_str(&format!("_Research id: `{}`_\n\n", spec.id));
        md.push_str(&format!("**Total findings:** {}\n\n", findings.len()));

        md.push_str("## Latest findings\n\n");
        if findings.is_empty() {
            md.push_str("_(none yet)_\n\n");
        } else {
            for f in findings.iter().rev().take(25) {
                let title = f.title.as_deref().unwrap_or("(untitled)");
                md.push_str(&format!("- [{}]({})", title, f.url));
                if let Some(p) = &f.price {
                    md.push_str(&format!(" — **{p}**"));
                }
                if let Some(d) = &f.listing_date {
                    md.push_str(&format!(" _{d}_"));
                }
                md.push('\n');
                if let Some(ex) = &f.excerpt
                    && !ex.is_empty()
                {
                    md.push_str(&format!("  > {ex}\n"));
                }
            }
            md.push('\n');
        }

        md.push_str("## Run history (last 10)\n\n");
        if runs.is_empty() {
            md.push_str("_(none)_\n\n");
        } else {
            md.push_str("| Started | Provider | Model | New | Total | Reason |\n");
            md.push_str("|---------|----------|-------|----:|------:|--------|\n");
            for r in runs.iter().rev() {
                md.push_str(&format!(
                    "| {} | {} | {} | {} | {} | {} |\n",
                    r.started_at.format("%Y-%m-%d %H:%M UTC"),
                    r.provider,
                    r.model,
                    r.new_findings,
                    r.total_findings_after,
                    r.stop_reason,
                ));
            }
            md.push('\n');
        }

        self.store.write_report(&spec.id, &md).await
    }

    /// Generate a structured brief for the next agent run. Contains all findings
    /// with full detail, run history summary, and actionable guidance so the next
    /// agent knows what was already collected and where to focus.
    pub(crate) async fn regenerate_agent_brief(&self, spec: &ResearchSpec) -> Result<()> {
        let findings = self.store.list_findings(&spec.id, None).await?;
        let runs = self.store.list_runs(&spec.id, Some(5)).await?;

        let mut md = String::new();
        md.push_str(&format!(
            "## Collected findings ({} total)\n\n",
            findings.len()
        ));

        if findings.is_empty() {
            md.push_str("No findings yet.\n\n");
        } else {
            for (i, f) in findings.iter().rev().enumerate() {
                let title = f.title.as_deref().unwrap_or("(untitled)");
                let price = f.price.as_deref().unwrap_or("—");
                let date = f.listing_date.as_deref().unwrap_or("—");
                md.push_str(&format!("### {}. {} — {}\n", i + 1, title, price));
                md.push_str(&format!("- URL: {}\n", f.url));
                md.push_str(&format!("- Date: {}\n", date));
                if let Some(ex) = &f.excerpt {
                    md.push_str(&format!("- Details: {ex}\n"));
                }
                md.push('\n');
            }
        }

        if !runs.is_empty() {
            md.push_str("## Run history\n\n");
            for r in runs.iter().rev() {
                md.push_str(&format!(
                    "- {} via {}/{}: +{} new (total {}), stopped: {}\n",
                    r.started_at.format("%Y-%m-%d %H:%M UTC"),
                    r.provider,
                    r.model,
                    r.new_findings,
                    r.total_findings_after,
                    r.stop_reason,
                ));
            }
            md.push('\n');
        }

        md.push_str("## Guidance for this run\n\n");
        md.push_str(
            "- Skip all URLs already in the Known findings list.\n\
             - Focus on NEW listings posted since the last run.\n\
             - Explore deeper pages (page 2+) and alternative sources not yet tried.\n\
             - Prioritize listings with recent dates.\n",
        );

        self.store.write_agent_brief(&spec.id, &md).await
    }

    // REGISTRY-WAIVE: too_many_arguments — refactor-defer, signature complexity acceptable
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn write_record(
        &self,
        spec: &ResearchSpec,
        run_id: &str,
        new_findings: u32,
        reason: StopReason,
        started: std::time::Instant,
        provider: &str,
        model: &str,
    ) -> Result<RunReport> {
        self.write_record_with_verification(
            spec,
            run_id,
            new_findings,
            reason,
            started,
            provider,
            model,
            None,
        )
        .await
    }

    /// Variant of [`write_record`] that records gatekeeper verification stats.
    /// Pass `verification = None` for plain `run_once` rows, or `Some(stats)`
    /// for the final record emitted by `run_verified`.
    // REGISTRY-WAIVE: too_many_arguments — refactor-defer, signature complexity acceptable
    #[allow(clippy::too_many_arguments)]
    // REGISTRY-WAIVE: too_many_arguments — refactor-defer, signature complexity acceptable
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn write_record_with_verification(
        &self,
        spec: &ResearchSpec,
        run_id: &str,
        new_findings: u32,
        reason: StopReason,
        started: std::time::Instant,
        provider: &str,
        model: &str,
        verification: Option<VerificationSummary>,
    ) -> Result<RunReport> {
        let total = self.store.count_findings(&spec.id).await.unwrap_or(0);
        let elapsed = started.elapsed();
        let elapsed_secs = elapsed.as_secs();
        let record = RunRecord {
            run_id: run_id.to_string(),
            spec_id: spec.id.clone(),
            started_at: Utc::now() - chrono::Duration::from_std(elapsed).unwrap_or_default(),
            finished_at: Utc::now(),
            new_findings,
            total_findings_after: total,
            stop_reason: reason.as_str().to_string(),
            provider: provider.to_string(),
            model: model.to_string(),
            verification_rounds: verification.as_ref().map(|v| v.rounds),
            dead_removed: verification.as_ref().map(|v| v.dead_removed),
            replacements_found: verification.as_ref().map(|v| v.replacements_found),
            remaining_issues: verification.as_ref().map(|v| v.remaining_issues),
            elapsed_secs: Some(elapsed_secs),
        };
        if let Err(e) = self.store.append_run(&record).await {
            tracing::warn!(spec = %spec.id, "failed to append run record: {e}");
        }
        Ok(RunReport {
            spec_id: spec.id.clone(),
            run_id: run_id.to_string(),
            new_findings,
            total_findings_after: total,
            stop_reason: reason,
            elapsed,
            provider: provider.to_string(),
            model: model.to_string(),
        })
    }
}
