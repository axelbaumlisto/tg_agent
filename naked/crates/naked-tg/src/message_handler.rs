//! Message handler + BotDeps dispatch layer.
//!
//! Extracted from main.rs.

use super::fmt_utils::escape_html_min;
use super::*;

#[derive(Clone)]
pub(crate) struct BotDeps {
    pub bot: Bot,
    pub agent: Arc<AgentCore>,
    pub channel_map: Arc<ChannelSessionMap>,
    pub config: Config,
    pub pending_perms: PendingPermissions,
    pub http_client: Arc<reqwest::Client>,
    pub base_url: Arc<String>,
    pub rate_limiter: naked_tg::rate_limit::RateLimiter,
    pub attribution_flag: Arc<std::sync::atomic::AtomicBool>,
    pub bot_token: Arc<String>,
    pub bot_identity: Arc<naked_tg::bot_identity::BotIdentity>,
    pub tg_attach_queue: naked_tg::tg_attach::AttachmentQueue,
}

impl BotDeps {
    /// Entry point matching the old free-function `handle_message`; exists
    /// so callers can write `deps.handle(msg, extras).await` instead of
    /// unpacking eleven arguments at every dispatch site.
    pub async fn handle(
        self,
        msg: Message,
        extra_album_msgs: Vec<Message>,
    ) -> Result<(), teloxide::RequestError> {
        // Every TG message handle gets its own span with (chat_id, msg_id,
        // thread_id) so operators can follow one conversation through the
        // logs without grep-ing by free-form text. `album_size` counts
        // the current + any extras coalesced from a media-group burst.
        use tracing::Instrument;
        let chat_id = msg.chat.id.0;
        let msg_id = msg.id.0;
        let thread_id = msg.thread_id.map(|t| t.0.0).unwrap_or(0);
        let album_size = 1 + extra_album_msgs.len();
        let span = tracing::info_span!(
            "tg_handle",
            chat = chat_id,
            msg = msg_id,
            thread = thread_id,
            album = album_size,
        );

        async move {
            handle_message(
                self.bot,
                msg,
                self.agent,
                self.channel_map,
                self.config,
                self.pending_perms,
                self.http_client,
                self.base_url,
                self.rate_limiter,
                self.attribution_flag,
                self.bot_token,
                self.bot_identity,
                self.tg_attach_queue,
                extra_album_msgs,
            )
            .await
        }
        .instrument(span)
        .await
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn handle_message(
    bot: Bot,
    msg: Message,
    agent: Arc<AgentCore>,
    channel_map: Arc<ChannelSessionMap>,
    config: Config,
    pending_perms: PendingPermissions,
    http_client: Arc<reqwest::Client>,
    base_url: Arc<String>,
    rate_limiter: naked_tg::rate_limit::RateLimiter,
    attribution_flag: Arc<std::sync::atomic::AtomicBool>,
    bot_token: Arc<String>,
    bot_identity: Arc<naked_tg::bot_identity::BotIdentity>,
    tg_attach_queue: naked_tg::tg_attach::AttachmentQueue,
    extra_album_msgs: Vec<Message>,
) -> Result<(), teloxide::RequestError> {
    let ctx = ChatCtx::from_msg(&msg);
    let chat_id_raw = ctx.chat_id.0;

    // Pass-1: classify incoming content. Either `msg.text()`, or a caption on
    // top of media, or pure media — or literally nothing (service messages).
    let text_direct = msg.text().map(str::to_string).filter(|s| !s.is_empty());
    // Album captions: Telegram only attaches the caption to the **first**
    // photo in the group; for any subsequent message its `.caption()` is
    // empty. Search the whole batch for the first non-empty caption so the
    // user's intent isn't dropped just because the first part happened to
    // be processed without a caption.
    let caption = std::iter::once(&msg)
        .chain(extra_album_msgs.iter())
        .find_map(|m| {
            m.caption()
                .map(str::to_string)
                .filter(|s| !s.trim().is_empty())
        });
    let mut media_items = extract_media_items(&msg);
    for em in &extra_album_msgs {
        media_items.extend(extract_media_items(em));
    }

    // Replying to a message with media == "look at THIS". Pull the
    // attachments from the reply target into the current turn so the
    // agent sees actual bytes (and routes through vision / whisper /
    // artifact saving), not just a `[Photo]` placeholder in the quoted
    // reply context.
    //
    // Telegram's `reply_to_message` always points at a single message
    // id — so for an album we only get the specific photo the user
    // tapped Reply on (siblings in the same `media_group_id` aren't
    // reachable via Bot API). Empirically operators usually tap the
    // one they care about most, and we log what we pulled so it's
    // obvious in the trace.
    if let Some(reply) = msg.reply_to_message() {
        let reply_media = extract_media_items(reply);
        if !reply_media.is_empty() {
            tracing::info!(
                chat_id = chat_id_raw,
                count = reply_media.len(),
                reply_msg_id = reply.id.0,
                "pulled media from reply target into current turn"
            );
            media_items.extend(reply_media);
        }
    }

    if text_direct.is_none() && caption.is_none() && media_items.is_empty() {
        return Ok(());
    }

    // Permission check before spending any time on media processing or
    // touching the agent core.
    if !is_allowed(chat_id_raw, &config) {
        tracing::warn!("Rejected message from chat_id={chat_id_raw}");
        return Ok(());
    }

    // Research clarification intercept. A prior `r:stop:<spec>` callback
    // stashed a `PendingClarification` keyed on (chat, thread); the next
    // non-empty text message becomes a topic update + "Restart" prompt.
    // We intercept before the addressing gate because DMs are the normal
    // research delivery surface and we don't want to force a bot mention
    // in private chats just to reply to an inline button.
    if let Some(text) = text_direct
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        && !text.starts_with('/')
    {
        let key = (chat_id_raw, ctx.raw_thread_id());
        let maybe_pending = PENDING_CLARIFICATIONS.write().await.remove(&key);
        if let Some(pending) = maybe_pending {
            let spec_id = pending.spec_id;
            let store = agent.research_store();
            let outcome = match store.load_spec(&spec_id).await {
                Ok(mut spec) => {
                    let stamp = chrono::Utc::now().format("%Y-%m-%d %H:%M UTC").to_string();
                    if !spec.topic.trim_end().ends_with('\n') && !spec.topic.is_empty() {
                        spec.topic.push('\n');
                    }
                    spec.topic.push_str(&format!("\nUPDATE {stamp}: {text}"));
                    store.save_spec(&spec).await
                }
                Err(e) => Err(e),
            };
            match outcome {
                Ok(()) => {
                    let body = format!(
                        "✅ Clarification saved to <code>{}</code>.\nTap below to relaunch with the updated topic.",
                        escape_html_min(&spec_id)
                    );
                    let _ = bot
                        .edit_message_text(ctx.chat_id, pending.message_id, body)
                        .parse_mode(ParseMode::Html)
                        .reply_markup(keyboard_paused_awaiting_clarification(&spec_id))
                        .await;
                }
                Err(e) => {
                    let body = format!(
                        "⚠️ Could not save clarification for <code>{}</code>: {}",
                        escape_html_min(&spec_id),
                        escape_html_min(&format!("{e:#}"))
                    );
                    let _ = bot
                        .send_message(ctx.chat_id, body)
                        .parse_mode(ParseMode::Html)
                        .maybe_thread(ctx.thread_id)
                        .await;
                }
            }
            return Ok(());
        }
    }

    // Group-chat addressing gate. Privacy mode is OFF for this bot
    // (`can_read_all_group_messages: true` from getMe), so Telegram
    // delivers every message in every group the bot has joined. We
    // **read** all of them (logged via the `tg_handle` span above so
    // the LLM-side memory loop can pick them up later if desired) but
    // only **respond** when the message is explicitly addressed to us
    // — see `naked_tg::bot_identity::is_addressed_to_bot` for the
    // exact rules. Private chats always pass this gate.
    if !naked_tg::bot_identity::is_addressed_to_bot(&msg, &bot_identity) {
        // INFO-level on purpose: in groups with privacy-mode OFF this
        // is the only way to confirm "yes, we saw the message, and we
        // intentionally chose not to respond". The volume is bounded
        // by group activity (the bot is silent in DMs — those bypass
        // the gate). If this becomes too chatty in a high-traffic
        // group, demote to debug! and add a counter metric instead.
        tracing::info!(
            chat_id = chat_id_raw,
            addressed = false,
            "group msg not addressed to bot — read but not answered"
        );
        return Ok(());
    }

    // Peek the existing session (without creating one) so we know the active
    // model and can decide whether photos go through the **native multimodal**
    // path (raw bytes → main vision-capable model) or the legacy text-only
    // path (vision provider → text description → main model). We deliberately
    // do NOT call `get_or_create_session` here — that would spawn an empty
    // session for unrecognized commands like `/help`, `/clear`, `/new` sent as
    // the first message in a fresh chat.
    let chat_id_for_peek = ctx.chat_id.0;
    let tid_for_peek = ctx.raw_thread_id();
    let existing_session_id = channel_map.get(chat_id_for_peek, tid_for_peek).await;
    let (pre_prov, pre_model) = match existing_session_id.as_deref() {
        Some(sid) => agent.session_provider_model(sid).await,
        // No session yet — use the global default; this is read-only and never
        // creates session directories on disk.
        None => agent.default_provider_model(),
    };
    let pre_provider_cfg = config.providers.get(&pre_prov);
    let route_images_natively = config.tg_media.native_image_context
        && config
            .tg_media
            .is_vision_capable_with_provider(&pre_model, pre_provider_cfg);
    // Per-provider effective image cap = floor(global cap, provider hard limit).
    let (provider_type, provider_base_url): (String, Option<String>) = match pre_provider_cfg {
        Some(pc) => (pc.provider_type.clone(), pc.base_url.clone()),
        None => (String::new(), None),
    };
    let native_cap_bytes = config
        .tg_media
        .provider_image_cap(&provider_type, provider_base_url.as_deref());
    if !media_items.is_empty() {
        let has_oversize = media_items
            .iter()
            .any(|i| i.size_hint.is_some_and(|s| u64::from(s) > native_cap_bytes));
        tracing::info!(
            model = %pre_model,
            provider = %pre_prov,
            native = route_images_natively,
            native_cap_bytes,
            count = media_items.len(),
            "media routing decision"
        );
        crate::metrics::record_media_routing(
            route_images_natively,
            has_oversize,
            !route_images_natively && config.tg_media.vision.is_some(),
        );
    }

    // Pass-2: if there's media, acknowledge and process it (download + transform).
    let media_processed = if !media_items.is_empty() {
        let _ = send_text(
            &bot,
            ctx.chat_id,
            ctx.thread_id,
            "\u{1F4E5} processing media\u{2026}",
        )
        .await;
        Some(
            process_media_items(
                &media_items,
                &bot_token,
                &config,
                http_client.clone(),
                base_url.clone(),
                caption.as_deref(),
                msg.id.0,
                route_images_natively,
                native_cap_bytes,
                &pre_model,
            )
            .await,
        )
    } else {
        None
    };

    // Pass-3: merge media_block + text into a single user prompt.
    let (media_text, native_images) = match media_processed {
        Some(m) => (Some(m.text), m.native_images),
        None => (None, Vec::new()),
    };
    let base_text = {
        let mut parts: Vec<String> = Vec::new();
        if let Some(block) = media_text {
            parts.push(block);
        }
        if let Some(t) = text_direct.as_ref() {
            parts.push(t.clone());
        } else if let Some(c) = caption.as_ref() {
            parts.push(c.clone());
        }
        parts.join("\n\n")
    };
    if base_text.trim().is_empty() && native_images.is_empty() {
        return Ok(());
    }

    let in_group = is_group_chat(&msg);
    let sender = sender_label(&msg);
    let has_reply = msg.reply_to_message().is_some();
    tracing::info!(
        chat_id = chat_id_raw,
        thread_id = ?ctx.thread_id,
        raw_thread = ?ctx.raw_thread_id(),
        sender = %sender,
        is_group = in_group,
        has_reply,
        media_count = media_items.len(),
        "handle_message: ctx"
    );

    // Compose the final text for the agent: optional reply quote + optional
    // `@sender:` attribution prefix (groups only). Commands skip composition
    // so `/clear`, `/new`, etc. still work when sent as a reply.
    let text = if base_text.starts_with('/') {
        base_text.clone()
    } else {
        let quote = extract_reply_context(&msg);
        let attr_enabled = attribution_flag.load(std::sync::atomic::Ordering::Relaxed);
        let need_attr = attr_enabled && in_group;
        let attributed = if need_attr {
            format!("{sender}: {base_text}")
        } else {
            base_text.clone()
        };
        match quote {
            Some(q) => format!("{q}\n\n{attributed}"),
            None => attributed,
        }
    };

    if text.starts_with('/') {
        // Persona chats with `allow_slash_commands=false` are pure
        // natural-language conversations — drop any `/cmd` here before
        // dispatch and (on the first hit per chat) drop a one-line hint
        // so the operator knows the silence is intentional. See
        // `Config.chat_personas` in `naked-core::config` for the contract.
        if drop_slash_for_persona(&bot, chat_id_raw, &msg, &config).await {
            return Ok(());
        }
        // `/start@zGsR_bot args` → `/start args` so command parsing
        // doesn't have to know about the @-suffix Telegram appends in
        // groups. `/start@OtherBot` is already filtered out by the
        // addressing gate above (returned as not-addressed), so any
        // `@bot` suffix that survives to this point either targets
        // us or doesn't exist at all.
        let canonical = naked_tg::bot_identity::strip_bot_command_suffix(&text, &bot_identity)
            .unwrap_or_else(|| text.clone());
        let handled = handle_command(
            &bot,
            &msg,
            &canonical,
            &agent,
            &channel_map,
            &config,
            ctx,
            &pending_perms,
            &attribution_flag,
        )
        .await?;
        if handled {
            return Ok(());
        }
        // Unrecognized /command — fall through to agent as regular message
    }

    // Now we know the message is going to the agent — only now do we create
    // a session if one didn't exist yet.
    let session_id = match existing_session_id {
        Some(sid) => sid,
        None => get_or_create_session(ctx, &agent, &channel_map, &config).await,
    };

    // Build optional native-multimodal blocks. When present, these go through
    // `send_prompt_multimodal` / `queue_message_multimodal`; otherwise we fall
    // back to the text-only entry points.
    //
    // Block ordering: images FIRST, text LAST. Anthropic's vision docs
    // explicitly recommend placing image blocks before text for best response
    // quality; OpenAI-compatible vision models accept either order. See
    // https://docs.anthropic.com/en/docs/build-with-claude/vision
    let multimodal_blocks: Option<Vec<naked_core::types::ContentBlock>> =
        if !native_images.is_empty() {
            let mut blocks: Vec<naked_core::types::ContentBlock> = Vec::new();
            for img in &native_images {
                use base64::Engine as _;
                let b64 = base64::engine::general_purpose::STANDARD.encode(&img.bytes);
                blocks.push(naked_core::types::ContentBlock::Image {
                    mime: img.mime.clone(),
                    data_base64: b64,
                    detail: config.tg_media.image_detail,
                });
            }
            if !text.is_empty() {
                blocks.push(naked_core::types::ContentBlock::Text { text: text.clone() });
            }
            Some(blocks)
        } else {
            None
        };

    if agent.is_session_active(&session_id).await {
        let key = (ctx.chat_id.0, ctx.raw_thread_id());

        // Try to steer the active turn (inject message into running loop).
        let steered = {
            let map = STEER_SENDERS.read().await;
            if let Some(steer_tx) = map.get(&key) {
                let steer_msg = naked_core::types::SteerMessage {
                    msg_id: msg.id.0,
                    text: text.clone(),
                    is_edit: false,
                };
                steer_tx.try_send(steer_msg).is_ok()
            } else {
                false
            }
        };

        if steered {
            bot.send_message(ctx.chat_id, "\u{21a9}\u{fe0f} Steering")
                .maybe_thread(ctx.thread_id)
                .maybe_reply_to(ctx.reply_to)
                .await?;
        } else {
            // Steer channel not available — fall back to queue.
            match multimodal_blocks {
                Some(blocks) => agent.queue_message_multimodal(&session_id, blocks).await,
                None => agent.queue_message(&session_id, &text).await,
            }
            // Increment queued message counter for status preview.
            {
                let map = QUEUE_COUNTS.read().await;
                if let Some(counter) = map.get(&key) {
                    counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }
            let qcount = {
                let map = QUEUE_COUNTS.read().await;
                map.get(&key)
                    .map(|c| c.load(std::sync::atomic::Ordering::Relaxed))
                    .unwrap_or(0)
            };
            let note = if qcount > 0 {
                format!("⏳ +{qcount} in queue")
            } else {
                "⏳ Queued".into()
            };
            bot.send_message(ctx.chat_id, note)
                .maybe_thread(ctx.thread_id)
                .maybe_reply_to(ctx.reply_to)
                .await?;
        }
        return Ok(());
    }

    // Reuse the (provider, model) we resolved earlier for the multimodal
    // routing decision — for both existing and freshly-created sessions the
    // result is identical, so calling `session_provider_model` again would be
    // a redundant lock acquisition.
    let (prov, model) = (pre_prov, pre_model);
    let model_tag = format!("{prov}/{model}");

    // Record the current author for this turn so the memory tool can resolve
    // `scope=user` without extra parameters. Use the raw numeric id (stable
    // across username changes).
    let sender_id = msg.from.as_ref().map(|u| u.id.0.to_string());
    agent.set_session_sender(&session_id, sender_id).await;

    let send_result = match multimodal_blocks {
        Some(blocks) => {
            agent
                .send_prompt_multimodal(&session_id, blocks, text.clone())
                .await
        }
        None => agent.send_prompt(&session_id, &text).await,
    };
    let handle = match send_result {
        Ok(h) => h,
        Err(e) => {
            bot.send_message(ctx.chat_id, format!("Error: {e}"))
                .maybe_thread(ctx.thread_id)
                .maybe_reply_to(ctx.reply_to)
                .await?;
            return Ok(());
        }
    };

    stream_response(
        bot,
        ctx,
        handle,
        &channel_map,
        &pending_perms,
        model_tag,
        &http_client,
        &base_url,
        &tg_attach_queue,
        &rate_limiter,
    )
    .await;

    Ok(())
}
