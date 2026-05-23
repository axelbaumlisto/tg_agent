//! Message handler + BotDeps dispatch layer.
//!
//! Extracted from main.rs.

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

        async move { handle_message(self, msg, extra_album_msgs).await }
            .instrument(span)
            .await
    }
}

/// Core message handler — dispatches text, media, and commands.
///
/// T3 (PLAN_v13_SOLID_AUDIT): takes `BotDeps` instead of 14 individual args.
pub(crate) async fn handle_message(
    deps: BotDeps,
    msg: Message,
    extra_album_msgs: Vec<Message>,
) -> Result<(), teloxide::RequestError> {
    // Borrow from deps; field access via `deps.X` where needed.
    // Short aliases for the most-used fields.
    let bot = &deps.bot;
    let agent = &deps.agent;
    let channel_map = &deps.channel_map;
    let config = &deps.config;
    let pending_perms = &deps.pending_perms;
    let attribution_flag = &deps.attribution_flag;
    let http_client = &deps.http_client;
    let base_url = &deps.base_url;
    let bot_token: &str = &deps.bot_token;
    let bot_identity = &deps.bot_identity;
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
    if !is_allowed(chat_id_raw, config) {
        tracing::warn!("Rejected message from chat_id={chat_id_raw}");
        return Ok(());
    }

    // T3.3 (PLAN_RESEARCH_AGENT_FLOW_v1): Research clarification intercept
    // was here. Removed — a paused research run is now resumed naturally
    // via `/research run <spec>` (or `/abort` cancels the in-flight turn).
    // Clarifications = just type a new message and re-run; no magic
    // PENDING_CLARIFICATIONS state machine, no special restart button.
    // See BUG_REGISTRY B56 (parallel-cancel-mechanisms): keeping the bot
    // free of bespoke clarification state aligns with single-mechanism flow.

    // Group-chat addressing gate. Privacy mode is OFF for this bot
    // (`can_read_all_group_messages: true` from getMe), so Telegram
    // delivers every message in every group the bot has joined. We
    // **read** all of them (logged via the `tg_handle` span above so
    // the LLM-side memory loop can pick them up later if desired) but
    // only **respond** when the message is explicitly addressed to us
    // — see `naked_tg::bot_identity::is_addressed_to_bot` for the
    // exact rules. Private chats always pass this gate.
    if !naked_tg::bot_identity::is_addressed_to_bot(&msg, bot_identity) {
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
            bot,
            ctx.chat_id,
            ctx.thread_id,
            "\u{1F4E5} processing media\u{2026}",
        )
        .await;
        let media_ctx = crate::media_dispatch::MediaCtx {
            bot_token,
            config,
            http: http_client.clone(),
            base_url: base_url.clone(),
            user_caption: caption.as_deref(),
            msg_id: msg.id.0,
            route_images_natively,
            native_cap_bytes,
            active_model: &pre_model,
        };
        Some(process_media_items(&media_items, &media_ctx).await)
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
        if drop_slash_for_persona(bot, chat_id_raw, &msg, config).await {
            return Ok(());
        }
        // `/start@zGsR_bot args` → `/start args` so command parsing
        // doesn't have to know about the @-suffix Telegram appends in
        // groups. `/start@OtherBot` is already filtered out by the
        // addressing gate above (returned as not-addressed), so any
        // `@bot` suffix that survives to this point either targets
        // us or doesn't exist at all.
        let canonical = naked_tg::bot_identity::strip_bot_command_suffix(&text, bot_identity)
            .unwrap_or_else(|| text.clone());
        let handled = handle_command(&deps, &msg, &canonical, ctx, pending_perms).await?;
        if handled {
            return Ok(());
        }
        // Unrecognized /command — fall through to agent as regular message
    }

    // Now we know the message is going to the agent — only now do we create
    // a session if one didn't exist yet.
    let session_id = match existing_session_id {
        Some(sid) => sid,
        None => get_or_create_session(ctx, agent, channel_map, config).await,
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
            // S6: track the ack message id so the streaming pipeline
            // can delete it once the steer is actually delivered
            // (AgentEvent::SteerReceived). Without this the user is
            // left with a permanent "Принято — доставлю" hanging
            // in the chat even after the model already replied.
            let ack = bot
                .send_message(
                    ctx.chat_id,
                    "\u{21a9}\u{fe0f} Принято \u{2014} доставлю между шагами",
                )
                .maybe_thread(ctx.thread_id)
                .maybe_reply_to(ctx.reply_to)
                .await?;
            crate::shared::STEER_ACK_IDS
                .write()
                .await
                .insert((key.0, key.1, msg.id.0), (ctx.chat_id, ack.id));
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

    // Expand @-mentions: @src/main.rs → inject file content.
    let workspace = agent
        .session_workspace(&session_id)
        .await
        .unwrap_or_default();
    let (text, mention_ctx) = naked_core::mentions::expand_mentions(&text, &workspace).await;
    let text = if mention_ctx.is_empty() {
        text
    } else {
        format!("{text}\n{mention_ctx}")
    };

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
            crate::shared::safe_send(bot, &ctx, format!("Error: {e}"), None).await?;
            return Ok(());
        }
    };

    stream_response(&deps, ctx, handle, model_tag).await;

    Ok(())
}

#[cfg(test)]
mod tests {
    //! Source-text sentinel tests for message_handler.rs.
    //!
    //! The handler requires Bot + AgentCore + ChannelSessionMap to run,
    //! which is too heavy for unit tests. These sentinels pin the
    //! structural contracts via include_str!.

    fn source() -> &'static str {
        include_str!("message_handler.rs")
    }

    #[test]
    fn permission_check_before_any_processing() {
        let src = source();
        let allowed_pos = src.find("is_allowed(").expect("is_allowed call must exist");
        let agent_pos = src
            .find("get_or_create_session")
            .expect("session creation must exist");
        assert!(
            allowed_pos < agent_pos,
            "is_allowed must run BEFORE get_or_create_session (authz gate)"
        );
    }

    #[test]
    fn empty_messages_short_circuit() {
        let src = source();
        assert!(
            src.contains("text_direct.is_none() && caption.is_none() && media_items.is_empty()"),
            "empty-message short-circuit must exist"
        );
    }

    #[test]
    fn group_addressing_gate_exists() {
        let src = source();
        assert!(
            src.contains("is_addressed_to_bot"),
            "group addressing gate must exist for group-chat safety"
        );
    }

    #[test]
    fn command_routing_via_handle_command() {
        let src = source();
        assert!(
            src.contains("handle_command("),
            "slash commands must route through handle_command"
        );
    }

    #[test]
    fn media_extraction_before_agent_turn() {
        let src = source();
        let media_pos = src
            .find("extract_media_items(")
            .expect("media extraction must exist");
        let session_pos = src
            .find("get_or_create_session")
            .expect("session creation must exist");
        assert!(
            media_pos < session_pos,
            "media extraction must happen BEFORE session creation"
        );
    }

    #[test]
    fn stream_response_dispatched_via_bot_deps() {
        // T3 contract: stream_response takes &BotDeps, not 10 individual args.
        let src = source();
        assert!(
            src.contains("stream_response(&deps,"),
            "stream_response must be called with &deps (T3 BotDeps pattern)"
        );
    }

    #[test]
    fn album_caption_coalescing() {
        // Telegram only puts caption on first photo in album. The handler
        // must search the whole batch for the first non-empty caption.
        let src = source();
        assert!(
            src.contains("extra_album_msgs.iter()") && src.contains("caption"),
            "album caption coalescing must search extra_album_msgs"
        );
    }
}
