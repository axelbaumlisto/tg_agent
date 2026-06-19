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
    pub research_scheduler:
        Option<std::sync::Arc<crate::shared::research_scheduler::ResearchScheduler>>,
    pub per_chat_locks: Arc<crate::per_chat_locks::PerChatLocks>,
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

    let IncomingContent {
        text_direct,
        caption,
        mut media_items,
    } = collect_incoming_content(&msg, &extra_album_msgs);

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

    let media_processed = process_media_if_any(
        bot,
        ctx,
        &media_items,
        &caption,
        MediaProcessEnv {
            bot_token,
            config,
            http_client,
            base_url,
            msg_id: msg.id.0,
            route_images_natively,
            native_cap_bytes,
            active_model: &pre_model,
        },
    )
    .await;

    // Pass-3: merge media_block + text into a single user prompt.
    let (media_text, native_images) = match media_processed {
        Some(m) => (Some(m.text), m.native_images),
        None => (None, Vec::new()),
    };
    let base_text = build_base_text(media_text, text_direct.as_ref(), caption.as_ref());
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
    let text = compose_agent_text(&msg, &base_text, attribution_flag, in_group, &sender);

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

    let multimodal_blocks = build_multimodal_blocks(&native_images, &text, config);

    let key = crate::per_chat_locks::ChatThreadKey::new(ctx.chat_id.0, ctx.raw_thread_id());
    let guard = deps.per_chat_locks.lock(key).await;
    if guard.waited() {
        crate::metrics::record_concurrent_same_key_turn_wait();
    }

    // Short per-(chat,thread) critical section: session-create plus the
    // new-turn-vs-steer/queue decision and registration. The guard is dropped
    // before the multi-minute stream is awaited below, so a same-chat follow-up
    // can acquire it quickly and route into steer/queue instead of blocking
    // behind the active turn.
    let dispatch = prepare_agent_dispatch(
        &deps,
        ctx,
        &msg,
        existing_session_id,
        text,
        multimodal_blocks,
        (pre_prov, pre_model),
    )
    .await?;
    drop(guard);

    match dispatch {
        PreparedDispatch::StartTurn(turn) => {
            stream_response(&deps, ctx, turn.handle, turn.model_tag).await;
        }
        PreparedDispatch::SteerAck { msg_id, key } => {
            send_steer_ack(bot, ctx, msg_id, key).await?;
        }
        PreparedDispatch::Busy => {
            crate::shared::safe_send(bot, &ctx, crate::ux_text::BUSY_ACK.to_string(), None).await?;
        }
        PreparedDispatch::Error { message } => {
            crate::shared::safe_send(bot, &ctx, message, None).await?;
        }
    }

    Ok(())
}

fn build_multimodal_blocks(
    native_images: &[crate::media_dispatch::NativeImage],
    text: &str,
    config: &Config,
) -> Option<Vec<naked_core::types::ContentBlock>> {
    // Block ordering: images FIRST, text LAST. Anthropic's vision docs
    // recommend placing image blocks before text for best response quality.
    if native_images.is_empty() {
        return None;
    }
    let mut blocks = Vec::new();
    for img in native_images {
        use base64::Engine as _;
        let b64 = base64::engine::general_purpose::STANDARD.encode(&img.bytes);
        blocks.push(naked_core::types::ContentBlock::Image {
            mime: img.mime.clone(),
            data_base64: b64,
            detail: config.tg_media.image_detail,
        });
    }
    if !text.is_empty() {
        blocks.push(naked_core::types::ContentBlock::Text {
            text: text.to_string(),
        });
    }
    Some(blocks)
}

enum PreparedDispatch {
    StartTurn(PreparedTurn),
    SteerAck {
        msg_id: i32,
        key: (i64, Option<i32>),
    },
    Busy,
    Error {
        message: String,
    },
}

struct PreparedTurn {
    handle: AgentHandle,
    model_tag: String,
}

fn should_count_session_busy_ack(steer_succeeded: bool) -> bool {
    !steer_succeeded
}

fn session_busy_dispatch(
    key: (i64, Option<i32>),
    msg_id: i32,
    steer_succeeded: bool,
) -> PreparedDispatch {
    if steer_succeeded {
        PreparedDispatch::SteerAck { msg_id, key }
    } else {
        if should_count_session_busy_ack(steer_succeeded) {
            crate::metrics::record_session_busy_ack();
        }
        PreparedDispatch::Busy
    }
}

async fn prepare_agent_dispatch(
    deps: &BotDeps,
    ctx: ChatCtx,
    msg: &Message,
    existing_session_id: Option<String>,
    text: String,
    multimodal_blocks: Option<Vec<naked_core::types::ContentBlock>>,
    provider_model: (String, String),
) -> Result<PreparedDispatch, teloxide::RequestError> {
    let agent = &deps.agent;
    let channel_map = &deps.channel_map;
    let config = &deps.config;

    // Now we know the message is going to the agent — only now do we create
    // a session if one didn't exist yet. This runs under the per-chat short
    // lock, and ChannelSessionMap also protects the first-create path.
    let session_id = match existing_session_id {
        Some(sid) => sid,
        None => get_or_create_session(ctx, agent, channel_map, config).await,
    };

    if agent.is_session_active(&session_id).await {
        return prepare_active_session_message(ctx, msg, text).await;
    }

    start_new_agent_turn(
        deps,
        ctx,
        msg,
        &session_id,
        text,
        multimodal_blocks,
        provider_model,
    )
    .await
}

async fn start_new_agent_turn(
    deps: &BotDeps,
    ctx: ChatCtx,
    msg: &Message,
    session_id: &str,
    text: String,
    multimodal_blocks: Option<Vec<naked_core::types::ContentBlock>>,
    provider_model: (String, String),
) -> Result<PreparedDispatch, teloxide::RequestError> {
    let agent = &deps.agent;
    let (prov, model) = provider_model;
    let model_tag = format!("{prov}/{model}");

    let sender_id = msg.from.as_ref().map(|u| u.id.0.to_string());
    agent.set_session_sender(session_id, sender_id).await;

    let workspace = agent
        .session_workspace(session_id)
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
                .send_prompt_multimodal(session_id, blocks, text.clone())
                .await
        }
        None => agent.send_prompt(session_id, &text).await,
    };
    let handle = match send_result {
        Ok(h) => h,
        Err(e) => {
            if matches!(&e, naked_core::error::AgentError::SessionBusy(_)) {
                let key = (ctx.chat_id.0, ctx.raw_thread_id());
                let steered = try_steer_active_turn(key, msg.id.0, &text).await;
                return Ok(session_busy_dispatch(key, msg.id.0, steered));
            }
            return Ok(PreparedDispatch::Error {
                message: format!("Error: {e}"),
            });
        }
    };

    crate::streaming::register_turn_routing(ctx, &handle).await;
    Ok(PreparedDispatch::StartTurn(PreparedTurn {
        handle,
        model_tag,
    }))
}

struct IncomingContent {
    text_direct: Option<String>,
    caption: Option<String>,
    media_items: Vec<crate::media_dispatch::MediaItem>,
}

fn collect_incoming_content(msg: &Message, extra_album_msgs: &[Message]) -> IncomingContent {
    // Pass-1: classify incoming content. Either `msg.text()`, or a caption on
    // top of media, or pure media — or literally nothing (service messages).
    let text_direct = msg.text().map(str::to_string).filter(|s| !s.is_empty());
    // Album captions: Telegram only attaches the caption to the **first** photo
    // in the group; for any subsequent message its `.caption()` is empty.
    let caption = std::iter::once(msg)
        .chain(extra_album_msgs.iter())
        .find_map(|m| {
            m.caption()
                .map(str::to_string)
                .filter(|s| !s.trim().is_empty())
        });
    let mut media_items = extract_media_items(msg);
    for em in extra_album_msgs {
        media_items.extend(extract_media_items(em));
    }
    IncomingContent {
        text_direct,
        caption,
        media_items,
    }
}

struct MediaProcessEnv<'a> {
    bot_token: &'a str,
    config: &'a Config,
    http_client: &'a Arc<reqwest::Client>,
    base_url: &'a Arc<String>,
    msg_id: i32,
    route_images_natively: bool,
    native_cap_bytes: u64,
    active_model: &'a str,
}

async fn process_media_if_any(
    bot: &Bot,
    ctx: ChatCtx,
    media_items: &[crate::media_dispatch::MediaItem],
    caption: &Option<String>,
    env: MediaProcessEnv<'_>,
) -> Option<crate::media_dispatch::MediaProcessed> {
    if media_items.is_empty() {
        return None;
    }
    let _ = send_text(
        bot,
        ctx.chat_id,
        ctx.thread_id,
        "\u{1F4E5} processing media\u{2026}",
    )
    .await;
    let media_ctx = crate::media_dispatch::MediaCtx {
        bot_token: env.bot_token,
        config: env.config,
        http: env.http_client.clone(),
        base_url: env.base_url.clone(),
        user_caption: caption.as_deref(),
        msg_id: env.msg_id,
        route_images_natively: env.route_images_natively,
        native_cap_bytes: env.native_cap_bytes,
        active_model: env.active_model,
    };
    Some(process_media_items(media_items, &media_ctx).await)
}

fn build_base_text(
    media_text: Option<String>,
    text_direct: Option<&String>,
    caption: Option<&String>,
) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Some(block) = media_text {
        parts.push(block);
    }
    if let Some(t) = text_direct {
        parts.push(t.clone());
    } else if let Some(c) = caption {
        parts.push(c.clone());
    }
    parts.join("\n\n")
}

fn compose_agent_text(
    msg: &Message,
    base_text: &str,
    attribution_flag: &Arc<std::sync::atomic::AtomicBool>,
    in_group: bool,
    sender: &str,
) -> String {
    if base_text.starts_with('/') {
        return base_text.to_string();
    }
    let quote = extract_reply_context(msg);
    let attr_enabled = attribution_flag.load(std::sync::atomic::Ordering::Relaxed);
    let need_attr = attr_enabled && in_group;
    let attributed = if need_attr {
        format!("{sender}: {base_text}")
    } else {
        base_text.to_string()
    };
    match quote {
        Some(q) => format!("{q}\n\n{attributed}"),
        None => attributed,
    }
}

async fn prepare_active_session_message(
    ctx: ChatCtx,
    msg: &Message,
    text: String,
) -> Result<PreparedDispatch, teloxide::RequestError> {
    let key = (ctx.chat_id.0, ctx.raw_thread_id());
    let steered = try_steer_active_turn(key, msg.id.0, &text).await;
    Ok(session_busy_dispatch(key, msg.id.0, steered))
}

async fn try_steer_active_turn(key: (i64, Option<i32>), msg_id: i32, text: &str) -> bool {
    let map = STEER_SENDERS.read().await;
    if let Some(steer_tx) = map.get(&key) {
        let steer_msg = naked_core::types::SteerMessage {
            msg_id,
            text: text.to_string(),
            is_edit: false,
        };
        steer_try_send(steer_tx, steer_msg)
    } else {
        false
    }
}

fn steer_try_send(
    steer_tx: &tokio::sync::mpsc::Sender<naked_core::types::SteerMessage>,
    steer_msg: naked_core::types::SteerMessage,
) -> bool {
    steer_tx.try_send(steer_msg).is_ok()
}

async fn send_steer_ack(
    bot: &Bot,
    ctx: ChatCtx,
    msg_id: i32,
    key: (i64, Option<i32>),
) -> Result<(), teloxide::RequestError> {
    let ack = bot
        .send_message(ctx.chat_id, crate::ux_text::STEER_ACK)
        .maybe_thread(ctx.thread_id)
        .maybe_reply_to(ctx.reply_to)
        .await?;
    crate::shared::STEER_ACK_IDS
        .write()
        .await
        .insert((key.0, key.1, msg_id), (ctx.chat_id, ack.id));
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

    #[test]
    fn session_busy_ack_counter_bumps_once_per_busy_ack() {
        assert!(super::should_count_session_busy_ack(false));
        assert!(!super::should_count_session_busy_ack(true));

        let key = (42, Some(7));
        let msg_id = 1001;
        let before = crate::metrics::snapshot().session_busy_ack;
        let busy = super::session_busy_dispatch(key, msg_id, false);
        let after_busy = crate::metrics::snapshot().session_busy_ack;
        assert_eq!(after_busy, before + 1);
        assert!(matches!(busy, super::PreparedDispatch::Busy));

        let steered = super::session_busy_dispatch(key, msg_id, true);
        let after_steer = crate::metrics::snapshot().session_busy_ack;
        assert_eq!(after_steer, after_busy);
        assert!(matches!(
            steered,
            super::PreparedDispatch::SteerAck {
                msg_id: actual_msg_id,
                key: actual_key,
            } if actual_msg_id == msg_id && actual_key == key
        ));
    }

    #[test]
    fn ux_text_busy_ack_is_honest_and_busy_arm_uses_constant() {
        let busy_ack = crate::ux_text::BUSY_ACK;
        assert!(
            busy_ack.contains("не принято"),
            "BUSY_ACK must explicitly say the message was not accepted"
        );
        for forbidden in ["очеред", "queued", "queue", "сохран", "saved"] {
            assert!(
                !busy_ack.to_lowercase().contains(forbidden),
                "BUSY_ACK must not imply queueing/saving via {forbidden:?}: {busy_ack}"
            );
        }

        let src = source();
        let dispatch_match = src
            .split("match dispatch")
            .nth(1)
            .and_then(|tail| tail.split("fn build_multimodal_blocks").next())
            .expect("dispatch match body must be findable");
        assert!(
            dispatch_match.contains("crate::ux_text::BUSY_ACK"),
            "Busy dispatch arm must use the central BUSY_ACK constant"
        );
        assert!(
            dispatch_match.contains("send_steer_ack(bot, ctx, msg_id, key)"),
            "SteerAck dispatch arm should stay delegated to send_steer_ack"
        );

        let steer_ack_fn = src
            .split("async fn send_steer_ack")
            .nth(1)
            .and_then(|tail| tail.split("#[cfg(test)]").next())
            .expect("send_steer_ack body must be findable");
        assert!(
            steer_ack_fn.contains("crate::ux_text::STEER_ACK"),
            "send_steer_ack must use the central STEER_ACK constant"
        );
    }

    #[test]
    fn active_session_busy_dispatch_steers_or_soft_busy_without_queueing() {
        let key = (42, Some(7));
        let msg_id = 1001;

        let steered = super::session_busy_dispatch(key, msg_id, true);
        assert!(matches!(
            steered,
            super::PreparedDispatch::SteerAck {
                msg_id: actual_msg_id,
                key: actual_key,
            } if actual_msg_id == msg_id && actual_key == key
        ));
        assert!(
            !matches!(steered, super::PreparedDispatch::Error { ref message } if message.starts_with("Error:")),
            "SessionBusy with steer available must not render as Error:"
        );

        let busy = super::session_busy_dispatch(key, msg_id, false);
        assert!(matches!(busy, super::PreparedDispatch::Busy));
        assert!(
            !matches!(busy, super::PreparedDispatch::Error { ref message } if message.starts_with("Error:")),
            "SessionBusy without steer sender must be a soft Busy ack, not Error:"
        );

        let src = source();
        let active_fn = src
            .split("async fn prepare_active_session_message")
            .nth(1)
            .and_then(|tail| tail.split("async fn try_steer_active_turn").next())
            .expect("prepare_active_session_message body must be findable");
        assert!(
            !contains_queue_call(active_fn),
            "active-session no-steer path must steer-or-Busy, never append to live Session.history"
        );
        assert!(
            active_fn.contains("session_busy_dispatch(key, msg.id.0, steered)"),
            "active-session path must reuse the steer-or-Busy decision helper"
        );
    }

    #[tokio::test]
    async fn active_session_busy_full_steer_channel_returns_busy() {
        let key = (-9_876_543_210, Some(31_415));
        let msg_id = 2002;
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        tx.try_send(naked_core::types::SteerMessage {
            msg_id: 1,
            text: "fills the channel".to_string(),
            is_edit: false,
        })
        .expect("first steer send should fill the bounded channel");
        let steered = super::steer_try_send(
            &tx,
            naked_core::types::SteerMessage {
                msg_id,
                text: "follow-up while full".to_string(),
                is_edit: false,
            },
        );

        assert!(!steered, "full steer channel must be treated as no-steer");
        assert!(matches!(
            super::session_busy_dispatch(key, msg_id, steered),
            super::PreparedDispatch::Busy
        ));
    }

    #[test]
    fn queue_message_call_sites_are_classified_with_rationale() {
        use std::path::{Path, PathBuf};

        #[derive(Debug)]
        struct Allow<'a> {
            file: &'a str,
            needle: String,
            rationale: &'a str,
        }

        let q = "queue_message";
        let allowlist = [
            Allow {
                file: "src/callbacks/model.rs",
                needle: format!("agent.{q}(sid, &continuation).await;"),
                rationale: "B3c: model-switch continuation after abort; intentional post-abort delivery, shares §R4 risk and is tracked/allowlisted.",
            },
            Allow {
                file: "src/runtime.rs",
                needle: format!(".{q}(\n                    &sid,"),
                rationale: "B3d: thumbs-down advisory correction note; best-effort non-user-authored content, shares §R4 risk and is tracked/allowlisted.",
            },
            Allow {
                file: "../naked-core/src/test_support.rs",
                needle: format!("tc.core.{q}(&sid, \"hello\").await;"),
                rationale: "test-only helper smoke test; not a production live-history append path.",
            },
        ];
        assert!(
            allowlist.iter().all(|entry| !entry.rationale.is_empty()),
            "every queue-message allowlist entry must carry an explicit rationale"
        );

        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let roots = [
            (manifest.join("src"), "src"),
            (manifest.join("../naked-core/src"), "../naked-core/src"),
        ];
        let mut actual = Vec::new();
        for (root, rel_root) in roots {
            collect_queue_message_calls(&root, rel_root, &mut actual);
        }
        actual.sort();

        for entry in &allowlist {
            let path = manifest.join(entry.file);
            let src = std::fs::read_to_string(&path)
                .unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()));
            assert!(
                src.contains(&entry.needle),
                "allowlisted queue_message site missing or changed in {} ({})",
                entry.file,
                entry.rationale
            );
        }

        let allowed_files: std::collections::BTreeSet<_> = allowlist
            .iter()
            .map(|entry| entry.file.to_string())
            .collect();
        let unclassified: Vec<_> = actual
            .iter()
            .filter(|site| !allowed_files.contains(site.file.as_str()))
            .collect();
        assert!(
            unclassified.is_empty(),
            "unclassified queue_message* call site(s): {unclassified:#?}; add an explicit allowlist+rationale or remove the live-history append"
        );
        assert!(
            actual
                .iter()
                .all(|site| site.file != "src/message_handler.rs"),
            "message_handler.rs must not call queue_message* on active no-steer paths"
        );
        assert_eq!(
            actual.len(),
            allowlist.len(),
            "queue_message* call-site inventory changed: actual={actual:#?} allowlist={allowlist:#?}"
        );

        fn collect_queue_message_calls(root: &Path, rel_root: &str, out: &mut Vec<Site>) {
            for entry in std::fs::read_dir(root)
                .unwrap_or_else(|err| panic!("failed to read dir {}: {err}", root.display()))
            {
                let path = entry.expect("dir entry must be readable").path();
                if path.is_dir() {
                    let child_rel_root = format!(
                        "{rel_root}/{}",
                        path.file_name()
                            .expect("directory must have a file name")
                            .to_string_lossy()
                    );
                    collect_queue_message_calls(&path, &child_rel_root, out);
                } else if path.extension().is_some_and(|ext| ext == "rs") {
                    let src = std::fs::read_to_string(&path)
                        .unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()));
                    let rel = format!(
                        "{rel_root}/{}",
                        path.file_name()
                            .expect("source file must have a file name")
                            .to_string_lossy()
                    );
                    for token in [short_queue_call_token(), multimodal_queue_call_token()] {
                        for (idx, line) in src.lines().enumerate() {
                            if line.contains(&token) {
                                out.push(Site {
                                    file: rel.clone(),
                                    line: idx + 1,
                                    call: token.clone(),
                                    text: line.trim().to_string(),
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    #[derive(Debug, Eq, Ord, PartialEq, PartialOrd)]
    struct Site {
        file: String,
        line: usize,
        call: String,
        text: String,
    }

    fn short_queue_call_token() -> String {
        format!(".{}(", "queue_message")
    }

    fn multimodal_queue_call_token() -> String {
        format!(".{}(", "queue_message_multimodal")
    }

    fn contains_queue_call(src: &str) -> bool {
        src.contains(&short_queue_call_token()) || src.contains(&multimodal_queue_call_token())
    }
}
