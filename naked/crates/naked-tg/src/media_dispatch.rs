//! Media extraction, processing, and dispatch helpers.
//!
//! Extracted from main.rs. Handles Telegram media items (photos, documents,
//! audio, video, stickers), native image routing, file downloads, and
//! media-group processing.

use super::*;

// ── Media extraction & dispatch ─────────────────────────────────────────

/// Shared context for media processing (kills 2× too_many_arguments).
pub(crate) struct MediaCtx<'a> {
    pub bot_token: &'a str,
    pub config: &'a Config,
    pub http: Arc<reqwest::Client>,
    pub base_url: Arc<String>,
    pub user_caption: Option<&'a str>,
    #[allow(dead_code)] // reserved for future per-message metrics
    pub msg_id: i32,
    pub route_images_natively: bool,
    pub native_cap_bytes: u64,
    pub active_model: &'a str,
}

/// A single media attachment that we're willing to download and process.
///
/// A Telegram message never contains more than one media "kind" at a time
/// (photo+video are mutually exclusive), but we still return a `Vec` so the
/// caller can decide the policy in one place.
#[derive(Debug, Clone)]
pub(crate) struct MediaItem {
    pub(crate) kind: media::MediaKind,
    pub(crate) file_id: String,
    pub(crate) file_name: String,
    pub(crate) mime_hint: Option<String>,
    pub(crate) duration_secs: Option<u64>,
    pub(crate) emoji: Option<String>,
    /// Raw byte size hint from Telegram metadata (best PhotoSize / file.size).
    /// Used to short-circuit downloads or skip native vision routing **before**
    /// we spend bandwidth pulling the file from `api.telegram.org`.
    pub(crate) size_hint: Option<u32>,
    /// Sticker format. `None` for non-stickers. Animated/video stickers
    /// (`.tgs` / `.webm`) cannot be consumed by vision models, so we skip the
    /// native multimodal path for them.
    pub(crate) sticker_format: Option<StickerFormat>,
}

/// Subset of `teloxide_types::sticker::StickerFormat` we need locally.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StickerFormat {
    /// Static `.webp` raster — vision-model compatible.
    Static,
    /// Animated `.tgs` (Lottie) — NOT vision-compatible.
    Animated,
    /// Video `.webm` — NOT vision-compatible.
    Video,
}

pub(crate) fn extract_media_items(msg: &Message) -> Vec<MediaItem> {
    let mid = msg.id.0;

    if let Some(v) = msg.voice() {
        return vec![MediaItem {
            kind: media::MediaKind::Voice,
            file_id: v.file.id.clone().to_string(),
            file_name: format!("voice_{mid}.ogg"),
            mime_hint: v.mime_type.as_ref().map(|m| m.to_string()),
            duration_secs: Some(u64::from(v.duration.seconds())),
            emoji: None,
            size_hint: Some(v.file.size),
            sticker_format: None,
        }];
    }
    if let Some(a) = msg.audio() {
        let ext = a
            .mime_type
            .as_ref()
            .and_then(|m| mime_ext(m.essence_str()))
            .unwrap_or("bin");
        let name = a
            .file_name
            .clone()
            .unwrap_or_else(|| format!("audio_{mid}.{ext}"));
        return vec![MediaItem {
            kind: media::MediaKind::Audio,
            file_id: a.file.id.clone().to_string(),
            file_name: name,
            mime_hint: a.mime_type.as_ref().map(|m| m.to_string()),
            duration_secs: Some(u64::from(a.duration.seconds())),
            emoji: None,
            size_hint: Some(a.file.size),
            sticker_format: None,
        }];
    }
    if let Some(photos) = msg.photo()
        && let Some(best) = photos.iter().max_by_key(|p| p.width * p.height)
    {
        return vec![MediaItem {
            kind: media::MediaKind::Photo,
            file_id: best.file.id.clone().to_string(),
            file_name: format!("photo_{mid}.jpg"),
            mime_hint: Some("image/jpeg".to_string()),
            duration_secs: None,
            emoji: None,
            size_hint: Some(best.file.size),
            sticker_format: None,
        }];
    }
    if let Some(v) = msg.video() {
        let name = v
            .file_name
            .clone()
            .unwrap_or_else(|| format!("video_{mid}.mp4"));
        return vec![MediaItem {
            kind: media::MediaKind::Video,
            file_id: v.file.id.clone().to_string(),
            file_name: name,
            mime_hint: v.mime_type.as_ref().map(|m| m.to_string()),
            duration_secs: Some(u64::from(v.duration.seconds())),
            emoji: None,
            size_hint: Some(v.file.size),
            sticker_format: None,
        }];
    }
    if let Some(a) = msg.animation() {
        let name = a
            .file_name
            .clone()
            .unwrap_or_else(|| format!("animation_{mid}.mp4"));
        return vec![MediaItem {
            kind: media::MediaKind::Animation,
            file_id: a.file.id.clone().to_string(),
            file_name: name,
            mime_hint: a.mime_type.as_ref().map(|m| m.to_string()),
            duration_secs: Some(u64::from(a.duration.seconds())),
            emoji: None,
            size_hint: Some(a.file.size),
            sticker_format: None,
        }];
    }
    if let Some(d) = msg.document() {
        let name = d
            .file_name
            .clone()
            .unwrap_or_else(|| format!("document_{mid}.bin"));
        return vec![MediaItem {
            kind: media::MediaKind::Document,
            file_id: d.file.id.clone().to_string(),
            file_name: name,
            mime_hint: d.mime_type.as_ref().map(|m| m.to_string()),
            duration_secs: None,
            emoji: None,
            size_hint: Some(d.file.size),
            sticker_format: None,
        }];
    }
    if let Some(s) = msg.sticker() {
        let format = if s.is_animated() {
            StickerFormat::Animated
        } else if s.is_video() {
            StickerFormat::Video
        } else {
            StickerFormat::Static
        };
        // .tgs is Lottie JSON, .webm is video — vision models can't ingest
        // either. Use the proper extension so the on-disk artifact is sane.
        let ext = match format {
            StickerFormat::Static => "webp",
            StickerFormat::Animated => "tgs",
            StickerFormat::Video => "webm",
        };
        let mime = match format {
            StickerFormat::Static => "image/webp",
            StickerFormat::Animated => "application/x-tgsticker",
            StickerFormat::Video => "video/webm",
        };
        return vec![MediaItem {
            kind: media::MediaKind::Sticker,
            file_id: s.file.id.clone().to_string(),
            file_name: format!("sticker_{mid}.{ext}"),
            mime_hint: Some(mime.to_string()),
            duration_secs: None,
            emoji: s.emoji.clone(),
            size_hint: Some(s.file.size),
            sticker_format: Some(format),
        }];
    }
    Vec::new()
}

pub(crate) fn mime_ext(mime: &str) -> Option<&'static str> {
    match mime {
        "audio/ogg" | "audio/opus" => Some("ogg"),
        "audio/mpeg" | "audio/mp3" => Some("mp3"),
        "audio/x-wav" | "audio/wav" => Some("wav"),
        "audio/flac" | "audio/x-flac" => Some("flac"),
        "audio/mp4" | "audio/m4a" | "audio/x-m4a" => Some("m4a"),
        _ => None,
    }
}

/// Format a short `MM:SS` duration, capped at `99:59`.
pub(crate) fn fmt_duration(secs: u64) -> String {
    let total = secs.min(60 * 99 + 59);
    format!("{:02}:{:02}", total / 60, total % 60)
}

/// Outcome of processing a batch of Telegram media items: the text block to
/// prepend to the user prompt, plus any **raw** image bytes that should be
/// passed natively to the main model as image content blocks.
///
/// `native_images` is non-empty only when the caller passed
/// `route_images_natively = true` (i.e. the active model is vision-capable
/// and `tg_media.native_image_context` is on). In all other cases images are
/// described via `tg_media.vision` and the description is folded into `text`.
#[derive(Debug, Default)]
pub struct MediaProcessed {
    pub text: String,
    pub native_images: Vec<NativeImage>,
}

#[derive(Debug, Clone)]
pub struct NativeImage {
    pub mime: String,
    pub bytes: Vec<u8>,
}

/// Process one or more media items → produce the user-facing "media block"
/// that gets prepended to the agent prompt. Errors degrade to `[⚠ … error: …]`
/// and a path-only fallback whenever we've managed to save the file.
pub(crate) async fn process_media_items(items: &[MediaItem], ctx: &MediaCtx<'_>) -> MediaProcessed {
    let mut text_blocks: Vec<String> = Vec::with_capacity(items.len());
    let mut native_images: Vec<NativeImage> = Vec::new();
    for item in items {
        match process_one_media(item, ctx).await {
            Ok(out) => {
                text_blocks.push(out.text);
                native_images.extend(out.native_images);
            }
            Err(e) => {
                let safe = redact_for_log(&e);
                tracing::warn!(kind = item.kind.as_str(), error = %safe, "media processing failed");
                text_blocks.push(format!("[\u{26A0} {} error: {}]", item.kind.as_str(), safe));
            }
        }
    }
    MediaProcessed {
        text: text_blocks.join("\n\n"),
        native_images,
    }
}

/// Pure routing predicate: should this `MediaItem` be sent through the **native**
/// multimodal path (raw image bytes attached to the chat request) or fall back to
/// the legacy describer path (vision-provider summary inlined as text)?
///
/// The native path is taken only when **all** are true:
///   * the caller already decided the model+config support native routing,
///   * the item is a `Photo` or a static `Sticker` (animated `.tgs` and video
///     `.webm` cannot be ingested by Anthropic / OpenAI vision endpoints),
///   * the size hint from Telegram metadata fits under `native_image_max_bytes`
///     (pre-download guard — saves bandwidth and avoids API "image too large"
///     errors at request time).
///
/// Non-photo / non-sticker media (audio, video, files, …) bypass this predicate
/// — the caller's `route_images_natively` flag is forwarded as-is for them so the
/// rest of `process_one_media` can keep its branching logic uniform.
/// Sniff the first few bytes for a recognised image-format magic header.
/// Used as a defence-in-depth check before attaching `bytes` as raw image
/// content to a vision API: if the magic is wrong (truncated download,
/// mis-typed mime, repurposed extension), the upstream API will reject the
/// request with an opaque 400 — instead we downgrade to the describer path,
/// which can at least produce a useful "I cannot decode this image" reply.
///
/// Recognised: JPEG (`FF D8 FF`), PNG (`89 50 4E 47 0D 0A 1A 0A`), GIF
/// (`GIF87a` / `GIF89a`), WebP (`RIFF....WEBP`). Returns `false` for any
/// payload shorter than 12 bytes or whose header doesn't match.
pub(crate) fn looks_like_supported_image(bytes: &[u8]) -> bool {
    if bytes.len() < 12 {
        return false;
    }
    if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return true;
    }
    if bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]) {
        return true;
    }
    if &bytes[..6] == b"GIF87a" || &bytes[..6] == b"GIF89a" {
        return true;
    }
    if &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        return true;
    }
    false
}

pub(crate) fn decide_native_route(
    item: &MediaItem,
    route_images_natively: bool,
    native_cap: u32,
) -> bool {
    if !matches!(
        item.kind,
        media::MediaKind::Photo | media::MediaKind::Sticker
    ) {
        return route_images_natively;
    }
    let static_image = matches!(item.sticker_format, None | Some(StickerFormat::Static));
    let too_big = item.size_hint.is_some_and(|s| s > native_cap);
    route_images_natively && static_image && !too_big
}

pub(crate) async fn process_one_media(
    item: &MediaItem,
    ctx: &MediaCtx<'_>,
) -> anyhow::Result<MediaProcessed> {
    let bot_token = ctx.bot_token;
    let config = ctx.config;
    let http = ctx.http.clone();
    let base_url = ctx.base_url.clone();
    let user_caption = ctx.user_caption;
    let route_images_natively = ctx.route_images_natively;
    let native_cap_bytes = ctx.native_cap_bytes;
    let active_model = ctx.active_model;
    use media::MediaKind;
    let pre_model_for_err = active_model.to_string();

    // Pre-download routing: if Telegram already told us this image is bigger
    // than the effective native cap (per-provider floor of the global
    // `native_image_max_bytes`), force the legacy (text-describer) path so we
    // don't waste bandwidth pulling a file we'd discard for native routing
    // anyway. The download still happens (we always need the bytes for either
    // the describer or the file artifact), but `route_natively` is downgraded
    // here, which surfaces the right routing decision in logs.
    let native_cap_u32 = u32::try_from(native_cap_bytes).unwrap_or(u32::MAX);
    let route_natively = decide_native_route(item, route_images_natively, native_cap_u32);
    if matches!(
        item.kind,
        media::MediaKind::Photo | media::MediaKind::Sticker
    ) {
        let static_image = matches!(item.sticker_format, None | Some(StickerFormat::Static));
        if route_images_natively && !static_image {
            tracing::info!(
                kind = item.kind.as_str(),
                fmt = ?item.sticker_format,
                "downgrading native route: animated/video stickers not vision-capable"
            );
        }
        if route_images_natively && item.size_hint.is_some_and(|s| s > native_cap_u32) {
            tracing::info!(
                kind = item.kind.as_str(),
                size_hint = item.size_hint,
                native_cap = native_cap_u32,
                "downgrading native route: image larger than native_image_max_bytes"
            );
        }
    }

    let cap = item.kind.cap(&config.tg_media);
    let dl = media::download_to_artifacts(
        http.clone(),
        bot_token,
        &base_url,
        &item.file_id,
        &config.workspace,
        &item.file_name,
        cap,
    )
    .await?;
    let mime = item.mime_hint.as_deref().unwrap_or(&dl.mime);
    let size = dl.bytes.len();
    let rel_path = dl.path.display().to_string();

    match item.kind {
        MediaKind::Voice | MediaKind::Audio => {
            let dur = item
                .duration_secs
                .map(fmt_duration)
                .unwrap_or_else(|| "?".to_string());
            let kind_label = if matches!(item.kind, MediaKind::Voice) {
                "\u{1F3A4} voice"
            } else {
                "\u{1F3B5} audio"
            };
            let header = format!("[{kind_label} {dur}, saved: {rel_path}]");
            let text = match &config.tg_media.audio {
                Some(audio_cfg) => {
                    match media::transcribe_audio(
                        http.clone(),
                        audio_cfg,
                        &dl.bytes,
                        &item.file_name,
                        mime,
                    )
                    .await
                    {
                        Ok(text) => format!("{header}\nTranscript:\n{text}"),
                        Err(e) => {
                            let safe = redact_for_log(&e);
                            tracing::warn!(error = %safe, "transcription failed, falling back to path");
                            format!("{header}\n[\u{26A0} transcription failed: {safe}]")
                        }
                    }
                }
                None => {
                    format!("{header}\n[transcription not configured \u{2014} set tg_media.audio]")
                }
            };
            Ok(MediaProcessed {
                text,
                native_images: Vec::new(),
            })
        }
        MediaKind::Photo | MediaKind::Sticker => {
            let kind_label = if matches!(item.kind, MediaKind::Photo) {
                "\u{1F4F8} photo"
            } else {
                "\u{1F5BC} sticker"
            };
            let emoji = item
                .emoji
                .as_deref()
                .map(|e| format!(" {e}"))
                .unwrap_or_default();
            let header = format!("[{kind_label}{emoji}, saved: {rel_path}]");

            // Native path: hand raw bytes to the main model. We still emit a
            // tiny text header so the conversation log is human-readable and
            // the artifact path is preserved for `read_file` retrieval.
            //
            // Animated/video stickers (`.tgs`, `.webm`) are NOT consumable by
            // vision models — Anthropic and OpenAI both reject them — so we
            // force the legacy (text-describer) path for those, which will at
            // least produce a fallback "[sticker not supported]" line instead
            // of an opaque API 4xx.
            let native_cap = native_cap_bytes as usize;
            // Magic-byte sniff: if Telegram tagged the file `image/*` but the
            // payload obviously isn't (truncated / malformed / wrong mime),
            // skip the native path. The describer often produces a useful
            // "I can't read this image" reply, while a vision provider would
            // return an opaque 400.
            let valid_image_magic = looks_like_supported_image(&dl.bytes);
            let native_ok = route_natively
                && size > 0
                && size <= native_cap
                && mime.starts_with("image/")
                && valid_image_magic;
            if route_natively && !valid_image_magic && mime.starts_with("image/") {
                tracing::warn!(
                    bytes = size,
                    mime = %mime,
                    "downgrading native route: payload does not match a known image magic header"
                );
            }
            if native_ok {
                let user_caption_part = user_caption
                    .map(|c| format!("\nUser caption: {c}"))
                    .unwrap_or_default();
                return Ok(MediaProcessed {
                    text: format!("{header}{user_caption_part}"),
                    native_images: vec![NativeImage {
                        mime: mime.to_string(),
                        bytes: dl.bytes.clone(),
                    }],
                });
            }
            // Animated/video stickers: vision providers can't ingest .tgs/.webm
            // either — short-circuit with a clear note instead of pretending
            // to call the describer (which would error out anyway).
            let is_animated = matches!(
                item.sticker_format,
                Some(StickerFormat::Animated) | Some(StickerFormat::Video)
            );
            let text = if is_animated {
                let fmt = match item.sticker_format {
                    Some(StickerFormat::Animated) => "animated (.tgs)",
                    Some(StickerFormat::Video) => "video (.webm)",
                    _ => "non-static",
                };
                format!(
                    "{header}\n[{fmt} sticker \u{2014} vision models cannot describe this format; artifact saved on disk]"
                )
            } else {
                // Legacy text-only path: describe via vision provider, inline the text.
                match &config.tg_media.vision {
                    Some(vision_cfg) => {
                        match media::describe_image(
                            http.clone(),
                            vision_cfg,
                            &dl.bytes,
                            mime,
                            user_caption,
                        )
                        .await
                        {
                            Ok(text) => format!("{header}\nDescription:\n{text}"),
                            Err(e) => {
                                // M5/B01 DRY: classifier lifted to media.rs.
                                let reason = media::classify_media_error(&e);
                                let hint = match reason {
                                    "rate_limit" => {
                                        " (rate-limited \u{2014} the describer provider \
                                                     is throttling; consider a different model in \
                                                     `tg_media.vision`)"
                                    }
                                    "auth" => {
                                        " (auth rejected \u{2014} check the describer's \
                                                `api_key`)"
                                    }
                                    "payload" => {
                                        " (image too large for the describer; lower \
                                                   `tg_media.limits.photo_max_bytes` or pick a \
                                                   model with a higher cap)"
                                    }
                                    "timeout" => {
                                        " (describer timed out; the upstream is slow \
                                                   or unreachable)"
                                    }
                                    _ => "",
                                };
                                let safe = redact_for_log(&e);
                                tracing::warn!(
                                    error = %safe,
                                    reason,
                                    classified = %hint,
                                    "vision describer failed"
                                );
                                let msg = safe;
                                let tail = msg.chars().take(180).collect::<String>();
                                format!(
                                    "{header}\n[\u{26A0} vision describer failed{hint}: {tail}]"
                                )
                            }
                        }
                    }
                    None => {
                        format!(
                            "{header}\n[\u{26A0} vision not configured \u{2014} the active \
                             model `{model}` doesn't accept images natively and no \
                             `tg_media.vision` describer is set; only the file path was \
                             saved. Either switch to a vision-capable model (e.g. \
                             `gpt-4o`, `claude-sonnet-4`, `llama-4-scout`) or set \
                             `tg_media.vision`.]",
                            model = pre_model_for_err.as_str(),
                        )
                    }
                }
            };
            Ok(MediaProcessed {
                text,
                native_images: Vec::new(),
            })
        }
        MediaKind::Document => {
            let header = format!(
                "[\u{1F4C4} document: {} ({} bytes, {mime}), saved: {rel_path}]",
                item.file_name, size
            );
            let text = if media::is_inlineable_doc(
                &item.file_name,
                mime,
                size as u64,
                config.tg_media.docs_inline_max_bytes,
            ) {
                match std::str::from_utf8(&dl.bytes) {
                    Ok(text) => {
                        let safe = media::truncate_for_inline(text, 64 * 1024);
                        format!("{header}\n```\n{safe}\n```")
                    }
                    Err(_) => format!("{header}\n[binary content \u{2014} path only]"),
                }
            } else {
                header
            };
            Ok(MediaProcessed {
                text,
                native_images: Vec::new(),
            })
        }
        MediaKind::Video | MediaKind::Animation => {
            let dur = item
                .duration_secs
                .map(fmt_duration)
                .unwrap_or_else(|| "?".to_string());
            let kind_label = if matches!(item.kind, MediaKind::Video) {
                "\u{1F3AC} video"
            } else {
                "\u{1F3A1} animation"
            };
            Ok(MediaProcessed {
                text: format!("[{kind_label} {dur}, {size} bytes, saved: {rel_path}]"),
                native_images: Vec::new(),
            })
        }
    }
}

pub(crate) async fn send_text(
    bot: &Bot,
    chat_id: ChatId,
    thread_id: Option<ThreadId>,
    text: &str,
) -> Result<(), teloxide::RequestError> {
    let ctx = crate::shared::ChatCtx {
        chat_id,
        thread_id,
        reply_to: None,
    };
    crate::shared::safe_send(bot, &ctx, text.to_string(), None).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── mime_ext ────────────────────────────────────────────────────

    #[test]
    fn mime_ext_audio_formats() {
        assert_eq!(mime_ext("audio/ogg"), Some("ogg"));
        assert_eq!(mime_ext("audio/opus"), Some("ogg"));
        assert_eq!(mime_ext("audio/mpeg"), Some("mp3"));
        assert_eq!(mime_ext("audio/mp3"), Some("mp3"));
        assert_eq!(mime_ext("audio/x-wav"), Some("wav"));
        assert_eq!(mime_ext("audio/flac"), Some("flac"));
        assert_eq!(mime_ext("audio/mp4"), Some("m4a"));
    }

    #[test]
    fn mime_ext_unknown_returns_none() {
        assert_eq!(mime_ext("video/mp4"), None);
        assert_eq!(mime_ext(""), None);
        assert_eq!(mime_ext("text/plain"), None);
    }

    // ── fmt_duration ────────────────────────────────────────────────

    #[test]
    fn fmt_duration_minutes_seconds() {
        assert_eq!(fmt_duration(0), "00:00");
        assert_eq!(fmt_duration(5), "00:05");
        assert_eq!(fmt_duration(61), "01:01");
        assert_eq!(fmt_duration(3599), "59:59");
    }

    #[test]
    fn fmt_duration_capped_at_99_59() {
        assert_eq!(fmt_duration(99 * 60 + 59), "99:59");
        assert_eq!(fmt_duration(100 * 60), "99:59"); // capped
        assert_eq!(fmt_duration(u64::MAX), "99:59"); // capped
    }

    // ── looks_like_supported_image ──────────────────────────────────

    #[test]
    fn image_detection_jpeg() {
        let mut data = vec![0xFF, 0xD8, 0xFF, 0xE0];
        data.extend_from_slice(&[0; 20]);
        assert!(looks_like_supported_image(&data));
    }

    #[test]
    fn image_detection_png() {
        let mut data = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        data.extend_from_slice(&[0; 20]);
        assert!(looks_like_supported_image(&data));
    }

    #[test]
    fn image_detection_gif() {
        let mut data = b"GIF89a".to_vec();
        data.extend_from_slice(&[0; 20]);
        assert!(looks_like_supported_image(&data));
    }

    #[test]
    fn image_detection_webp() {
        let mut data = b"RIFF".to_vec();
        data.extend_from_slice(&[0; 4]); // size
        data.extend_from_slice(b"WEBP");
        data.extend_from_slice(&[0; 10]);
        assert!(looks_like_supported_image(&data));
    }

    #[test]
    fn image_detection_rejects_small() {
        assert!(!looks_like_supported_image(&[0xFF, 0xD8]));
        assert!(!looks_like_supported_image(&[]));
    }

    #[test]
    fn image_detection_rejects_unknown() {
        assert!(!looks_like_supported_image(&[0; 20]));
    }
}
