# LumaPhoto Changelog

---

## v1.1 — 2026-06-07

### Performance & Responsiveness

- **Async HEIC/HEIF loading** — Opening HEIC images no longer freezes the UI.
  Decoding now runs on a background thread via `Task.Run`; a `CancellationTokenSource`
  ensures switching images quickly cancels any in-flight decode.

- **Parallel filmstrip loading** — The image strip no longer loads one thumbnail at a time.
  Placeholder tiles appear instantly for every image in the folder, then up to 4 thumbnails
  decode in parallel and fill in as they complete (`SemaphoreSlim(4)`).

---

### Looks

- **Custom look × delete button** — Custom looks in the Looks dialog no longer delete on
  left-click. Each row now has a small red × button on the right side.
  - Left-click the row → applies the look.
  - Click × → deletes it (with event propagation stopped so the row click doesn't also fire).

---

### Auto Enhance

- **Accurate scene label** — The Auto Enhance panel now shows the detected scene immediately
  (e.g. *Portrait · ✓*, *Landscape · ✓*, *Sunset · ✓*) instead of the stale
  "AI · analyzing…" placeholder that previously appeared and never resolved.

- **Scene detection accuracy** — Reduced false positives from the rule-based analyser:
  - *Portrait*: skin-ratio threshold raised from 0.13 → 0.18; added guard requiring
    skin pixels to outnumber green pixels, preventing foliage from triggering portrait mode.
  - *Sunset*: warm-ratio threshold raised from 0.12 → 0.22 and luminance range narrowed
    to 60–140, so yellow-background studio shots no longer register as sunsets.

---

### Neural Enhancer (disabled pending training)

- **NN blend disabled** — The neural-enhancer blend step is turned off until the
  MIT-FiveK–trained models (`fivek_expert_c.onnx`, `_a.onnx`, `_e.onnx`) are ready.
  Rule-based auto enhance runs as normal in the meantime.

- **AI Ready startup popup removed** — The diagnostic toast ("AI Ready" / "AI Model Not Found")
  that appeared on every launch has been removed.

---

### Training Pipeline (internal)

- **Dataset switched to MIT-FiveK** — PPR10K dropped (portrait-only bias).
  Now training on Adobe MIT-FiveK with three expert targets:
  - Expert C — natural/balanced (slider centre)
  - Expert A — vibrant (slider right)
  - Expert E — dramatic (slider left)
- **Three-model Auto Enhance slider** — Architecture in place for a
  left (dramatic) → centre (natural) → right (vibrant) blend once models are trained.
- Per-expert Colab checkpoints: `last_expert_c.pt` / `best_expert_c.pt`, etc.

---

## v1.0 — initial release

- Rule-based auto enhance (exposure, contrast, highlights, shadows, saturation, warmth)
- Looks: save/apply/delete custom looks, built-in look presets
- Filters, Crop, Markup tabs
- Smart zoom, recent files
- Copy/paste look between images
- Upscale export
- Slider nudge (arrow keys)
- JPEG / PNG / WebP / TIFF export, batch export
- Single-file self-contained Windows exe
