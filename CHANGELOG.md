# LumaPhoto Changelog

---

## v1.3 — 2026-06-10

### AI Auto Enhance — fully live 🎉

- **Trained models now bundled** — All three MIT-FiveK expert models ship with the installer.
  The Auto Enhance slider is now a true three-style blend:
  - Slider left → **Dramatic** (Expert E: moody, contrasty)
  - Slider centre → **Natural** (Expert C: balanced, true-to-life)
  - Slider right → **Bright** (Expert A: vibrant, punchy)
- The Auto panel shows *"FiveK E/C/A"* when the models are active.

---

### UI & Workflow

- **Recent photos with thumbnails** — The Recent dialog now shows a small preview
  of each photo next to its name (loaded in the background, HEIC included).
- **✕ close button on dialogs** — Looks, Recent, and other card dialogs now have a
  close button in the upper-right corner.
- **Markup visible everywhere** — Markup strokes now stay visible on every tab
  (Adjust, Filters, Crop, Design, Layers), not just the Markup tab.
  Drawing still happens on the Markup tab only.
- **Font picker redesigned** — Now lists **every font installed on your system**,
  each previewed in its own typeface, in the same dark style as the other dropdowns.

---

### Export

- **More formats** — Export now supports **BMP** and **GIF** in addition to
  JPEG, PNG, and TIFF.
  (WebP and HEIC can be opened but not exported — Windows has no built-in encoders for them.)

---

### Design

- **7 new frame styles** — Double white, Gold, Silver, Walnut wood, Navy,
  Forest green, and Burgundy.
- **Collage: arrange before you create** — After picking photos, a live preview of the
  layout opens. Click two photos to swap their positions, click an empty slot to add
  a photo into it. *Create* is enabled once every slot is filled — no more
  half-empty collages.

---

## v1.2 — 2026-06-08

### Design Tools

- **Collage builder** — Added Split, Stack, Grid, and Feature collage layouts that create a new editable canvas from multiple photos.
- **Photo frames** — Added Classic white, Gallery black, Polaroid, and Soft shadow frame overlays.
- **Watermarks** — Added per-photo watermark text with position, opacity, and size controls.
- **Layers panel** — Added a simple stack view for Watermark, Markup, Frame, and Photo layers, with export visibility toggles.

---

### Auto Enhance Neural Model

- **Neural Auto Enhance re-enabled** — When `enhancer_params.onnx` is present next to the app, Auto Enhance now runs the trained direct-parameter model.
- **Three-style Auto slider** — When FiveK expert models are present, the Auto slider blends `Expert E` dramatic on the left, `Expert C` natural in the middle, and `Expert A` bright/vibrant on the right.
- **Safe NN blend** — Neural predictions are blended with rule-based parameters and clamped to sensible ranges, so the app keeps a stable fallback while gaining better learned edits.
- **Training flow improved** — FiveK training now supports explicit expert selection (`--fivek_expert c/a/e`) and common folder layouts (`input/`, `raw/`, `expertC/`, `c/`).
- **ONNX export improved** — Export now prefers EMA weights from `last.pt` when present.
- **Old model guard** — Direct neural Auto Enhance now requires `enhancer_params.json`; unlabeled old PPR10K-era models are ignored so they cannot make general photos dull or desaturated.

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
