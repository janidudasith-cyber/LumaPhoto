# Luma Photo Editor — STATUS

_Snapshot taken 2026-08-29._

## What this is

A native Windows desktop photo editor — C# + WPF, .NET 8. Auto-enhance, 15 manual
adjustments, 10 filters, crop/transform, markup, collage, frames, watermark, and
(since v1.4) fully local one-click background removal via ONNX. Ships as a single
`LumaPhoto.exe` plus loose `.onnx` model files, distributed with an Inno Setup
installer and updated in place by `UpdateChecker` off GitHub Releases.

See `README.md` for features, `DEVELOPMENT.md` for architecture, `CHANGELOG.md` for
release history.

## Working tree: one uncommitted change set ⚠️

Branch `main`, up to date with `origin/main`
(github.com/janidudasith-cyber/LumaPhoto). Last commit `b1ec942 v1.4 — local
background removal` (2026-08-25), tagged `v1.4`; released to users 2026-08-22.

Everything below is **uncommitted**. Release build is clean (0 warnings) and the
download path is covered by a passing test harness.

### Feature — on-demand background-removal models

| File | Change |
| --- | --- |
| `LumaPhoto.Vision/Models/ModelDownloader.cs` | **New.** Fetches optional models from the `danielgatis/rembg` release with % progress. Downloads to `.part`, verifies SHA-256, then moves into place — a cancelled, truncated, or corrupted download can never install a model that would load and silently produce garbage masks. Returns a `DownloadResult` enum rather than a bool. |
| `LumaPhoto.Vision/Models/ModelDescriptor.cs` | New `U2NetHumanSeg` descriptor — masks every person in a group photo, unlike the salient-object general models. |
| `LumaPhoto/BackgroundRemoval.cs` | `BgModelPreference` — `EnsureRemover()` picks the best model present (IS-Net → Human → Full → Lite). Upgrade-link handler with cancel-on-second-click; disposes the live ONNX session on success so the new model applies without a restart. |
| `LumaPhoto/MainWindow.xaml` | "Better results for group photos" hyperlink under the Remove Background button, hidden once installed. |
| `LumaPhoto/MainWindow.xaml.cs` | `RefreshBgUpgradeLink()` on `Loaded`. |

### Cleanup and corrections

| File | Change |
| --- | --- |
| `LumaPhoto.Vision/U2NetBackgroundRemover.cs` | Removed dead `FromAppFolder()` — zero call sites. |
| `installer.iss` | Comment explains the ship-lite / download-heavy split. |
| `LumaPhoto.Vision/LumaPhoto.Vision.csproj` | Corrected stale comment. |
| `README.md` | Rewritten. Was from 2026-05-30 and predated background removal, collage, frames, watermark, and the Layers tab; also had wrong export formats and debounce timing. Every claim re-checked against source. |
| `DEVELOPMENT.md` | **File-layout section was backwards** — it called `training/training/` the current modular code and the root files "older versions", when the root files are newer and are what the architecture section actually describes. Also fixed stale installer output name and bundle contents; documented model selection and the `.models-cache` rule. |
| `CHANGELOG.md` | New "Unreleased" section. Fixed **two** wrong release dates against GitHub's `published_at`: v1.4 was 2026-08-19 → **2026-08-22**, v1.3 was 2026-06-10 → **2026-06-14**. |
| `.gitignore` | Added `.models-cache/` and `.claude/settings.local.json`; fixed training paths that pointed at the removed nested folder. |
| `training/training/` | **Deleted** (13 files). Superseded May-2026 snapshot; predated `ImageStatsEncoder` in `model.py`. See note below. |
| `STATUS.md` | This file (new). |

## Model delivery

Decided 2026-08-29: **only the ~4.7 MB lite model ships**; heavier weights are
downloaded in-app on demand. Bundling all three would have taken the installer from
~5 MB to ~360 MB for a feature most users never open.

| Model | Size | Route | Licence |
| --- | --- | --- | --- |
| `u2netp.onnx` U²-Net Lite | 4.6 MB | in installer | Apache-2.0 ✅ |
| `u2net_human_seg.onnx` U²-Net Human | 176 MB | in-app download, SHA-256 verified | Apache-2.0 ✅ |
| `u2net.onnx` U²-Net Full | 176 MB | manual drop-in | Apache-2.0 ✅ |
| `isnet-general-use.onnx` IS-Net | 173 MB | manual drop-in; **not** on this machine | ⚠️ unverified for commercial use |

**`.models-cache/`** at repo root holds the two 176 MB models, gitignored and
**outside the build tree on purpose**. In `Assets\Models` they were copied into four
build outputs — 2.6 GB working folder and 27 s Release builds. Moved out: builds are
~3.5 s and the output carries only what ships. Copy one back in to test auto-select
without re-downloading.

## Verification, 2026-08-29

A console harness (`scratchpad/DlTest`, not in the repo) exercised `ModelDownloader`
against the real release. **All 20 checks passed:**

- Cancel mid-download → `Cancelled`, no `.part` and no model left behind.
- Full 176 MB download in 24 s; exact byte count and SHA-256 match.
- Progress monotonic 0→100, 101 reports, no duplicates.
- Second call short-circuits to `AlreadyPresent`, file untouched.
- A tampered file's hash diverges, so the verify gate rejects it.
- The downloaded model loads and infers correctly on real photographs — **person**
  15.2% of frame kept, **animal** 23.8%, centre alpha far above corner alpha in both.

The animal result matters: it confirms the human-seg model is not humans-only, so
preferring it over the general model in `BgModelPreference` does not regress
non-human subjects. (A first attempt probed with a flat synthetic rectangle and got
an empty mask — U²-Net variants are trained on photographs and return nothing for
featureless input. The test was wrong, not the model.)

## Disk

Working folder went **2.2 GB → ~800 MB**. Deleted build artifacts (`bin/`, `obj/`,
`publish/`, `__pycache__`, empty `checkpoints_*`), then moved the heavy models out of
the build tree, which was the actual cause of the bloat.

Deliberately kept:

- **`installer_output/` (407 MB)** — v1.1–v1.4 installers, verified byte-for-byte
  against GitHub Releases, so safe to delete any time you want the space back.
- **`LumaPhoto Shortcuts.docx`** — untracked personal file.

## What needs attention

1. **Commit.** Nothing here is committed yet. Nothing has been pushed.

2. **The UI download link has not been clicked by a human.** The downloader itself is
   thoroughly tested, but the WPF wiring — link visibility, progress text, cancel on
   second click, toast — has only been verified by compilation. Run the app with
   `Assets\Models` holding only `u2netp.onnx` (its current state) and click it.

3. **`training/training/` deletion loses two functions from working tree** —
   `_gdrive_url` and `_gdrive_confirm_url` in `download_data.py` existed only in the
   nested copy, dropped from the root version when datasets moved off Google Drive.
   They remain in git history; noted in `DEVELOPMENT.md`.

4. **IS-Net weight licence is still unverified** — it sits at the top of
   `BgModelPreference`, so if anyone drops that file in, it wins. Fine for personal
   use; verify before any commercial distribution.

5. **No automated tests in the repo.** The harness that verified the downloader lives
   in a scratchpad and will vanish. Worth promoting to a real test project if this
   area keeps growing.
