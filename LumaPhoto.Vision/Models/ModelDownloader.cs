using System.Net.Http;
using System.Security.Cryptography;

namespace LumaPhoto.Vision.Models;

/// <summary>Outcome of a model download. <see cref="Failed"/> carries no detail by design —
/// the UI only ever offers "retry", and the distinction between offline, 404, and a bad
/// hash is not actionable for the user.</summary>
public enum DownloadResult
{
    Succeeded,
    AlreadyPresent,
    Cancelled,
    Failed,
}

/// <summary>
/// Fetches the optional background-removal models on demand into the app's
/// <c>Assets\Models</c> folder. The installer only ships the ~4.7 MB lite model;
/// the ~176 MB group-photo and hair models are downloaded the first time a user
/// asks for better quality. Once a file lands, <c>EnsureRemover()</c> auto-selects it.
///
/// Weights are the U²-Net ONNX exports published by the <c>danielgatis/rembg</c>
/// project (Apache-2.0).
///
/// Lives here rather than in the WPF project so the headless batch tool can prefetch
/// models too — this assembly stays UI-framework agnostic.
/// </summary>
public static class ModelDownloader
{
    // Long timeout: these are ~176 MB files and the whole response is streamed
    // through one HttpClient call.
    private static readonly HttpClient _http = new() { Timeout = TimeSpan.FromMinutes(30) };

    static ModelDownloader()
        => _http.DefaultRequestHeaders.UserAgent.ParseAdd("LumaPhoto");

    /// <summary>rembg publishes every model as an asset on this single release tag.</summary>
    private const string RembgRelease =
        "https://github.com/danielgatis/rembg/releases/download/v0.0.0/";

    /// <summary>
    /// SHA-256 of each downloadable model, verified 2026-08-29 against the rembg
    /// release assets. A model that downloads cleanly but hashes wrong is discarded:
    /// a corrupt or substituted ONNX loads happily and silently produces garbage
    /// masks, which is far harder to diagnose than a failed download.
    /// </summary>
    private static readonly IReadOnlyDictionary<string, string> Sha256 =
        new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            ["u2net_human_seg.onnx"] = "01eb6a29a5c4d8edb30b56adad9bb3a2a0535338e480724a213e0acfd2d1c73c",
            ["u2net.onnx"]           = "8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491",
        };

    /// <summary>Models that are safe to redistribute and worth offering as a download.</summary>
    public static readonly IReadOnlyList<ModelDescriptor> Optional =
    [
        ModelDescriptor.U2NetHumanSeg,
        ModelDescriptor.U2Net,
    ];

    /// <summary>The folder models are resolved from — next to the running exe.</summary>
    public static string ModelsDirectory =>
        Path.Combine(AppContext.BaseDirectory, "Assets", "Models");

    /// <summary>Absolute path where <paramref name="model"/> is expected on disk.</summary>
    public static string PathFor(ModelDescriptor model)
        => Path.Combine(ModelsDirectory, model.FileName);

    public static bool IsInstalled(ModelDescriptor model) => File.Exists(PathFor(model));

    /// <summary>True when a checksum is known, i.e. the model can be offered as a download.</summary>
    public static bool IsDownloadable(ModelDescriptor model) => Sha256.ContainsKey(model.FileName);

    /// <summary>
    /// Downloads <paramref name="model"/> to its <see cref="PathFor"/> location,
    /// reporting integer % progress (0–100).
    ///
    /// Downloads to a <c>.part</c> file, verifies SHA-256, and only then moves it
    /// into place — so a cancelled, truncated, or corrupted download never leaves
    /// a model that would load and produce silently-wrong masks.
    /// </summary>
    public static async Task<DownloadResult> DownloadAsync(
        ModelDescriptor model, IProgress<int>? progress = null, CancellationToken ct = default)
    {
        if (IsInstalled(model)) return DownloadResult.AlreadyPresent;

        string dest = PathFor(model);
        string part = dest + ".part";

        try
        {
            Directory.CreateDirectory(ModelsDirectory);

            using (var response = await _http
                .GetAsync(RembgRelease + model.FileName, HttpCompletionOption.ResponseHeadersRead, ct)
                .ConfigureAwait(false))
            {
                response.EnsureSuccessStatusCode();
                long? total = response.Content.Headers.ContentLength;

                await using var netStream = await response.Content
                    .ReadAsStreamAsync(ct).ConfigureAwait(false);
                await using var fileStream = new FileStream(
                    part, FileMode.Create, FileAccess.Write, FileShare.None,
                    bufferSize: 81_920, useAsync: true);

                byte[] buf        = new byte[81_920];
                long   downloaded = 0;
                int    lastPct    = -1;
                int    read;

                while ((read = await netStream.ReadAsync(buf, ct).ConfigureAwait(false)) > 0)
                {
                    await fileStream.WriteAsync(buf.AsMemory(0, read), ct).ConfigureAwait(false);
                    downloaded += read;

                    // Only report on change — a 176 MB file is ~2 200 chunks, and
                    // every report marshals to the UI thread.
                    if (total > 0)
                    {
                        int pct = (int)(downloaded * 100 / total.Value);
                        if (pct != lastPct) { progress?.Report(pct); lastPct = pct; }
                    }
                }
            }

            if (!await VerifyAsync(part, model.FileName, ct).ConfigureAwait(false))
            {
                TryDelete(part);
                return DownloadResult.Failed;
            }

            File.Move(part, dest, overwrite: true);
            return DownloadResult.Succeeded;
        }
        catch (OperationCanceledException)
        {
            TryDelete(part);
            return DownloadResult.Cancelled;
        }
        catch (Exception)
        {
            // Offline, DNS failure, 404, disk full — all equally "retry later".
            TryDelete(part);
            return DownloadResult.Failed;
        }
    }

    /// <summary>
    /// Compares the file's SHA-256 against the known-good hash. An unknown filename
    /// passes: the checksum table covers what we offer for download, and a model the
    /// user placed by hand is their business.
    /// </summary>
    private static async Task<bool> VerifyAsync(string path, string fileName, CancellationToken ct)
    {
        if (!Sha256.TryGetValue(fileName, out var expected)) return true;

        await using var stream = new FileStream(
            path, FileMode.Open, FileAccess.Read, FileShare.Read,
            bufferSize: 81_920, useAsync: true);

        byte[] hash = await SHA256.HashDataAsync(stream, ct).ConfigureAwait(false);
        return Convert.ToHexString(hash).Equals(expected, StringComparison.OrdinalIgnoreCase);
    }

    private static void TryDelete(string path)
    {
        try { if (File.Exists(path)) File.Delete(path); }
        catch { /* best effort — a stray .part is harmless, it is never loaded */ }
    }
}
