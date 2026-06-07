using System.IO;
using System.Net.Http;
using System.Text.Json.Nodes;

namespace LumaPhoto;

/// <summary>Metadata returned when a newer GitHub Release is found.</summary>
public record UpdateInfo(string Version, string DownloadUrl);

/// <summary>
/// Checks GitHub Releases for a newer version and optionally downloads the installer.
/// All methods are silent on failure — no internet / GitHub down simply returns null.
/// </summary>
public static class UpdateChecker
{
    private static readonly HttpClient _http = new() { Timeout = TimeSpan.FromSeconds(10) };

    static UpdateChecker()
    {
        // GitHub API requires a User-Agent header.
        _http.DefaultRequestHeaders.UserAgent.ParseAdd($"LumaPhoto/{AppVersion.Current}");
    }

    // ── Check ─────────────────────────────────────────────────────────────────

    /// <summary>
    /// Queries the GitHub Releases API.
    /// Returns <see cref="UpdateInfo"/> if a newer version exists, otherwise null.
    /// </summary>
    public static async Task<UpdateInfo?> CheckAsync()
    {
        try
        {
            string apiUrl = $"https://api.github.com/repos/{AppVersion.GitHubRepo}/releases/latest";
            string json   = await _http.GetStringAsync(apiUrl).ConfigureAwait(false);
            var    doc    = JsonNode.Parse(json);

            string? tag = doc?["tag_name"]?.GetValue<string>();
            if (tag == null) return null;

            string latestVer = tag.TrimStart('v');
            if (!IsNewer(latestVer, AppVersion.Current)) return null;

            // Prefer the .exe asset; fall back to the HTML release page.
            string? downloadUrl = FindExeAsset(doc)
                               ?? doc?["html_url"]?.GetValue<string>()
                               ?? $"https://github.com/{AppVersion.GitHubRepo}/releases/latest";

            return new UpdateInfo(latestVer, downloadUrl);
        }
        catch
        {
            return null;   // No internet, rate-limited, bad JSON, etc.
        }
    }

    // ── Download ──────────────────────────────────────────────────────────────

    /// <summary>
    /// Downloads the installer to a temp file, reporting integer % progress (0–100).
    /// Returns the local file path on success, null on failure.
    /// </summary>
    public static async Task<string?> DownloadInstallerAsync(
        string url, IProgress<int>? progress = null, CancellationToken ct = default)
    {
        try
        {
            string dest = Path.Combine(Path.GetTempPath(), "LumaPhoto-Update.exe");

            using var response = await _http
                .GetAsync(url, HttpCompletionOption.ResponseHeadersRead, ct)
                .ConfigureAwait(false);
            response.EnsureSuccessStatusCode();

            long? total = response.Content.Headers.ContentLength;
            await using var netStream  = await response.Content.ReadAsStreamAsync(ct).ConfigureAwait(false);
            await using var fileStream = File.Create(dest);

            byte[] buf        = new byte[81_920];
            long   downloaded = 0;
            int    read;

            while ((read = await netStream.ReadAsync(buf, ct).ConfigureAwait(false)) > 0)
            {
                await fileStream.WriteAsync(buf.AsMemory(0, read), ct).ConfigureAwait(false);
                downloaded += read;
                if (total > 0)
                    progress?.Report((int)(downloaded * 100 / total.Value));
            }

            return dest;
        }
        catch
        {
            return null;
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static string? FindExeAsset(JsonNode? doc)
    {
        var assets = doc?["assets"]?.AsArray();
        if (assets == null) return null;

        foreach (var asset in assets)
        {
            string? name = asset?["name"]?.GetValue<string>();
            if (name != null && name.EndsWith(".exe", StringComparison.OrdinalIgnoreCase))
                return asset?["browser_download_url"]?.GetValue<string>();
        }
        return null;
    }

    /// <summary>Returns true when <paramref name="latest"/> is strictly newer than <paramref name="current"/>.</summary>
    private static bool IsNewer(string latest, string current)
    {
        try
        {
            return Parse(latest) > Parse(current);

            static Version Parse(string v)
            {
                // Normalise "1.1" → "1.1.0" so Version.Parse doesn't throw.
                int dots = v.Count(c => c == '.');
                return Version.Parse(dots == 0 ? v + ".0.0" : dots == 1 ? v + ".0" : v);
            }
        }
        catch { return false; }
    }
}
