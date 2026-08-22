using LumaPhoto.Vision.Imaging;
using LumaPhoto.Vision.Pipeline;

namespace LumaPhoto.Vision.Batch;

public sealed record BatchItem(string SourcePath, string OutputPath);

public sealed record BatchProgress(
    int Completed,
    int Total,
    string CurrentFile,
    int Failed)
{
    public float Fraction => Total == 0 ? 0f : (float)Completed / Total;
}

public sealed record BatchResult(
    int Succeeded,
    int Failed,
    IReadOnlyList<(string Path, string Error)> Errors,
    TimeSpan Elapsed);

/// <summary>
/// Batch cut-out. This is the paid feature — single-image removal stays free as
/// the hook, because nobody pays $30 to cut out one photo, but somebody with 200
/// product shots pays immediately.
///
/// Decoding and encoding are delegated so this assembly stays WPF-free; the host
/// app supplies the codec (WPF's BitmapDecoder/PngBitmapEncoder in LumaPhoto).
/// </summary>
public sealed class BatchRemover
{
    private readonly IBackgroundRemover _remover;
    private readonly Func<string, CancellationToken, Task<ImageBuffer>> _load;
    private readonly Func<ImageBuffer, string, CancellationToken, Task> _save;

    public BatchRemover(
        IBackgroundRemover remover,
        Func<string, CancellationToken, Task<ImageBuffer>> loader,
        Func<ImageBuffer, string, CancellationToken, Task> saver)
    {
        _remover = remover;
        _load = loader;
        _save = saver;
    }

    public async Task<BatchResult> RunAsync(
        IReadOnlyList<BatchItem> items,
        RemovalOptions? options = null,
        IProgress<BatchProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        var started = DateTime.UtcNow;
        var errors = new List<(string, string)>();
        int completed = 0, failed = 0;

        await _remover.WarmupAsync(cancellationToken).ConfigureAwait(false);

        // Sequential by design. Inference already saturates the thread pool or
        // GPU, so parallelising here mainly multiplies peak memory — a 6000px
        // image is ~140 MB decoded, and eight at once will OOM a 16 GB machine.
        foreach (var item in items)
        {
            cancellationToken.ThrowIfCancellationRequested();

            progress?.Report(new BatchProgress(
                completed, items.Count, Path.GetFileName(item.SourcePath), failed));

            try
            {
                var source = await _load(item.SourcePath, cancellationToken).ConfigureAwait(false);
                var result = await _remover.RemoveAsync(source, options, cancellationToken)
                                           .ConfigureAwait(false);

                Directory.CreateDirectory(Path.GetDirectoryName(item.OutputPath)!);
                await _save(result, item.OutputPath, cancellationToken).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch (Exception ex)
            {
                // One bad file must not kill a 200-image run.
                failed++;
                errors.Add((item.SourcePath, ex.Message));
            }

            completed++;
        }

        progress?.Report(new BatchProgress(completed, items.Count, string.Empty, failed));

        return new BatchResult(
            completed - failed, failed, errors, DateTime.UtcNow - started);
    }

    /// <summary>Maps a folder of images to PNG outputs, preserving base names.</summary>
    public static IReadOnlyList<BatchItem> PlanFolder(
        string inputFolder, string outputFolder, bool recursive = false)
    {
        string[] extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"];

        var search = recursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;

        return Directory.EnumerateFiles(inputFolder, "*.*", search)
            .Where(f => extensions.Contains(Path.GetExtension(f).ToLowerInvariant()))
            .Select(f => new BatchItem(
                f,
                // Always PNG out — JPEG has no alpha channel, and silently
                // flattening a cut-out onto white is the worst possible default.
                Path.Combine(outputFolder, Path.GetFileNameWithoutExtension(f) + ".png")))
            .ToList();
    }
}
