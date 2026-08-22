using LumaPhoto.Vision.Imaging;
using LumaPhoto.Vision.Pipeline;

namespace LumaPhoto.Vision;

/// <summary>
/// Defines the UI-independent background-removal contract used by interactive
/// and batch hosts.
/// </summary>
public interface IBackgroundRemover : IDisposable
{
    /// <summary>Human-readable model name, for the UI.</summary>
    string ModelName { get; }

    /// <summary>Execution provider used by the active inference session.</summary>
    string ActiveProvider { get; }

    /// <summary>
    /// Produces the alpha mask without applying it. Use this for the interactive
    /// path so the user can preview and tweak <see cref="RemovalOptions"/> without
    /// re-running inference — mask computation is the expensive half.
    /// </summary>
    Task<MaskBuffer> ComputeMaskAsync(
        ImageBuffer source,
        RemovalOptions? options = null,
        CancellationToken cancellationToken = default);

    /// <summary>Runs inference and returns the cut-out with alpha applied.</summary>
    Task<ImageBuffer> RemoveAsync(
        ImageBuffer source,
        RemovalOptions? options = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Loads the model ahead of first use to avoid session-initialization latency
    /// during the first removal request.
    /// </summary>
    Task WarmupAsync(CancellationToken cancellationToken = default);
}
