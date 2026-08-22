using LumaPhoto.Vision.Imaging;
using LumaPhoto.Vision.Pipeline;

namespace LumaPhoto.Vision;

/// <summary>
/// The single seam between the UI and inference. LumaPhoto's ViewModel and the
/// batch tool both depend on this and nothing below it — which is what lets the
/// standalone product be a new window over the same assembly instead of a rewrite.
/// </summary>
public interface IBackgroundRemover : IDisposable
{
    /// <summary>Human-readable model name, for the UI.</summary>
    string ModelName { get; }

    /// <summary>"DirectML (GPU)" or "CPU" — worth surfacing, the speed gap is large.</summary>
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
    /// Loads the model ahead of first use. Call on a background task at startup
    /// so the first click doesn't eat the several-hundred-ms session build.
    /// </summary>
    Task WarmupAsync(CancellationToken cancellationToken = default);
}
