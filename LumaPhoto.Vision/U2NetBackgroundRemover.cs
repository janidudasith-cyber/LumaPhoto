using LumaPhoto.Vision.Imaging;
using LumaPhoto.Vision.Models;
using LumaPhoto.Vision.Pipeline;
using Microsoft.ML.OnnxRuntime;

namespace LumaPhoto.Vision;

/// <summary>
/// U²-Net / IS-Net background remover.
///
/// Thread-safety: inference is serialised through a semaphore. ONNX Runtime
/// sessions are technically thread-safe, but concurrent Run() calls on one
/// session contend for the same intra-op thread pool and end up slower than
/// sequential. The batch processor parallelises over *files*, not over Run().
/// </summary>
public sealed class U2NetBackgroundRemover : IBackgroundRemover
{
    private readonly OnnxSessionManager _sessions;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private bool _disposed;

    public string ModelName => _sessions.Descriptor.Name;
    public string ActiveProvider => _sessions.ActiveProvider;

    public U2NetBackgroundRemover(ModelDescriptor descriptor, string modelPath, bool preferGpu = true)
        => _sessions = new OnnxSessionManager(descriptor, modelPath, preferGpu);

    /// <summary>Convenience factory resolving the model from the app's Assets\Models folder.</summary>
    public static U2NetBackgroundRemover FromAppFolder(
        ModelDescriptor? descriptor = null, bool preferGpu = true)
    {
        descriptor ??= ModelDescriptor.U2NetP;
        var path = Path.Combine(
            AppContext.BaseDirectory, "Assets", "Models", descriptor.FileName);
        return new U2NetBackgroundRemover(descriptor, path, preferGpu);
    }

    public Task WarmupAsync(CancellationToken cancellationToken = default)
        => Task.Run(() => _sessions.Warmup(), cancellationToken);

    public async Task<MaskBuffer> ComputeMaskAsync(
        ImageBuffer source,
        RemovalOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        options ??= RemovalOptions.Default;

        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            return await Task.Run(() => Infer(source, options, cancellationToken), cancellationToken)
                             .ConfigureAwait(false);
        }
        finally
        {
            _gate.Release();
        }
    }

    public async Task<ImageBuffer> RemoveAsync(
        ImageBuffer source,
        RemovalOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var mask = await ComputeMaskAsync(source, options, cancellationToken).ConfigureAwait(false);
        return source.WithAlpha(mask);
    }

    private MaskBuffer Infer(ImageBuffer source, RemovalOptions options, CancellationToken ct)
    {
        var model = _sessions.Descriptor;

        ct.ThrowIfCancellationRequested();
        var input = Preprocessor.ToTensor(source, model);

        ct.ThrowIfCancellationRequested();
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_sessions.InputName, input),
        };

        using var results = _sessions.Session.Run(inputs);

        // d0 — the fused prediction. Later outputs are deep-supervision stages.
        var output = results.First().AsTensor<float>();
        var dims = output.Dimensions;

        // Expect [1,1,H,W]; read the trailing two dims rather than assuming
        // InputSize, since IS-Net variants can return a different resolution.
        int predHeight = dims[^2];
        int predWidth = dims[^1];

        var prediction = output.ToArray();

        ct.ThrowIfCancellationRequested();
        return Postprocessor.Build(
            prediction, predWidth, predHeight, source.Width, source.Height, options);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _gate.Dispose();
        _sessions.Dispose();
    }
}
