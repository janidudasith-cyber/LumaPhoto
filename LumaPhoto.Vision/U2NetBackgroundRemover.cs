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

    public Task WarmupAsync(CancellationToken cancellationToken = default)
        => Task.Run(async () =>
        {
            // Through the same gate as inference: warm-up is what actually builds
            // the session, so racing Dispose here would leave a freshly created
            // session with nothing left to dispose it.
            await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
            try
            {
                if (!_disposed) _sessions.Warmup();
            }
            finally
            {
                _gate.Release();
            }
        }, cancellationToken);

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

        // Wait for any in-flight inference before tearing the session down.
        // Disposing an InferenceSession while Run() is executing frees native
        // memory that call is still reading: an AccessViolationException, which
        // .NET cannot catch, so it kills the process instead of surfacing as a
        // failed removal. The window-close handler and the post-download model
        // swap both dispose from the UI thread while a removal may be running.
        _gate.Wait();
        try { _sessions.Dispose(); }
        finally { _gate.Release(); }

        // _gate is deliberately not disposed. SemaphoreSlim only needs disposal
        // once AvailableWaitHandle has been touched, and disposing it here would
        // turn a late ComputeMaskAsync into an ObjectDisposedException thrown
        // from Wait itself rather than from the guard above.
    }
}
