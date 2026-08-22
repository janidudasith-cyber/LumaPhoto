using Microsoft.ML.OnnxRuntime;

namespace LumaPhoto.Vision.Models;

/// <summary>
/// Owns the <see cref="InferenceSession"/>. Sessions are expensive to build
/// (hundreds of ms) and cheap to reuse, so create one per model and keep it
/// alive for the life of the window — never per-image, and never per batch item.
/// </summary>
public sealed class OnnxSessionManager : IDisposable
{
    private readonly Lazy<InferenceSession> _session;
    private bool _disposed;

    public ModelDescriptor Descriptor { get; }
    public string ModelPath { get; }

    /// <summary>Which execution provider actually loaded, for the status bar / bug reports.</summary>
    public string ActiveProvider { get; private set; } = "not initialised";

    public OnnxSessionManager(ModelDescriptor descriptor, string modelPath, bool preferGpu = true)
    {
        Descriptor = descriptor;
        ModelPath = modelPath;

        if (!File.Exists(modelPath))
        {
            throw new FileNotFoundException(
                $"Model '{descriptor.Name}' not found. Expected {descriptor.FileName} " +
                $"(~{descriptor.ApproxSizeMb} MB) at this path.", modelPath);
        }

        // Lazy so constructing the remover on the UI thread stays instant;
        // the first inference call pays the load cost.
        _session = new Lazy<InferenceSession>(() => Create(modelPath, preferGpu));
    }

    public InferenceSession Session => _session.Value;

    /// <summary>Name of the single input tensor, read from metadata rather than hardcoded.</summary>
    public string InputName => Session.InputMetadata.Keys.First();

    /// <summary>
    /// U²-Net emits seven side outputs (d0..d6). d0 is the fused prediction and
    /// the only one worth using; the rest are deep-supervision artefacts.
    /// </summary>
    public string PrimaryOutputName => Session.OutputMetadata.Keys.First();

    private InferenceSession Create(string path, bool preferGpu)
    {
        var options = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            // Leave headroom so a batch run doesn't freeze the UI thread.
            IntraOpNumThreads = Math.Max(1, Environment.ProcessorCount / 2),
        };

        if (preferGpu)
        {
            try
            {
                // Only resolves if the DirectML package is referenced. Wrapped
                // because a missing provider throws at call time, not compile time.
                options.AppendExecutionProvider_DML(0);
                ActiveProvider = "DirectML (GPU)";
                return new InferenceSession(path, options);
            }
            catch (Exception)
            {
                // Fall through to CPU. Common on VMs and older integrated GPUs.
            }
        }

        ActiveProvider = "CPU";
        return new InferenceSession(path, options);
    }

    /// <summary>Forces the model to load now — call from a background task at startup.</summary>
    public void Warmup() => _ = Session;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_session.IsValueCreated) _session.Value.Dispose();
    }
}
