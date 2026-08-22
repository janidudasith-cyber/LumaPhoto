namespace LumaPhoto.Vision.Models;

/// <summary>
/// Everything that varies between segmentation models: input geometry and the
/// normalisation constants baked into training. Getting these wrong produces a
/// plausible-looking but subtly bad mask, so they live in one place per model
/// rather than scattered through the preprocessor.
/// </summary>
public sealed record ModelDescriptor
{
    public required string Name { get; init; }
    public required string FileName { get; init; }
    public required int InputSize { get; init; }

    /// <summary>Per-channel mean in RGB order, applied after scaling to [0,1].</summary>
    public required float[] Mean { get; init; }

    /// <summary>Per-channel standard deviation in RGB order.</summary>
    public required float[] Std { get; init; }

    /// <summary>
    /// U²-Net's reference preprocessing divides by the image's own maximum
    /// channel value rather than a fixed 255. On normal photos the two agree,
    /// but on dark or low-contrast images they diverge, so match the model.
    /// </summary>
    public bool ScaleByImageMax { get; init; } = true;

    /// <summary>Approximate on-disk size, for installer and download-prompt copy.</summary>
    public required int ApproxSizeMb { get; init; }

    /// <summary>SPDX identifier of the *weights*, which often differ from the code licence.</summary>
    public required string WeightsLicense { get; init; }

    // ---- Presets -------------------------------------------------------

    /// <summary>Small U²-Net. Ships in the installer — good enough for hard-edged subjects.</summary>
    public static readonly ModelDescriptor U2NetP = new()
    {
        Name = "U²-Net Lite",
        FileName = "u2netp.onnx",
        InputSize = 320,
        Mean = [0.485f, 0.456f, 0.406f],
        Std = [0.229f, 0.224f, 0.225f],
        ApproxSizeMb = 5,
        WeightsLicense = "Apache-2.0",
    };

    /// <summary>Full U²-Net. Optional download; markedly better on hair and fur.</summary>
    public static readonly ModelDescriptor U2Net = new()
    {
        Name = "U²-Net Full",
        FileName = "u2net.onnx",
        InputSize = 320,
        Mean = [0.485f, 0.456f, 0.406f],
        Std = [0.229f, 0.224f, 0.225f],
        ApproxSizeMb = 176,
        WeightsLicense = "Apache-2.0",
    };

    /// <summary>
    /// IS-Net / DIS. Best quality of the three, different normalisation (plain
    /// /255, no mean-std). VERIFY THE WEIGHT LICENCE BEFORE SHIPPING COMMERCIALLY —
    /// the code is Apache-2.0 but published checkpoints have been inconsistent.
    /// </summary>
    public static readonly ModelDescriptor IsNet = new()
    {
        Name = "IS-Net",
        FileName = "isnet-general-use.onnx",
        InputSize = 1024,
        Mean = [0.5f, 0.5f, 0.5f],
        Std = [1.0f, 1.0f, 1.0f],
        ScaleByImageMax = false,
        ApproxSizeMb = 173,
        WeightsLicense = "VERIFY BEFORE COMMERCIAL USE",
    };
}
