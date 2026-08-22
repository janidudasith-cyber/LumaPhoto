using LumaPhoto.Vision.Imaging;
using LumaPhoto.Vision.Models;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LumaPhoto.Vision.Pipeline;

/// <summary>
/// Turns a BGRA photo into the NCHW float tensor the network expects.
/// Mirrors U²-Net's RescaleT + ToTensorLab reference transforms.
/// </summary>
internal static class Preprocessor
{
    public static DenseTensor<float> ToTensor(ImageBuffer source, ModelDescriptor model)
    {
        int size = model.InputSize;
        var resized = Resampler.ResizeBgra(source, size, size);
        var px = resized.Pixels;

        // Reference implementation divides by the image's own max channel value.
        // Fall back to 255 on a fully black image to avoid dividing by zero.
        float scale = 255f;
        if (model.ScaleByImageMax)
        {
            byte max = 0;
            for (int i = 0; i < px.Length; i += 4)
            {
                if (px[i] > max) max = px[i];
                if (px[i + 1] > max) max = px[i + 1];
                if (px[i + 2] > max) max = px[i + 2];
            }
            scale = max > 0 ? max : 255f;
        }

        var tensor = new DenseTensor<float>([1, 3, size, size]);
        var buffer = tensor.Buffer.Span;

        int plane = size * size;
        int rOff = 0, gOff = plane, bOff = plane * 2;

        for (int i = 0, p = 0; i < plane; i++, p += 4)
        {
            // Source is BGRA; the model wants RGB.
            float r = px[p + 2] / scale;
            float g = px[p + 1] / scale;
            float b = px[p + 0] / scale;

            buffer[rOff + i] = (r - model.Mean[0]) / model.Std[0];
            buffer[gOff + i] = (g - model.Mean[1]) / model.Std[1];
            buffer[bOff + i] = (b - model.Mean[2]) / model.Std[2];
        }

        return tensor;
    }
}
