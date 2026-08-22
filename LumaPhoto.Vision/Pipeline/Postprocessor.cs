using LumaPhoto.Vision.Imaging;

namespace LumaPhoto.Vision.Pipeline;

/// <summary>
/// Converts raw network output into a usable alpha mask.
/// Order matters: normalise → upscale → levels → shrink → feather.
/// Feathering last is what stops the levels step from re-hardening the edge.
/// </summary>
internal static class Postprocessor
{
    public static MaskBuffer Build(
        float[] prediction,
        int predWidth,
        int predHeight,
        int targetWidth,
        int targetHeight,
        RemovalOptions options)
    {
        NormalizeInPlace(prediction);

        // Interpolate in float, then quantise — preserves gradient in fine detail.
        var mask = Resampler.ResizeMask(prediction, predWidth, predHeight, targetWidth, targetHeight);

        var scaled = options.ForImageSize(targetWidth, targetHeight);

        ApplyLevels(mask, scaled.AlphaFloor, scaled.AlphaCeiling, scaled.EdgeContrast);

        if (scaled.Shrink > 0) Erode(mask, scaled.Shrink);
        if (scaled.FeatherRadius > 0) BoxBlur(mask, scaled.FeatherRadius);

        return mask;
    }

    /// <summary>
    /// U²-Net output is unbounded, not a probability. The reference code
    /// min-max normalises per image before use — skipping this gives washed-out
    /// masks on low-contrast photos.
    /// </summary>
    private static void NormalizeInPlace(float[] values)
    {
        float min = float.MaxValue, max = float.MinValue;
        foreach (var v in values)
        {
            if (v < min) min = v;
            if (v > max) max = v;
        }

        float range = max - min;
        if (range < 1e-6f)
        {
            Array.Fill(values, 0f);
            return;
        }

        for (int i = 0; i < values.Length; i++)
            values[i] = (values[i] - min) / range;
    }

    private static void ApplyLevels(MaskBuffer mask, byte floor, byte ceiling, float contrast)
    {
        if (ceiling <= floor) return;

        var lut = new byte[256];
        float span = ceiling - floor;

        for (int i = 0; i < 256; i++)
        {
            float t = Math.Clamp((i - floor) / span, 0f, 1f);

            if (contrast > 0f)
            {
                // Smoothstep blended by contrast strength: an S-curve that leaves
                // 0 and 1 fixed while pushing midtones outward.
                float s = t * t * (3f - 2f * t);
                t = t + (s - t) * contrast;
            }

            lut[i] = (byte)Math.Clamp(t * 255f + 0.5f, 0f, 255f);
        }

        var v = mask.Values;
        for (int i = 0; i < v.Length; i++) v[i] = lut[v[i]];
    }

    /// <summary>Separable minimum filter — pulls the mask boundary inward.</summary>
    private static void Erode(MaskBuffer mask, int radius)
    {
        int w = mask.Width, h = mask.Height;
        var src = mask.Values;
        var tmp = new byte[src.Length];

        for (int y = 0; y < h; y++)
        {
            int row = y * w;
            for (int x = 0; x < w; x++)
            {
                byte min = 255;
                int x0 = Math.Max(0, x - radius), x1 = Math.Min(w - 1, x + radius);
                for (int i = x0; i <= x1; i++)
                    if (src[row + i] < min) min = src[row + i];
                tmp[row + x] = min;
            }
        }

        for (int x = 0; x < w; x++)
        {
            for (int y = 0; y < h; y++)
            {
                byte min = 255;
                int y0 = Math.Max(0, y - radius), y1 = Math.Min(h - 1, y + radius);
                for (int i = y0; i <= y1; i++)
                    if (tmp[i * w + x] < min) min = tmp[i * w + x];
                src[y * w + x] = min;
            }
        }
    }

    /// <summary>
    /// Two separable box-blur passes ≈ a Gaussian, at a fraction of the cost.
    /// Uses a running sum so it stays O(n) regardless of radius.
    /// </summary>
    private static void BoxBlur(MaskBuffer mask, int radius)
    {
        for (int pass = 0; pass < 2; pass++)
        {
            BlurHorizontal(mask, radius);
            BlurVertical(mask, radius);
        }
    }

    private static void BlurHorizontal(MaskBuffer mask, int radius)
    {
        int w = mask.Width, h = mask.Height;
        var src = mask.Values;
        var dst = new byte[src.Length];
        int window = radius * 2 + 1;

        for (int y = 0; y < h; y++)
        {
            int row = y * w;
            int sum = 0;

            for (int i = -radius; i <= radius; i++)
                sum += src[row + Math.Clamp(i, 0, w - 1)];

            for (int x = 0; x < w; x++)
            {
                dst[row + x] = (byte)(sum / window);
                sum -= src[row + Math.Clamp(x - radius, 0, w - 1)];
                sum += src[row + Math.Clamp(x + radius + 1, 0, w - 1)];
            }
        }

        Array.Copy(dst, src, src.Length);
    }

    private static void BlurVertical(MaskBuffer mask, int radius)
    {
        int w = mask.Width, h = mask.Height;
        var src = mask.Values;
        var dst = new byte[src.Length];
        int window = radius * 2 + 1;

        for (int x = 0; x < w; x++)
        {
            int sum = 0;

            for (int i = -radius; i <= radius; i++)
                sum += src[Math.Clamp(i, 0, h - 1) * w + x];

            for (int y = 0; y < h; y++)
            {
                dst[y * w + x] = (byte)(sum / window);
                sum -= src[Math.Clamp(y - radius, 0, h - 1) * w + x];
                sum += src[Math.Clamp(y + radius + 1, 0, h - 1) * w + x];
            }
        }

        Array.Copy(dst, src, src.Length);
    }
}
