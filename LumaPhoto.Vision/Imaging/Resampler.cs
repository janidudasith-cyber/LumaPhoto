namespace LumaPhoto.Vision.Imaging;

/// <summary>
/// Hand-rolled bilinear resampling. Written out rather than pulled from
/// System.Drawing because that package is Windows-only and deprecated for
/// cross-platform use — and this is ~60 lines we fully control.
/// </summary>
internal static class Resampler
{
    public static ImageBuffer ResizeBgra(ImageBuffer src, int dstWidth, int dstHeight)
    {
        if (src.Width == dstWidth && src.Height == dstHeight) return src.Clone();

        var dst = new ImageBuffer(dstWidth, dstHeight);
        var s = src.Pixels;
        var d = dst.Pixels;

        float xRatio = (float)src.Width / dstWidth;
        float yRatio = (float)src.Height / dstHeight;

        for (int y = 0; y < dstHeight; y++)
        {
            // Clamp in source space *before* splitting into index + fraction,
            // otherwise the edge rows get the wrong interpolation weight.
            float sy = Math.Clamp((y + 0.5f) * yRatio - 0.5f, 0f, src.Height - 1f);
            int y0 = (int)MathF.Floor(sy);
            int y1 = Math.Min(y0 + 1, src.Height - 1);
            float fy = sy - y0;

            for (int x = 0; x < dstWidth; x++)
            {
                float sx = Math.Clamp((x + 0.5f) * xRatio - 0.5f, 0f, src.Width - 1f);
                int x0 = (int)MathF.Floor(sx);
                int x1 = Math.Min(x0 + 1, src.Width - 1);
                float fx = sx - x0;

                int i00 = (y0 * src.Width + x0) * 4;
                int i01 = (y0 * src.Width + x1) * 4;
                int i10 = (y1 * src.Width + x0) * 4;
                int i11 = (y1 * src.Width + x1) * 4;
                int di = (y * dstWidth + x) * 4;

                for (int c = 0; c < 4; c++)
                {
                    float top = s[i00 + c] + (s[i01 + c] - s[i00 + c]) * fx;
                    float bot = s[i10 + c] + (s[i11 + c] - s[i10 + c]) * fx;
                    d[di + c] = (byte)Math.Clamp(top + (bot - top) * fy + 0.5f, 0f, 255f);
                }
            }
        }

        return dst;
    }

    /// <summary>
    /// Upsamples the raw float network output straight to target resolution.
    /// Interpolating in float before quantising to bytes keeps noticeably more
    /// gradient in hair and fur than resizing a byte mask would.
    /// </summary>
    public static MaskBuffer ResizeMask(
        float[] src, int srcWidth, int srcHeight, int dstWidth, int dstHeight)
    {
        var dst = new MaskBuffer(dstWidth, dstHeight);
        var d = dst.Values;

        float xRatio = (float)srcWidth / dstWidth;
        float yRatio = (float)srcHeight / dstHeight;

        for (int y = 0; y < dstHeight; y++)
        {
            float sy = Math.Clamp((y + 0.5f) * yRatio - 0.5f, 0f, srcHeight - 1f);
            int y0 = (int)MathF.Floor(sy);
            int y1 = Math.Min(y0 + 1, srcHeight - 1);
            float fy = sy - y0;

            for (int x = 0; x < dstWidth; x++)
            {
                float sx = Math.Clamp((x + 0.5f) * xRatio - 0.5f, 0f, srcWidth - 1f);
                int x0 = (int)MathF.Floor(sx);
                int x1 = Math.Min(x0 + 1, srcWidth - 1);
                float fx = sx - x0;

                float top = src[y0 * srcWidth + x0] + (src[y0 * srcWidth + x1] - src[y0 * srcWidth + x0]) * fx;
                float bot = src[y1 * srcWidth + x0] + (src[y1 * srcWidth + x1] - src[y1 * srcWidth + x0]) * fx;
                float v = top + (bot - top) * fy;

                d[y * dstWidth + x] = (byte)Math.Clamp(v * 255f + 0.5f, 0f, 255f);
            }
        }

        return dst;
    }
}
