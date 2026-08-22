namespace LumaPhoto.Vision.Imaging;

/// <summary>
/// A plain BGRA32 pixel buffer, tightly packed (stride == Width * 4).
/// This is the only image type crossing the library boundary — it keeps the
/// assembly free of WPF, System.Drawing, and ImageSharp so the same code can
/// back the LumaPhoto canvas and a headless batch CLI.
/// </summary>
public sealed class ImageBuffer
{
    public const int BytesPerPixel = 4;

    public int Width { get; }
    public int Height { get; }

    /// <summary>BGRA, non-premultiplied, tightly packed.</summary>
    public byte[] Pixels { get; }

    public int Stride => Width * BytesPerPixel;

    public ImageBuffer(int width, int height, byte[] pixels)
    {
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height));

        var expected = width * height * BytesPerPixel;
        if (pixels.Length != expected)
        {
            throw new ArgumentException(
                $"Expected {expected} bytes for {width}x{height} BGRA32, got {pixels.Length}. " +
                "The source is probably strided — copy row by row before constructing.",
                nameof(pixels));
        }

        Width = width;
        Height = height;
        Pixels = pixels;
    }

    public ImageBuffer(int width, int height)
        : this(width, height, new byte[width * height * BytesPerPixel]) { }

    public ImageBuffer Clone() => new(Width, Height, (byte[])Pixels.Clone());

    /// <summary>
    /// Writes <paramref name="mask"/> into the alpha channel, producing a cut-out.
    /// Mask dimensions must match. Returns a new buffer; the source is untouched.
    /// </summary>
    public ImageBuffer WithAlpha(MaskBuffer mask)
    {
        if (mask.Width != Width || mask.Height != Height)
        {
            throw new ArgumentException(
                $"Mask is {mask.Width}x{mask.Height} but image is {Width}x{Height}. " +
                "Resample the mask first (Postprocessor does this).",
                nameof(mask));
        }

        var result = Clone();
        var px = result.Pixels;
        var m = mask.Values;

        for (int i = 0, p = 3; i < m.Length; i++, p += BytesPerPixel)
        {
            // Multiply rather than overwrite so an already-transparent source
            // (e.g. a previous cut-out being re-run) doesn't regain opacity.
            px[p] = (byte)(px[p] * m[i] / 255);
        }

        return result;
    }
}
