namespace LumaPhoto.Vision.Imaging;

/// <summary>
/// Single-channel 8-bit saliency mask. 0 = background, 255 = foreground.
/// Exposed separately from <see cref="ImageBuffer"/> so LumaPhoto can preview
/// and let the user refine the mask before it is baked into alpha.
/// </summary>
public sealed class MaskBuffer
{
    public int Width { get; }
    public int Height { get; }
    public byte[] Values { get; }

    public MaskBuffer(int width, int height, byte[] values)
    {
        if (values.Length != width * height)
            throw new ArgumentException("Mask length must equal Width * Height.", nameof(values));

        Width = width;
        Height = height;
        Values = values;
    }

    public MaskBuffer(int width, int height)
        : this(width, height, new byte[width * height]) { }

    public MaskBuffer Clone() => new(Width, Height, (byte[])Values.Clone());

    /// <summary>Renders the mask as a greyscale BGRA image for on-canvas preview.</summary>
    public ImageBuffer ToPreviewImage()
    {
        var img = new ImageBuffer(Width, Height);
        var px = img.Pixels;

        for (int i = 0, p = 0; i < Values.Length; i++, p += 4)
        {
            var v = Values[i];
            px[p] = v;      // B
            px[p + 1] = v;  // G
            px[p + 2] = v;  // R
            px[p + 3] = 255;
        }

        return img;
    }
}
