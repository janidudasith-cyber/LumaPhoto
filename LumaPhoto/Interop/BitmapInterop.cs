// ---------------------------------------------------------------------------
// THIS FILE BELONGS IN THE LumaPhoto WPF PROJECT, NOT IN LumaPhoto.Vision.
// It is the only place WPF and the Vision library meet. Keeping it out of the
// library is what preserves the "standalone tool is a weekend, not a rewrite"
// property — move it into Vision and you re-couple everything to WPF.
// ---------------------------------------------------------------------------

using System.IO;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using LumaPhoto.Vision.Imaging;

namespace LumaPhoto.Interop;

public static class BitmapInterop
{
    /// <summary>
    /// Copies a WPF bitmap into an <see cref="ImageBuffer"/>, converting format
    /// and de-striding as needed.
    /// </summary>
    public static ImageBuffer ToImageBuffer(this BitmapSource source)
    {
        // Normalise to straight Bgra32. RenderTargetBitmap hands back Pbgra32
        // (premultiplied); feeding that in unconverted darkens semi-transparent
        // pixels on every round trip.
        BitmapSource src = source.Format == PixelFormats.Bgra32
            ? source
            : new FormatConvertedBitmap(source, PixelFormats.Bgra32, null, 0);

        int width = src.PixelWidth;
        int height = src.PixelHeight;
        int stride = width * 4;
        var pixels = new byte[height * stride];

        src.CopyPixels(pixels, stride, 0);

        return new ImageBuffer(width, height, pixels);
    }

    /// <summary>Wraps an <see cref="ImageBuffer"/> as a frozen, cross-thread-safe bitmap.</summary>
    public static BitmapSource ToBitmapSource(this ImageBuffer buffer, double dpi = 96)
    {
        var bmp = BitmapSource.Create(
            buffer.Width,
            buffer.Height,
            dpi,
            dpi,
            PixelFormats.Bgra32,
            null,
            buffer.Pixels,
            buffer.Stride);

        // Freeze so it can be handed to the UI thread from a background task.
        bmp.Freeze();
        return bmp;
    }

    /// <summary>Loader for BatchRemover — decodes without caching the whole file in memory.</summary>
    public static Task<ImageBuffer> LoadAsync(string path, CancellationToken ct)
        => Task.Run(() =>
        {
            var bmp = new BitmapImage();
            bmp.BeginInit();
            bmp.CacheOption = BitmapCacheOption.OnLoad;
            bmp.UriSource = new Uri(path);
            bmp.EndInit();
            bmp.Freeze();
            return bmp.ToImageBuffer();
        }, ct);

    /// <summary>Saver for BatchRemover. PNG only — alpha must survive.</summary>
    public static Task SaveAsync(ImageBuffer image, string path, CancellationToken ct)
        => Task.Run(() =>
        {
            var encoder = new PngBitmapEncoder();
            encoder.Frames.Add(BitmapFrame.Create(image.ToBitmapSource()));

            using var stream = File.Create(path);
            encoder.Save(stream);
        }, ct);
}
