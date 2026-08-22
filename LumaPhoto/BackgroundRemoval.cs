using System.Windows;
using System.Windows.Controls.Primitives;
using System.Windows.Input;
using LumaPhoto.Interop;
using LumaPhoto.Vision;
using LumaPhoto.Vision.Imaging;
using LumaPhoto.Vision.Pipeline;

namespace LumaPhoto;

// Background removal UI. All inference lives in the LumaPhoto.Vision class
// library — this file only marshals pixels across the BitmapInterop seam and
// drives the panel, so the library stays UI-framework agnostic.
public partial class MainWindow
{
    private IBackgroundRemover? _bgRemover;
    private bool _bgRemoverUnavailable;          // model missing — don't retry every click
    private CancellationTokenSource? _bgCts;

    // Pixels as they were before the current removal, so the edge presets can
    // re-run against the original instead of compounding cut-out on cut-out.
    private byte[]? _bgBeforePixels;
    private int _bgBeforeW, _bgBeforeH;

    /// <summary>
    /// Creates the remover on first use and warms the ONNX session in the
    /// background. Returns null when the model file is absent.
    /// </summary>
    private IBackgroundRemover? EnsureRemover()
    {
        if (_bgRemover != null || _bgRemoverUnavailable) return _bgRemover;
        try
        {
            _bgRemover = U2NetBackgroundRemover.FromAppFolder();
            _ = _bgRemover.WarmupAsync();
        }
        catch (Exception)
        {
            // Missing/unreadable model — surfaced to the user by the caller.
            _bgRemoverUnavailable = true;
        }
        return _bgRemover;
    }

    private void RemoveBg_Click(object sender, RoutedEventArgs e)
        => _ = RunBackgroundRemovalAsync(RemovalOptions.Default, isRerun: false);

    private void RemoveBgEdge_Click(object sender, RoutedEventArgs e)
    {
        if (sender is not ToggleButton clicked) return;

        // Radio behaviour — the presets are mutually exclusive.
        foreach (var t in new[] { BgEdgeBalanced, BgEdgeSoft, BgEdgeHard })
            t.IsChecked = t == clicked;

        var options = (string)clicked.Tag switch
        {
            "Soft" => RemovalOptions.SoftSubject,
            "Hard" => RemovalOptions.HardSubject,
            _      => RemovalOptions.Default,
        };
        _ = RunBackgroundRemovalAsync(options, isRerun: true);
    }

    private async Task RunBackgroundRemovalAsync(RemovalOptions options, bool isRerun)
    {
        if (!_imageLoaded || _sourcePixels == null) return;

        var remover = EnsureRemover();
        if (remover == null)
        {
            RemoveBgStatus.Text = "Model not installed.";
            ShowToast("Background Removal Unavailable",
                "u2netp.onnx was not found next to the app. Reinstall LumaPhoto, or place the model " +
                "in an Assets\\Models folder beside LumaPhoto.exe.", success: false);
            return;
        }

        // A re-run replays the preset against the pre-removal pixels; a fresh run
        // snapshots them first. Either way only one history entry is created.
        byte[] basePixels;
        int baseW, baseH;
        if (isRerun && _bgBeforePixels != null)
        {
            basePixels = _bgBeforePixels;
            baseW = _bgBeforeW; baseH = _bgBeforeH;
        }
        else
        {
            PushHistory();
            _bgBeforePixels = (byte[])_sourcePixels.Clone();
            _bgBeforeW = _sourceW; _bgBeforeH = _sourceH;
            basePixels = _bgBeforePixels;
            baseW = _sourceW; baseH = _sourceH;
        }

        _bgCts?.Cancel();
        _bgCts?.Dispose();
        _bgCts = new CancellationTokenSource();
        var token = _bgCts.Token;

        RemoveBgBtn.IsEnabled = false;
        RemoveBgStatus.Text   = "Finding the subject…";
        Mouse.OverrideCursor  = Cursors.Wait;

        try
        {
            // ImageBuffer is BGRA32 tightly packed — the same layout as
            // _sourcePixels — so this crosses the boundary without a bitmap copy.
            var source = new ImageBuffer(baseW, baseH, (byte[])basePixels.Clone());
            var cut    = await remover.RemoveAsync(source, options, token);

            if (token.IsCancellationRequested) return;

            _sourcePixels = cut.Pixels;
            _sourceW = cut.Width; _sourceH = cut.Height;

            RemoveBgRefinePanel.Visibility = Visibility.Visible;
            RemoveBgStatus.Text = $"Done · {remover.ModelName} · {remover.ActiveProvider}";
            DoRender();
            UpdateLayerList();

            if (!isRerun)
                ShowToast("Background Removed",
                    "Export as PNG to keep the transparency — JPEG has no alpha channel.");
        }
        catch (OperationCanceledException)
        {
            // Superseded by a newer run; leave the canvas to that one.
        }
        catch (Exception ex)
        {
            RemoveBgStatus.Text = "Removal failed.";
            ShowToast("Background Removal Failed", ex.Message, success: false);
        }
        finally
        {
            Mouse.OverrideCursor = null;
            RemoveBgBtn.IsEnabled = _imageLoaded;
        }
    }

    /// <summary>Clears per-image removal state. Called when a new photo loads.</summary>
    private void ResetBackgroundRemovalUi()
    {
        _bgBeforePixels = null;
        RemoveBgRefinePanel.Visibility = Visibility.Collapsed;
        RemoveBgStatus.Text = "";
        BgEdgeBalanced.IsChecked = true;
        BgEdgeSoft.IsChecked = false;
        BgEdgeHard.IsChecked = false;
    }
}
