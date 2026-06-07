using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Controls.Primitives;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using Microsoft.Win32;

namespace LumaPhoto;

public partial class MainWindow : Window
{
    // ── State ──
    private byte[]? _sourcePixels;
    private int     _sourceW, _sourceH;
    private string  _fileName = "photo";

    private readonly AdjustmentState _adj = new();
    private bool _imageLoaded = false;
    private bool _compareMode = false;

    // Render debounce + async
    private System.Windows.Threading.DispatcherTimer? _renderTimer;
    private CancellationTokenSource? _renderCts;

    // ── Auto-enhance state ──
    // _autoBaseParams  = the computed "full strength" target (slider centre = 0)
    // _autoPreState    = the adjustment state BEFORE auto was toggled on
    // _autoEnhanceOn   = whether auto is currently active
    private AdjustmentState? _autoBaseParams;
    private AdjustmentState? _autoPreState;
    private SceneType         _autoScene = SceneType.General;
    private bool              _autoEnhanceOn = false;

    // ── Neural enhancer (Places365-MobileNet) ──
    private NeuralEnhancer?  _neuralEnhancer;
    private ImageAnalysis?   _autoAnalysis;   // stored for NN refinement after async run
    private SceneWeights?    _autoNNWeights;  // cached per loaded image; cleared on new load

    // Markup
    private string   _markupTool  = "pen";
    private Color    _markupColor = Color.FromRgb(0xFF, 0x45, 0x3A);
    private float    _brushSize   = 8;
    private bool     _drawing     = false;
    private Point    _drawStart;
    private List<UIElement> _markupStrokes = new();
    private Polyline? _currentPoly;
    private UIElement? _currentShape;
    private Ellipse?    _eraserCursor; // visual ring shown while erasing
    private TextBlock?  _rulerLabel;   // live distance label while ruler is dragged
    private float       _fontSize     = 24;
    private string      _fontFamily   = "Arial";
    private TextBox?    _activeTextBox;  // inline editor while text tool is placing
    private UIElement?  _draggedStroke;  // stroke being moved in hand mode
    private Point       _dragAnchor;     // canvas-space mouse position when drag started
    private Point       _handMouseStart; // position at mouse-down in hand mode
    private bool        _handDragging;   // true once movement threshold or long-press fires
    private readonly System.Windows.Threading.DispatcherTimer _longPressTimer = new()
        { Interval = TimeSpan.FromMilliseconds(280) };
    private bool _suppressAspectChanged = false;

    // Original pixels — set once on load, never changed; used for Revert
    private byte[]? _originalPixels;
    private int     _originalW, _originalH;

    // Serializable markup stroke — needed for full undo/redo
    private class StrokeData
    {
        public string Type = "";          // pen | brush | line | rect | arrow | ruler
        public Color  Color;
        public double Thickness;
        public List<Point>? Points;       // pen / brush polyline
        public double X1, Y1, X2, Y2;    // line / ruler shaft
        public double Left, Top, W, H;   // rect
        public List<Point>? HeadPts;     // arrowhead polygon
        public string? RulerText;        // ruler distance label
        public string? TextContent;      // text tool
        public double  FontSize;         // text tool font size
        public string? FontFamilyName;   // text tool font family
    }

    private class HistorySnapshot
    {
        public byte[]           Pixels;
        public int              W, H;
        public AdjustmentState  Adj;
        public List<StrokeData> Strokes;
        public bool             HasPendingCrop;
        public int              CropX, CropY, CropW, CropH;
        public HistorySnapshot(byte[] px, int w, int h, AdjustmentState adj, List<StrokeData> s,
            bool hasCrop = false, int cx = 0, int cy = 0, int cw = 0, int ch = 0)
        { Pixels = px; W = w; H = h; Adj = adj; Strokes = s;
          HasPendingCrop = hasCrop; CropX = cx; CropY = cy; CropW = cw; CropH = ch; }
    }

    // Universal undo/redo history
    private readonly Stack<HistorySnapshot> _history = new();
    private readonly Stack<HistorySnapshot> _future  = new();
    private const int MaxHistory = 30;

    // Flag for fast-preview mode during transform slider drag
    private bool _draggingTransform = false;

    // ── Curves editor state ──
    private bool   _curvesOpen    = false;
    private int    _curveDragIdx  = -1;     // which of the 5 control points is being dragged
    private const int CurvePtCount = 5;
    private System.Windows.Shapes.Ellipse[]? _curveHandles;


    // ── HSL Mixer state ──
    private bool  _hslOpen    = false;
    private int   _hslChannel = 0;   // 0=Red 1=Orange 2=Yellow 3=Green 4=Cyan 5=Blue 6=Purple
    private bool  _suppressHslSliders = false;
    private Border[]? _hslChannelBtns;

    // Crop
    private bool   _cropping = false;
    private Rect   _cropRect;
    private string _cropHandle = "";
    private Point  _cropDragStart;
    private Rect   _cropDragInitial;
    private string _cropOrientation = "Landscape"; // "Portrait" or "Landscape"

    private bool _isCropTabActive = false;
    private bool _hasPendingCrop  = false;
    private int  _pendCropX, _pendCropY, _pendCropW, _pendCropH;

    // Snapshot saved the moment Apply Crop is pressed — restored by Undo Crop button only
    private HistorySnapshot? _preCropSnapshot;
    private bool   _aspectDriveFromHeight = false; // true when N/S handles are dragged

    // Zoom / pan
    private Matrix _zoomMatrix = Matrix.Identity;
    private bool   _isPanning;
    private Point  _panAnchor;

    // JPEG export quality (60–100); shown + controlled from the status bar
    private int _jpegQuality = 85;

    // New feature state
    private string? _currentFilePath;
    private bool    _fullScreen      = false;
    private bool    _splitViewOn     = false;
    private bool    _exifOpen        = false;
    private bool    _inspectorVisible = true;
    private double  _splitRatio       = 0.5;   // 0–1, position of split divider
    private bool    _splitDragging    = false;
    private System.Windows.WindowState _preFullScreenState = System.Windows.WindowState.Normal;

    // Filter definitions — 18 filters
    private static readonly (FilterType type, string name, Color c1, Color c2)[] Filters =
    {
        // ── Neutral ──
        (FilterType.None,       "Original",    Color.FromRgb(0x55,0x55,0x55), Color.FromRgb(0xAA,0xAA,0xAA)),
        // ── Colorful ──
        (FilterType.Vivid,      "Vivid",       Color.FromRgb(0x0A,0x84,0xFF), Color.FromRgb(0xFF,0x2D,0x55)),
        (FilterType.Warm,       "Warm",        Color.FromRgb(0xFF,0x9F,0x0A), Color.FromRgb(0xFF,0x3A,0x20)),
        (FilterType.Cool,       "Cool",        Color.FromRgb(0x5A,0xC8,0xFA), Color.FromRgb(0x5E,0x5C,0xE6)),
        (FilterType.Sunset,     "Sunset",      Color.FromRgb(0xFF,0x6B,0x0A), Color.FromRgb(0xFF,0x20,0x40)),
        (FilterType.TealOrange, "Teal+Orange", Color.FromRgb(0xFF,0x7A,0x00), Color.FromRgb(0x00,0x8B,0x8B)),
        (FilterType.Process,    "Process",     Color.FromRgb(0x30,0xD1,0x58), Color.FromRgb(0xFF,0xD6,0x0A)),
        (FilterType.Lush,       "Lush",        Color.FromRgb(0x1A,0x7A,0x1A), Color.FromRgb(0x50,0xC8,0x50)),
        (FilterType.Retro,      "Retro",       Color.FromRgb(0xD4,0xA0,0x20), Color.FromRgb(0x80,0xC8,0x60)),
        (FilterType.Instant,    "Instant",     Color.FromRgb(0xE8,0xD0,0xA0), Color.FromRgb(0xB0,0x88,0x60)),
        (FilterType.Fade,       "Fade",        Color.FromRgb(0x7A,0x6E,0x66), Color.FromRgb(0xCE,0xC8,0xC0)),
        // ── Creative ──
        (FilterType.ZoomBlur,   "Zoom Blur",   Color.FromRgb(0x00,0xC8,0xFF), Color.FromRgb(0xF0,0xF0,0xFF)),
        (FilterType.Grainy,     "Grainy",      Color.FromRgb(0xC8,0xB4,0x78), Color.FromRgb(0x80,0x68,0x40)),
        (FilterType.Gritty,     "Gritty",      Color.FromRgb(0x50,0x42,0x30), Color.FromRgb(0x90,0x80,0x70)),
        // ── Dark / Tonal ──
        (FilterType.Chrome,     "Chrome",      Color.FromRgb(0x00,0xC8,0xFF), Color.FromRgb(0x22,0x22,0x44)),
        (FilterType.Sepia,      "Sepia",       Color.FromRgb(0x70,0x4E,0x28), Color.FromRgb(0xC8,0xA0,0x60)),
        (FilterType.Silvertone, "Silvertone",  Color.FromRgb(0x28,0x30,0x38), Color.FromRgb(0xC4,0xBE,0xAD)),
        (FilterType.Matte,      "Matte",       Color.FromRgb(0x60,0x60,0x70), Color.FromRgb(0xC0,0xC0,0xC8)),
        (FilterType.Dramatic,   "Dramatic",    Color.FromRgb(0x0D,0x10,0x14), Color.FromRgb(0x6E,0x7B,0x88)),
        (FilterType.Mono,       "Mono",        Color.FromRgb(0x11,0x11,0x11), Color.FromRgb(0xDD,0xDD,0xDD)),
        (FilterType.Noir,       "Noir",        Color.FromRgb(0x00,0x00,0x00), Color.FromRgb(0x44,0x44,0x44)),
    };

    public MainWindow()
    {
        InitializeComponent();
        BuildFilterGrid();
        BuildSwatches();
        BuildFontCombo();
        _longPressTimer.Tick += LongPressTimer_Tick;
        SetupRenderTimer();
        PopulateAspectRatios();

        // Push history before each slider drag or typed commit.
        // Also enable fast-preview mode (skip sharpen/noise) while any slider is being dragged.
        foreach (var sr in AllSliderRows())
        {
            sr.DragStarted   += (_, _) => { PushHistory(); _draggingTransform = true; };
            sr.DragCompleted += (_, _) => { _draggingTransform = false; ScheduleRender(); };
            sr.CommitChange  += (_, _) => PushHistory();
        }

        BuildHslChannelGrid();
        InitCurvesCanvas();

        // HSL sliders are raw Sliders (not SliderRow) — push a history entry when a
        // drag begins so HSL edits are undoable like every other adjustment.
        foreach (var sl in new[] { HslHueSlider, HslSatSlider })
            sl.AddHandler(System.Windows.Controls.Primitives.Thumb.DragStartedEvent,
                new System.Windows.Controls.Primitives.DragStartedEventHandler(
                    (_, _) => { if (_imageLoaded) PushHistory(); }));

        // SourceInitialized fires when the Win32 HWND is first created —
        // the earliest safe moment to call DWM APIs
        AppVersionLabel.Text = $"v{AppVersion.Current}";

        SourceInitialized += (_, _) => ApplyDarkTitleBar();
        Loaded  += (_, _) => { UpdateLayout(); _ = CheckForUpdateAsync(); };
        Closed  += (_, _) => _neuralEnhancer?.Dispose();
        SizeChanged += (_, _) => { if (_cropping) ClampCropToImage(); RefreshCropOverlay(); if (_splitViewOn) UpdateSplitView(); };
        this.Icon = CreateAppIcon();

        // Load neural enhancer in the background so startup is not delayed
        Task.Run(() =>
        {
            _neuralEnhancer = new NeuralEnhancer();
            // Diagnostic toasts removed — NN status visible via Auto button label.
        });
    }

    // ── Dark title bar via Windows DWM API ──
    // Use int (Win32 BOOL / DWORD) — more compatible than uint across DWM overloads
    [System.Runtime.InteropServices.DllImport("dwmapi.dll", PreserveSig = true)]
    private static extern int DwmSetWindowAttribute(IntPtr hwnd, int attr, ref int value, int size);

    private void ApplyDarkTitleBar()
    {
        try
        {
            var hwnd = new System.Windows.Interop.WindowInteropHelper(this).Handle;
            if (hwnd == IntPtr.Zero) return;

            int dark = 1;
            // Attr 20 = DWMWA_USE_IMMERSIVE_DARK_MODE (Win10 20H1 build 19041+ and Win11)
            DwmSetWindowAttribute(hwnd, 20, ref dark, 4);
            // Attr 19 = same flag but for older Win10 builds before 20H1
            DwmSetWindowAttribute(hwnd, 19, ref dark, 4);

            // Attr 35 = DWMWA_CAPTION_COLOR — sets exact title-bar background (Win11 only)
            // #111114 as COLORREF (0x00BBGGRR): R=0x11, G=0x11, B=0x14 → 0x00141111
            int captionColor = 0x00141111;
            DwmSetWindowAttribute(hwnd, 35, ref captionColor, 4);
        }
        catch { /* DWM unavailable — silently ignore */ }
    }

    private static BitmapImage CreateAppIcon()
    {
        // Render at 256×256 for sharp taskbar / title-bar icon
        const int size = 256;
        var dv = new DrawingVisual();
        using (var dc = dv.RenderOpen())
        {
            // Diagonal gradient: blue → green → orange
            var grad = new LinearGradientBrush();
            grad.GradientStops.Add(new GradientStop(Color.FromRgb(0x0A, 0x84, 0xFF), 0));
            grad.GradientStops.Add(new GradientStop(Color.FromRgb(0x30, 0xD1, 0x58), 0.45));
            grad.GradientStops.Add(new GradientStop(Color.FromRgb(0xFF, 0x9F, 0x0A), 1));
            grad.StartPoint = new Point(0, 0);
            grad.EndPoint   = new Point(1, 1);

            // Rounded square background
            dc.DrawRoundedRectangle(grad, null, new Rect(0, 0, size, size), 56, 56);

            // Subtle inner shadow / depth ring
            var innerPen = new Pen(new SolidColorBrush(Color.FromArgb(40, 0, 0, 0)), 6);
            dc.DrawRoundedRectangle(null, innerPen, new Rect(3, 3, size - 6, size - 6), 54, 54);

            // Bold "L" — shadow pass then white
            var typeface = new Typeface(new FontFamily("Segoe UI"),
                FontStyles.Normal, FontWeights.Bold, FontStretches.Normal);
            var shadow = new FormattedText("L",
                System.Globalization.CultureInfo.InvariantCulture,
                FlowDirection.LeftToRight, typeface, 180,
                new SolidColorBrush(Color.FromArgb(55, 0, 0, 0)), 96);
            dc.DrawText(shadow, new Point(78, 24));   // offset for drop-shadow

            var letter = new FormattedText("L",
                System.Globalization.CultureInfo.InvariantCulture,
                FlowDirection.LeftToRight, typeface, 180,
                Brushes.White, 96);
            dc.DrawText(letter, new Point(74, 20));
        }

        var rtb = new RenderTargetBitmap(size, size, 96, 96, PixelFormats.Pbgra32);
        rtb.Render(dv);

        var png = new PngBitmapEncoder();
        png.Frames.Add(BitmapFrame.Create(rtb));
        using var ms = new System.IO.MemoryStream();
        png.Save(ms);
        ms.Seek(0, System.IO.SeekOrigin.Begin);

        var bmi = new BitmapImage();
        bmi.BeginInit();
        bmi.StreamSource = ms;
        bmi.CacheOption  = BitmapCacheOption.OnLoad;
        bmi.EndInit();
        bmi.Freeze();

        // Also regenerate the .ico on disk so the next build embeds the updated icon
        Task.Run(() => TrySaveIcoFile(rtb));

        return bmi;
    }

    private static void TrySaveIcoFile(RenderTargetBitmap rtb)
    {
        try
        {
            // Build a minimal ICO container: header + 1 directory entry + PNG payload
            // (Windows Vista+ supports PNG-inside-ICO for 256×256)
            var enc = new PngBitmapEncoder();
            enc.Frames.Add(BitmapFrame.Create(rtb));
            using var pngMs = new System.IO.MemoryStream();
            enc.Save(pngMs);
            byte[] png = pngMs.ToArray();

            using var icoMs = new System.IO.MemoryStream();
            using var bw    = new System.IO.BinaryWriter(icoMs);
            bw.Write((ushort)0);               // reserved
            bw.Write((ushort)1);               // type = ICO
            bw.Write((ushort)1);               // image count = 1
            // Directory entry (16 bytes)
            bw.Write((byte)0);                 // width  0 → 256
            bw.Write((byte)0);                 // height 0 → 256
            bw.Write((byte)0);                 // colour count
            bw.Write((byte)0);                 // reserved
            bw.Write((ushort)1);               // planes
            bw.Write((ushort)32);              // bits per pixel
            bw.Write((uint)png.Length);        // image data size
            bw.Write((uint)22);                // offset to image data = 6 + 16
            bw.Write(png);

            // Save next to the exe (AppContext.BaseDirectory works in single-file apps)
            string dir     = AppContext.BaseDirectory;
            string icoPath = System.IO.Path.Combine(dir, "LumaPhoto.ico");
            System.IO.File.WriteAllBytes(icoPath, icoMs.ToArray());
        }
        catch { /* non-fatal */ }
    }

    // ── Render timer — debounce rapid slider changes, then render async ──
    private void SetupRenderTimer()
    {
        _renderTimer = new System.Windows.Threading.DispatcherTimer
        {
            Interval = TimeSpan.FromMilliseconds(25)  // snappier response
        };
        _renderTimer.Tick += async (_, _) =>
        {
            _renderTimer!.Stop();
            await DoRenderAsync();
        };
    }

    private void ScheduleRender()
    {
        _renderTimer?.Stop();
        _renderTimer?.Start();
    }

    // ── Async render — pixel work on background thread ──
    private async Task DoRenderAsync()
    {
        if (!_imageLoaded || _sourcePixels == null) return;

        _renderCts?.Cancel();
        _renderCts?.Dispose();
        _renderCts = new CancellationTokenSource();
        var token = _renderCts.Token;

        var pixels      = _sourcePixels;
        var w           = _sourceW;
        var h           = _sourceH;
        var adjSnap     = _adj.Clone();
        var compare     = _compareMode;
        bool fast       = _draggingTransform;

        // During transform drag, skip expensive sharpening/noise passes for speed
        if (fast)
        {
            adjSnap.Sharpness  = 0;
            adjSnap.Definition = 0;
            adjSnap.Noise      = 0;
        }

        WriteableBitmap? result = null;
        try
        {
            result = await Task.Run(() =>
            {
                token.ThrowIfCancellationRequested();
                if (compare) return PixelsToImage(pixels, w, h);
                token.ThrowIfCancellationRequested();

                byte[] composite = ImageProcessor.RenderToBuffer(pixels, w, h, adjSnap);
                return ImageProcessor.BufferToBitmap(composite, w, h);
            }, token);
        }
        catch (OperationCanceledException)
        {
            return;
        }

        if (!token.IsCancellationRequested && result != null)
        {
            PhotoDisplay.Source = result;
            ScheduleHistogram();
            UpdateStatusBar();
            if (_splitViewOn) UpdateSplitView();
        }
    }

    // Synchronous render for operations that commit pixel changes (rotate/flip/crop)
    // Returns source cropped for display; passthrough when in crop tab or no pending crop.
    private BitmapSource WithPendingCrop(WriteableBitmap src)
    {
        if (!_hasPendingCrop || _isCropTabActive) return src;
        var c = new CroppedBitmap(src, new Int32Rect(_pendCropX, _pendCropY, _pendCropW, _pendCropH));
        c.Freeze();
        return c;
    }

    private void DoRender()
    {
        if (!_imageLoaded || _sourcePixels == null) return;
        if (_compareMode)
        {
            PhotoDisplay.Source = PixelsToImage(_sourcePixels, _sourceW, _sourceH);
            return;
        }
        PhotoDisplay.Source = ImageProcessor.Render(_sourcePixels, _sourceW, _sourceH, _adj);
        if (_splitViewOn) UpdateSplitView();
    }

    private static WriteableBitmap PixelsToImage(byte[] pixels, int w, int h)
    {
        var wb = new WriteableBitmap(w, h, 96, 96, PixelFormats.Bgra32, null);
        wb.WritePixels(new Int32Rect(0, 0, w, h), pixels, w * 4, 0);
        wb.Freeze();
        return wb;
    }

    // ── Load image ──
    // Async so HEIC/large image decode never freezes the UI thread.
    // Cancels any prior in-flight load so rapid file switching stays responsive.
    private async void LoadImageFile(string path)
    {
        _loadCts?.Cancel();
        _loadCts?.Dispose();
        _loadCts = new CancellationTokenSource();
        var token = _loadCts.Token;

        byte[] pixels; int w, h;
        try
        {
            (pixels, w, h) = await Task.Run(() => ImageProcessor.LoadImageFile(path), token);
        }
        catch (OperationCanceledException) { return; }
        catch (Exception ex)
        {
            if (token.IsCancellationRequested) return;
            string ext = System.IO.Path.GetExtension(path).ToLowerInvariant();
            string msg = (ext == ".heic" || ext == ".heif")
                ? "Could not open HEIC file.\n\nInstall the free \"HEIC Image Extensions\" from the Microsoft Store to enable HEIC support:\nmicrosoft.com/store/apps/9PMMSR1CGPWG"
                : $"Could not open image:\n{ex.Message}";
            MessageBox.Show(msg, "Luma", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        if (token.IsCancellationRequested) return;

        _sourcePixels = pixels;
        _sourceW = w; _sourceH = h;
        // Save originals for Revert
        _originalPixels = (byte[])pixels.Clone();
        _originalW = w; _originalH = h;
        _fileName        = System.IO.Path.GetFileNameWithoutExtension(path);
        _currentFilePath = path;
        FileMetaText.Text = $"{System.IO.Path.GetFileName(path)}  ·  {w} × {h}";
        _imageLoaded = true;
        AddRecentFile(path);

        EmptyState.Visibility   = Visibility.Collapsed;
        PhotoDisplay.Visibility = Visibility.Visible;
        ResetZoom();

        // Trigger filmstrip + EXIF in background
        _ = LoadFilmstripAsync(System.IO.Path.GetDirectoryName(path) ?? "");
        _ = ReadExifAsync(path);

        if (_cropping) CancelCrop();
        _hasPendingCrop  = false;
        _preCropSnapshot = null;
        _autoNNWeights   = null;  // invalidate NN cache for the new image
        _history.Clear(); _future.Clear();
        ResetAll();
        UndoCropBtn.IsEnabled = false;
        RevertBtn.IsEnabled = true;
        SetControlsEnabled(true);
        MarkupCanvas.Visibility = Visibility.Collapsed;
        DoRender();
        if (TabCrop.IsChecked == true)
            Dispatcher.BeginInvoke(System.Windows.Threading.DispatcherPriority.Render,
                new Action(StartCropOverlay));
    }

    private void ResetAll()
    {
        _adj.Reset();
        _autoEnhanceOn  = false;
        _autoBaseParams = null;
        _autoPreState   = null;

        AutoBtn.Content    = "⚡ Auto";
        AutoBtn.Background = new SolidColorBrush(Color.FromRgb(0x0A, 0x84, 0xFF));
        AutoSubLabel.Text  = "Smart analysis of your photo";
        AutoIntensityPanel.Visibility = Visibility.Collapsed;
        AutoIntensitySlider.Value     = 0;
        AutoIntensityLabel.Text       = "Default";

        foreach (var sr in AllSliderRows()) sr.SetValueSilent(0);

        _adj.FilterIntensity        = 100;
        FilterIntensitySlider.Value = 100;
        FilterIntensityLabel.Text   = "100.0%";
        FilterIntensityPanel.Visibility = Visibility.Collapsed;
        foreach (var child in FilterGrid.Children)
            if (child is Border b)
                b.BorderBrush = new SolidColorBrush(Color.FromArgb(0x22, 0xFF, 0xFF, 0xFF));

        HighlightFilter(FilterType.None);

        MarkupCanvas.Children.Clear();
        _markupStrokes.Clear();
        UndoMarkupBtn.IsEnabled = false;

        _suppressHslSliders = true;
        HslHueSlider.Value = 0; HslSatSlider.Value = 0;
        HslHueLabel.Text = "0"; HslSatLabel.Text = "0";
        _suppressHslSliders = false;
        DrawCurves();
        ScheduleHistogram();
    }

    private SliderRow[]? _allSliderRows;
    private SliderRow[] AllSliderRows() => _allSliderRows ??= new[]
    {
        ExposureRow, BrillianceRow, HighlightsRow, ShadowsRow, ContrastRow, BrightnessRow, BlackPointRow,
        SaturationRow, VibranceRow, WarmthRow, TintRow,
        SharpnessRow, DefinitionRow, NoiseRow, VignetteRow,
        RotateZRow, TiltYRow, TiltXRow
    };

    private void SetControlsEnabled(bool on)
    {
        AutoBtn.IsEnabled      = on;
        CompareBtn.IsEnabled   = on;
        SplitViewBtn.IsEnabled = on;
        PresetsBtn.IsEnabled   = on;
        LooksBtn.IsEnabled     = on;
        CopySettingsBtn.IsEnabled = on;
        PasteSettingsBtn.IsEnabled = on && _copiedLook != null;
        ExportBtn.IsEnabled    = on;
        BatchBtn.IsEnabled     = on;
        ResetCurveBtn.IsEnabled = on;
        HslHueSlider.IsEnabled  = on;
        HslSatSlider.IsEnabled  = on;
        RotateLeftBtn.IsEnabled  = on;
        RotateRightBtn.IsEnabled = on;
        FlipHBtn.IsEnabled   = on;
        FlipVBtn.IsEnabled   = on;
        AspectCombo.IsEnabled  = on;
        PortraitOrientBtn.IsEnabled  = on;
        LandscapeOrientBtn.IsEnabled = on;
        BrushSlider.IsEnabled      = on;
        FontSizeSlider.IsEnabled   = on;
        FontFamilyCombo.IsEnabled  = on;
        PenTool.IsEnabled    = on;
        BrushTool.IsEnabled  = on;
        EraserTool.IsEnabled = on;
        LineTool.IsEnabled   = on;
        RectTool.IsEnabled   = on;
        ArrowTool.IsEnabled  = on;
        RulerTool.IsEnabled  = on;
        TextTool.IsEnabled   = on;
        UndoBtn.IsEnabled = _history.Count > 0 && on;
        RedoBtn.IsEnabled = _future.Count  > 0 && on;
        UndoMarkupBtn.IsEnabled  = false;
        ClearMarkupBtn.IsEnabled = on;
        foreach (var sr in AllSliderRows()) sr.SetEnabled(on);
        foreach (var child in FilterGrid.Children)
            if (child is Border b) b.Opacity = on ? 1.0 : 0.4;
    }

    // ── Open / Drop ──
    private void OpenBtn_Click(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFileDialog
        {
            Title  = "Open Photo",
            Filter = "Images|*.jpg;*.jpeg;*.png;*.bmp;*.webp;*.tiff;*.tif;*.gif;*.heic;*.heif|All files|*.*"
        };
        if (dlg.ShowDialog() == true) LoadImageFile(dlg.FileName);
    }

    private void Stage_DragOver(object sender, DragEventArgs e)
    {
        e.Effects = e.Data.GetDataPresent(DataFormats.FileDrop) ? DragDropEffects.Copy : DragDropEffects.None;
        e.Handled = true;
    }

    private void Stage_Drop(object sender, DragEventArgs e)
    {
        if (e.Data.GetData(DataFormats.FileDrop) is string[] files && files.Length > 0)
            LoadImageFile(files[0]);
    }

    // ── Export ──
    private void ExportBtn_Click(object sender, RoutedEventArgs e)
    {
        if (!_imageLoaded || _sourcePixels == null) return;
        var dlg = new SaveFileDialog
        {
            Title    = "Export Photo",
            FileName = $"{_fileName}-luma",
            Filter      = "JPEG Image|*.jpg;*.jpeg|PNG Image|*.png|TIFF Image|*.tiff;*.tif",
            FilterIndex = 1
        };
        if (dlg.ShowDialog() != true) return;

        var rendered = ImageProcessor.Render(_sourcePixels, _sourceW, _sourceH, _adj);
        BitmapSource exportSrc = _hasPendingCrop
            ? new CroppedBitmap(rendered, new Int32Rect(_pendCropX, _pendCropY, _pendCropW, _pendCropH))
            : (BitmapSource)rendered;

        // Flatten markup strokes onto the exported image
        if (_markupStrokes.Count > 0)
            exportSrc = CompositeMarkupOnExport(exportSrc);

        string ext   = System.IO.Path.GetExtension(dlg.FileName).ToLowerInvariant();

        try
        {
            BitmapEncoder encoder = ext switch
            {
                ".jpg" or ".jpeg" => new JpegBitmapEncoder { QualityLevel = _jpegQuality },
                ".tiff" or ".tif" => new TiffBitmapEncoder(),
                _                 => new PngBitmapEncoder()
            };
            encoder.Frames.Add(BitmapFrame.Create(exportSrc));
            using var fs = File.Create(dlg.FileName);
            encoder.Save(fs);

            ShowSaveSuccessDialog(dlg.FileName);
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Export failed:\n{ex.Message}", "Luma",
                MessageBoxButton.OK, MessageBoxImage.Warning);
        }
    }

    // ── Flatten markup canvas onto the photo for export ──
    private BitmapSource CompositeMarkupOnExport(BitmapSource photo)
    {
        if (_markupStrokes.Count == 0) return photo;

        int pw = photo.PixelWidth, ph = photo.PixelHeight;
        var imgBounds = GetImageDisplayBounds();
        if (imgBounds.Width <= 0 || imgBounds.Height <= 0) return photo;

        double scaleX = pw / imgBounds.Width;
        double scaleY = ph / imgBounds.Height;

        // Canvas lives inside ImageZoomGroup which fills StageGrid
        double cw = StageGrid.ActualWidth;
        double ch = StageGrid.ActualHeight;
        if (cw <= 1 || ch <= 1) return photo; // safety: stage not yet laid out

        // Rebuild all strokes in a fresh off-screen canvas
        // (UIElements can only have one parent; we can't reuse the live ones)
        var temp = new Canvas { Width = cw, Height = ch, Background = Brushes.Transparent };
        foreach (var sd in CaptureMarkupStrokes())
            temp.Children.Add(BuildStroke(sd));

        temp.Measure(new Size(cw, ch));
        temp.Arrange(new Rect(0, 0, cw, ch));

        var markupBmp = new RenderTargetBitmap((int)cw, (int)ch, 96, 96, PixelFormats.Pbgra32);
        markupBmp.Render(temp);
        markupBmp.Freeze();

        // Composite: photo first, then markup scaled from canvas-space → source-pixel-space
        // TranslateTransform shifts so the image top-left becomes origin;
        // ScaleTransform maps display-px to source-px.
        var dv = new DrawingVisual();
        using (var dc = dv.RenderOpen())
        {
            dc.DrawImage(photo, new Rect(0, 0, pw, ph));
            dc.PushTransform(new ScaleTransform(scaleX, scaleY));
            dc.PushTransform(new TranslateTransform(-imgBounds.X, -imgBounds.Y));
            dc.DrawImage(markupBmp, new Rect(0, 0, cw, ch));
            dc.Pop();
            dc.Pop();
        }

        var result = new RenderTargetBitmap(pw, ph, 96, 96, PixelFormats.Pbgra32);
        result.Render(dv);
        result.Freeze();
        return result;
    }

    // ── Unified toast dialog (green for success, red accent for errors) ──
    private void ShowToast(string title, string body, bool success = true)
    {
        var dlg = new Window
        {
            Width = 390, SizeToContent = SizeToContent.Height,
            WindowStyle = WindowStyle.None, AllowsTransparency = true,
            Background = Brushes.Transparent, ResizeMode = ResizeMode.NoResize,
            WindowStartupLocation = WindowStartupLocation.CenterOwner,
            Owner = this, ShowInTaskbar = false
        };

        var card = new Border
        {
            Background    = new SolidColorBrush(Color.FromRgb(0x1C, 0x1C, 0x1E)),
            CornerRadius  = new CornerRadius(14),
            Padding       = new Thickness(22, 18, 22, 18),
            Effect        = new System.Windows.Media.Effects.DropShadowEffect
                { Color = Colors.Black, Opacity = 0.75, BlurRadius = 28, ShadowDepth = 4 }
        };

        var stack = new StackPanel();

        var headerColor = success
            ? Color.FromRgb(0x30, 0xD1, 0x58)   // green
            : Color.FromRgb(0xFF, 0x45, 0x3A);   // red

        var header = new TextBlock
        {
            Text = title, FontSize = 15, FontWeight = FontWeights.SemiBold,
            Foreground = new SolidColorBrush(headerColor),
            Margin = new Thickness(0, 0, 0, 8)
        };
        stack.Children.Add(header);

        string display = body.Length > 120 ? "…" + body[^117..] : body;
        stack.Children.Add(new TextBlock
        {
            Text = display, FontSize = 11,
            Foreground = new SolidColorBrush(Color.FromRgb(0x8E, 0x8E, 0xA0)),
            TextWrapping = TextWrapping.Wrap,
            Margin = new Thickness(0, 0, 0, 18)
        });

        var okBtn = new Button
        {
            Content = "OK", Width = 90, Height = 34,
            HorizontalAlignment = HorizontalAlignment.Right,
            Foreground = Brushes.White, FontSize = 13,
            FontWeight = FontWeights.Medium, Cursor = Cursors.Hand
        };
        var tpl   = new ControlTemplate(typeof(Button));
        var bdFac = new FrameworkElementFactory(typeof(Border));
        bdFac.SetValue(Border.BackgroundProperty,
            new SolidColorBrush(Color.FromRgb(0x0A, 0x84, 0xFF)));
        bdFac.SetValue(Border.CornerRadiusProperty, new CornerRadius(9));
        var cpFac = new FrameworkElementFactory(typeof(ContentPresenter));
        cpFac.SetValue(ContentPresenter.HorizontalAlignmentProperty, HorizontalAlignment.Center);
        cpFac.SetValue(ContentPresenter.VerticalAlignmentProperty,   VerticalAlignment.Center);
        bdFac.AppendChild(cpFac);
        tpl.VisualTree = bdFac;
        okBtn.Template = tpl;
        okBtn.Click   += (_, _) => dlg.Close();
        stack.Children.Add(okBtn);

        card.Child  = stack;
        dlg.Content = card;
        dlg.KeyDown += (_, e) =>
        { if (e.Key == Key.Return || e.Key == Key.Escape) dlg.Close(); };
        dlg.ShowDialog();
    }

    // Keep backward-compat alias used by export
    private void ShowSaveSuccessDialog(string filePath)
        => ShowToast("✓  Image Saved", filePath);

    // ── Revert to original photo (undoes all crops/flips/rotates) ──
    private void RevertBtn_Click(object sender, RoutedEventArgs e)
    {
        if (!_imageLoaded || _originalPixels == null) return;
        if (MessageBox.Show(
                "Revert to the original photo?\nAll crops, rotations and edits will be lost.",
                "Luma — Revert to Original",
                MessageBoxButton.YesNo, MessageBoxImage.Warning) != MessageBoxResult.Yes) return;
        _sourcePixels = (byte[])_originalPixels.Clone();
        _sourceW = _originalW; _sourceH = _originalH;
        FileMetaText.Text = $"{_fileName}  ·  {_sourceW} × {_sourceH}";
        if (_cropping) CancelCrop();
        _hasPendingCrop  = false;
        _preCropSnapshot = null;
        UndoCropBtn.IsEnabled = false;
        ResetAll();
        DoRender();
    }

    // ── Universal undo/redo ──

    private List<StrokeData> CaptureMarkupStrokes()
    {
        if (_markupStrokes.Count == 0) return new List<StrokeData>();
        var result = new List<StrokeData>();
        foreach (var elem in _markupStrokes)
        {
            StrokeData? d = null;
            if (elem is Polyline poly)
            {
                // pen = opaque, brush = semi-transparent (alpha < 255)
                var col = ((SolidColorBrush)poly.Stroke).Color;
                d = new StrokeData { Type = col.A < 255 ? "brush" : "pen",
                    Color = col, Thickness = poly.StrokeThickness, Points = poly.Points.ToList() };
            }
            else if (elem is Line ln)
                d = new StrokeData { Type = "line", Color = ((SolidColorBrush)ln.Stroke).Color,
                    Thickness = ln.StrokeThickness, X1 = ln.X1, Y1 = ln.Y1, X2 = ln.X2, Y2 = ln.Y2 };
            else if (elem is Rectangle rc)
                d = new StrokeData { Type = "rect", Color = ((SolidColorBrush)rc.Stroke).Color,
                    Thickness = rc.StrokeThickness,
                    Left = Canvas.GetLeft(rc), Top = Canvas.GetTop(rc), W = rc.Width, H = rc.Height };
            else if (elem is TextBlock txt)
                d = new StrokeData { Type = "text",
                    Color = ((SolidColorBrush)txt.Foreground).Color,
                    FontSize = txt.FontSize, FontFamilyName = txt.FontFamily.Source,
                    Left = Canvas.GetLeft(txt), Top = Canvas.GetTop(txt),
                    TextContent = txt.Text };
            else if (elem is Canvas grp)
            {
                var shaft = grp.Children.OfType<Line>().FirstOrDefault();
                if ((string?)grp.Tag == "ruler")
                {
                    var lbl = grp.Children.OfType<TextBlock>().FirstOrDefault();
                    if (shaft != null)
                        d = new StrokeData { Type = "ruler",
                            Color = ((SolidColorBrush)shaft.Stroke).Color, Thickness = shaft.StrokeThickness,
                            X1 = shaft.X1, Y1 = shaft.Y1, X2 = shaft.X2, Y2 = shaft.Y2,
                            RulerText = lbl?.Text ?? "" };
                }
                else // arrow group
                {
                    var head = grp.Children.OfType<Polygon>().FirstOrDefault();
                    if (shaft != null)
                        d = new StrokeData { Type = "arrow",
                            Color = ((SolidColorBrush)shaft.Stroke).Color, Thickness = shaft.StrokeThickness,
                            X1 = shaft.X1, Y1 = shaft.Y1, X2 = shaft.X2, Y2 = shaft.Y2,
                            HeadPts = head?.Points.ToList() };
                }
            }
            if (d != null) result.Add(d);
        }
        return result;
    }

    private UIElement BuildStroke(StrokeData d)
    {
        var brush = new SolidColorBrush(d.Color);
        switch (d.Type)
        {
            case "pen":
            case "brush":
                var poly = new Polyline { Stroke = brush, StrokeThickness = d.Thickness,
                    StrokeLineJoin = PenLineJoin.Round,
                    StrokeStartLineCap = PenLineCap.Round, StrokeEndLineCap = PenLineCap.Round };
                foreach (var pt in d.Points!) poly.Points.Add(pt);
                return poly;
            case "line":
                return new Line { Stroke = brush, StrokeThickness = d.Thickness,
                    X1 = d.X1, Y1 = d.Y1, X2 = d.X2, Y2 = d.Y2 };
            case "rect":
                var rc = new Rectangle { Stroke = brush, StrokeThickness = d.Thickness,
                    Fill = Brushes.Transparent, Width = d.W, Height = d.H };
                Canvas.SetLeft(rc, d.Left); Canvas.SetTop(rc, d.Top);
                return rc;
            case "text":
                var lbl = new TextBlock
                {
                    Text = d.TextContent ?? "", Foreground = brush,
                    FontSize = d.FontSize, FontWeight = FontWeights.SemiBold,
                    FontFamily = new FontFamily(d.FontFamilyName ?? "Arial"),
                    TextWrapping = TextWrapping.Wrap,
                    IsHitTestVisible = false,
                    Effect = new System.Windows.Media.Effects.DropShadowEffect
                        { Color = Colors.Black, BlurRadius = 4, ShadowDepth = 1, Opacity = 0.7 }
                };
                Canvas.SetLeft(lbl, d.Left); Canvas.SetTop(lbl, d.Top);
                return lbl;
            case "ruler":
                return BuildRulerCanvas(d.Color, d.Thickness, d.X1, d.Y1, d.X2, d.Y2, d.RulerText ?? "");
            case "arrow":
                var grp = new Canvas { IsHitTestVisible = false };
                grp.Children.Add(new Line { Stroke = brush, StrokeThickness = d.Thickness,
                    X1 = d.X1, Y1 = d.Y1, X2 = d.X2, Y2 = d.Y2 });
                if (d.HeadPts != null)
                {
                    var head = new Polygon { Fill = brush };
                    foreach (var pt in d.HeadPts) head.Points.Add(pt);
                    grp.Children.Add(head);
                }
                return grp;
            default: return new Line();
        }
    }

    // ── Build a ruler Canvas: shaft line + tick marks + distance label ──
    private static UIElement BuildRulerCanvas(Color col, double thickness, double x1, double y1, double x2, double y2, string label)
    {
        var brush = new SolidColorBrush(col);
        var grp   = new Canvas { IsHitTestVisible = false, Tag = "ruler" };

        // Shaft
        grp.Children.Add(new Line { Stroke = brush, StrokeThickness = Math.Max(1, thickness * 0.6),
            X1 = x1, Y1 = y1, X2 = x2, Y2 = y2,
            StrokeStartLineCap = PenLineCap.Round, StrokeEndLineCap = PenLineCap.Round });

        // End caps (short perpendicular bars)
        double dx = x2 - x1, dy = y2 - y1;
        double len = Math.Sqrt(dx * dx + dy * dy);
        if (len > 4)
        {
            double px = -dy / len * 7, py = dx / len * 7; // perpendicular ±7px
            double tw = Math.Max(1, thickness * 0.5);
            foreach (var (bx, by) in new[] { (x1, y1), (x2, y2) })
                grp.Children.Add(new Line { Stroke = brush, StrokeThickness = tw,
                    X1 = bx + px, Y1 = by + py, X2 = bx - px, Y2 = by - py });

            // Tick marks every 25px along the shaft
            double nx = dx / len, ny = dy / len;
            double tpx = px * 0.4, tpy = py * 0.4; // shorter ticks
            for (double t = 25; t < len - 10; t += 25)
                grp.Children.Add(new Line { Stroke = brush, StrokeThickness = tw,
                    X1 = x1 + nx * t + tpx, Y1 = y1 + ny * t + tpy,
                    X2 = x1 + nx * t - tpx, Y2 = y1 + ny * t - tpy });
        }

        // Distance label
        if (!string.IsNullOrEmpty(label))
        {
            var tb = new TextBlock
            {
                Text = label, Foreground = brush, FontSize = 11,
                FontWeight = FontWeights.SemiBold,
                Effect = new System.Windows.Media.Effects.DropShadowEffect
                    { Color = Colors.Black, BlurRadius = 4, ShadowDepth = 1, Opacity = 0.8 }
            };
            Canvas.SetLeft(tb, x2 + 6);
            Canvas.SetTop(tb,  y2 - 8);
            grp.Children.Add(tb);
        }
        return grp;
    }

    // ── Eraser helpers ──
    private void EraseAtPoint(Point pt)
    {
        float radius = _brushSize * 1.5f;
        var toRemove = _markupStrokes.Where(s => StrokeHitsPoint(s, pt, radius)).ToList();
        foreach (var s in toRemove)
        {
            MarkupCanvas.Children.Remove(s);
            _markupStrokes.Remove(s);
        }
        UndoMarkupBtn.IsEnabled = _markupStrokes.Count > 0 && _imageLoaded;
    }

    private static bool StrokeHitsPoint(UIElement stroke, Point pt, float radius)
    {
        if (stroke is Polyline poly)
            return poly.Points.Any(p => Dist(p, pt) <= radius + poly.StrokeThickness / 2);
        if (stroke is Line ln)
            return DistToSeg(pt, new Point(ln.X1, ln.Y1), new Point(ln.X2, ln.Y2)) <= radius + ln.StrokeThickness / 2;
        if (stroke is Rectangle rc)
        {
            double l = Canvas.GetLeft(rc), t = Canvas.GetTop(rc);
            // Check if pt is near any of the 4 edges
            return DistToSeg(pt, new Point(l, t), new Point(l + rc.Width, t)) <= radius ||
                   DistToSeg(pt, new Point(l + rc.Width, t), new Point(l + rc.Width, t + rc.Height)) <= radius ||
                   DistToSeg(pt, new Point(l + rc.Width, t + rc.Height), new Point(l, t + rc.Height)) <= radius ||
                   DistToSeg(pt, new Point(l, t + rc.Height), new Point(l, t)) <= radius;
        }
        if (stroke is TextBlock tb)
        {
            double l = Canvas.GetLeft(tb), t = Canvas.GetTop(tb);
            return pt.X >= l - radius && pt.X <= l + tb.ActualWidth  + radius
                && pt.Y >= t - radius && pt.Y <= t + tb.ActualHeight + radius;
        }
        if (stroke is Canvas grp)
        {
            var shaft = grp.Children.OfType<Line>().FirstOrDefault();
            if (shaft != null)
                return DistToSeg(pt, new Point(shaft.X1, shaft.Y1), new Point(shaft.X2, shaft.Y2)) <= radius + shaft.StrokeThickness / 2;
        }
        return false;
    }

    private static void BakeTranslate(UIElement elem, double dx, double dy)
    {
        elem.RenderTransform = null;
        if (elem is Polyline poly)
        {
            var pts = poly.Points.ToList(); poly.Points.Clear();
            foreach (var p in pts) poly.Points.Add(new Point(p.X + dx, p.Y + dy));
        }
        else if (elem is Line ln)    { ln.X1 += dx; ln.Y1 += dy; ln.X2 += dx; ln.Y2 += dy; }
        else if (elem is Rectangle)  { Canvas.SetLeft(elem, Canvas.GetLeft(elem) + dx); Canvas.SetTop(elem, Canvas.GetTop(elem) + dy); }
        else if (elem is TextBlock)  { Canvas.SetLeft(elem, Canvas.GetLeft(elem) + dx); Canvas.SetTop(elem, Canvas.GetTop(elem) + dy); }
        else if (elem is Canvas grp)
        {
            foreach (UIElement child in grp.Children)
            {
                if (child is Line cln) { cln.X1 += dx; cln.Y1 += dy; cln.X2 += dx; cln.Y2 += dy; }
                else if (child is Polygon cpoly)
                {
                    var pts = cpoly.Points.ToList(); cpoly.Points.Clear();
                    foreach (var p in pts) cpoly.Points.Add(new Point(p.X + dx, p.Y + dy));
                }
                else if (child is TextBlock ctb) { Canvas.SetLeft(ctb, Canvas.GetLeft(ctb) + dx); Canvas.SetTop(ctb, Canvas.GetTop(ctb) + dy); }
            }
        }
    }

    private void LongPressTimer_Tick(object? sender, EventArgs e)
    {
        _longPressTimer.Stop();
        _handDragging = true;
        MarkupCanvas.Cursor = Cursors.SizeAll;
    }

    private void OpenTextEditor(TextBlock tb)
    {
        CommitTextBox();
        PushHistory();
        _markupStrokes.Remove(tb);
        MarkupCanvas.Children.Remove(tb);

        _markupColor = ((SolidColorBrush)tb.Foreground).Color;
        _fontSize    = (float)tb.FontSize;
        _fontFamily  = tb.FontFamily.Source;

        _activeTextBox = new TextBox
        {
            Text             = tb.Text,
            Background       = Brushes.Transparent,
            BorderThickness  = new Thickness(1),
            BorderBrush      = new SolidColorBrush(Color.FromArgb(140, 255, 255, 255)),
            Foreground       = tb.Foreground,
            CaretBrush       = tb.Foreground,
            FontSize         = tb.FontSize,
            FontWeight       = tb.FontWeight,
            FontFamily       = tb.FontFamily,
            MinWidth         = 60,
            AcceptsReturn    = true,
            TextWrapping     = TextWrapping.Wrap,
            IsHitTestVisible = true,
        };
        Canvas.SetLeft(_activeTextBox, Canvas.GetLeft(tb));
        Canvas.SetTop (_activeTextBox, Canvas.GetTop(tb));
        _activeTextBox.KeyDown   += TextBox_KeyDown;
        _activeTextBox.LostFocus += TextBox_LostFocus;
        MarkupCanvas.Children.Add(_activeTextBox);
        _activeTextBox.Focus();
        _activeTextBox.SelectAll();
    }

    private static double Dist(Point a, Point b)
        => Math.Sqrt((a.X - b.X) * (a.X - b.X) + (a.Y - b.Y) * (a.Y - b.Y));

    private static double DistToSeg(Point p, Point a, Point b)
    {
        double dx = b.X - a.X, dy = b.Y - a.Y, lenSq = dx * dx + dy * dy;
        if (lenSq == 0) return Dist(p, a);
        double t = Math.Clamp(((p.X - a.X) * dx + (p.Y - a.Y) * dy) / lenSq, 0, 1);
        return Dist(p, new Point(a.X + t * dx, a.Y + t * dy));
    }

    private void RestoreMarkupStrokes(List<StrokeData> strokes)
    {
        MarkupCanvas.Children.Clear();
        _markupStrokes.Clear();
        foreach (var d in strokes)
        {
            var elem = BuildStroke(d);
            MarkupCanvas.Children.Add(elem);
            _markupStrokes.Add(elem);
        }
        UndoMarkupBtn.IsEnabled = _markupStrokes.Count > 0 && _imageLoaded;
    }

    private void PushHistory()
    {
        if (!_imageLoaded || _sourcePixels == null) return;
        _history.Push(new HistorySnapshot(
            (byte[])_sourcePixels.Clone(), _sourceW, _sourceH,
            _adj.Clone(), CaptureMarkupStrokes(),
            _hasPendingCrop, _pendCropX, _pendCropY, _pendCropW, _pendCropH));
        if (_history.Count > MaxHistory)
        {
            // Drop the oldest entry (bottom of stack) to stay within cap
            var arr = _history.ToArray(); // index 0 = newest
            _history.Clear();
            for (int i = MaxHistory - 1; i >= 0; i--) _history.Push(arr[i]);
        }
        _future.Clear();
        UpdateUndoRedoButtons();
    }

    // Restore the full adjustment state from a snapshot — including curves and HSL.
    private void SetAdjFromSnapshot(AdjustmentState a) => _adj.CopyFrom(a);

    private void RestoreSnapshot(HistorySnapshot snap)
    {
        _sourcePixels = (byte[])snap.Pixels.Clone();
        _sourceW = snap.W; _sourceH = snap.H;
        _hasPendingCrop = snap.HasPendingCrop;
        _pendCropX = snap.CropX; _pendCropY = snap.CropY;
        _pendCropW = snap.CropW; _pendCropH = snap.CropH;
        SetAdjFromSnapshot(snap.Adj);
        ApplyAdjToSliders(snap.Adj);
        HighlightFilter(snap.Adj.Filter);
        FilterIntensitySlider.Value = snap.Adj.FilterIntensity;
        FilterIntensityLabel.Text   = $"{snap.Adj.FilterIntensity:0.0}%";
        FilterIntensityPanel.Visibility = snap.Adj.Filter != FilterType.None
            ? Visibility.Visible : Visibility.Collapsed;
        RestoreMarkupStrokes(snap.Strokes);
        DrawCurves();             // reflect restored tone-curve handles
        SyncHslSlidersFromAdj();  // reflect restored HSL values
        FileMetaText.Text = $"{_fileName}  ·  {_sourceW} × {_sourceH}";
        if (_cropping) CancelCrop();
        TurnOffAutoEnhanceUI();
        RevertBtn.IsEnabled = true;
        DoRender();
        UpdateUndoRedoButtons();
        if (_isCropTabActive && _imageLoaded)
            Dispatcher.BeginInvoke(System.Windows.Threading.DispatcherPriority.Render,
                new Action(StartCropOverlay));
    }

    private void UpdateUndoRedoButtons()
    {
        UndoBtn.IsEnabled = _history.Count > 0 && _imageLoaded;
        RedoBtn.IsEnabled = _future.Count  > 0 && _imageLoaded;
    }

    private void UndoBtn_Click(object sender, RoutedEventArgs e)
    {
        if (!_imageLoaded || _history.Count == 0 || _sourcePixels == null) return;
        _future.Push(new HistorySnapshot(
            (byte[])_sourcePixels.Clone(), _sourceW, _sourceH,
            _adj.Clone(), CaptureMarkupStrokes()));
        RestoreSnapshot(_history.Pop());
    }

    private void RedoBtn_Click(object sender, RoutedEventArgs e)
    {
        if (!_imageLoaded || _future.Count == 0 || _sourcePixels == null) return;
        _history.Push(new HistorySnapshot(
            (byte[])_sourcePixels.Clone(), _sourceW, _sourceH,
            _adj.Clone(), CaptureMarkupStrokes()));
        RestoreSnapshot(_future.Pop());
    }

    // ── Compare ──
    private void CompareBtn_MouseDown(object sender, MouseButtonEventArgs e)
    {
        _compareMode = true; DoRender();
    }
    private void CompareBtn_MouseUp(object sender, MouseButtonEventArgs e)
    {
        _compareMode = false; DoRender();
    }

    // ── Tabs ──
    private void Tab_Click(object sender, RoutedEventArgs e)
    {
        if (sender is not ToggleButton clicked) return;
        string tag = (string)clicked.Tag;
        bool wasOnCropTab = _isCropTabActive;

        // Leaving Crop tab without applying → discard the unsaved selection
        if (_cropping && tag != "Crop")
            CancelCrop();

        foreach (ToggleButton tb in new[] { TabAdjust, TabFilters, TabCrop, TabMarkup })
            tb.IsChecked = tb == clicked;
        AdjustPanel.Visibility  = tag == "Adjust"  ? Visibility.Visible : Visibility.Collapsed;
        FiltersPanel.Visibility = tag == "Filters" ? Visibility.Visible : Visibility.Collapsed;
        CropPanel.Visibility    = tag == "Crop"    ? Visibility.Visible : Visibility.Collapsed;
        MarkupPanel.Visibility  = tag == "Markup"  ? Visibility.Visible : Visibility.Collapsed;
        if (_imageLoaded)
            MarkupCanvas.Visibility = tag == "Markup" ? Visibility.Visible : Visibility.Collapsed;

        _isCropTabActive = tag == "Crop";

        // Crop & Markup overlays are positioned in un-zoomed image coordinates, so
        // reset any zoom/pan first to keep the overlay aligned with the photo.
        if ((tag == "Crop" || tag == "Markup") && _imageLoaded)
            ResetZoom();

        // Entering Crop tab → auto-start crop overlay
        if (tag == "Crop" && _imageLoaded && !_cropping)
            StartCropOverlay();

        // Re-render whenever crop tab entry/exit changes what gets displayed
        if (_imageLoaded && wasOnCropTab != _isCropTabActive)
            ScheduleRender();
    }

    // ── Adjustment sliders ──
    private void Adj_Changed(object sender, double value)
    {
        if (!_imageLoaded) return;
        // If the user manually touches a slider while auto is on, turn auto off
        if (_autoEnhanceOn) TurnOffAutoEnhanceUI();
        SyncAdjFromSliders();
        ScheduleRender();
    }

    private void SyncAdjFromSliders()
    {
        _adj.Exposure   = (float)ExposureRow.ValueSlider.Value;
        _adj.Brilliance = (float)BrillianceRow.ValueSlider.Value;
        _adj.Highlights = (float)HighlightsRow.ValueSlider.Value;
        _adj.Shadows    = (float)ShadowsRow.ValueSlider.Value;
        _adj.Contrast   = (float)ContrastRow.ValueSlider.Value;
        _adj.Brightness = (float)BrightnessRow.ValueSlider.Value;
        _adj.BlackPoint = (float)BlackPointRow.ValueSlider.Value;
        _adj.Saturation = (float)SaturationRow.ValueSlider.Value;
        _adj.Vibrance   = (float)VibranceRow.ValueSlider.Value;
        _adj.Warmth     = (float)WarmthRow.ValueSlider.Value;
        _adj.Tint       = (float)TintRow.ValueSlider.Value;
        _adj.Sharpness  = (float)SharpnessRow.ValueSlider.Value;
        _adj.Definition = (float)DefinitionRow.ValueSlider.Value;
        _adj.Noise      = (float)NoiseRow.ValueSlider.Value;
        _adj.Vignette   = (float)VignetteRow.ValueSlider.Value;
        _adj.RotateZ    = (float)RotateZRow.ValueSlider.Value;
        _adj.TiltX      = (float)TiltXRow.ValueSlider.Value;
        _adj.TiltY      = (float)TiltYRow.ValueSlider.Value;
    }

    private void ApplyAdjToSliders(AdjustmentState a)
    {
        ExposureRow.SetValueSilent(a.Exposure);
        BrillianceRow.SetValueSilent(a.Brilliance);
        HighlightsRow.SetValueSilent(a.Highlights);
        ShadowsRow.SetValueSilent(a.Shadows);
        ContrastRow.SetValueSilent(a.Contrast);
        BrightnessRow.SetValueSilent(a.Brightness);
        BlackPointRow.SetValueSilent(a.BlackPoint);
        SaturationRow.SetValueSilent(a.Saturation);
        VibranceRow.SetValueSilent(a.Vibrance);
        WarmthRow.SetValueSilent(a.Warmth);
        TintRow.SetValueSilent(a.Tint);
        SharpnessRow.SetValueSilent(a.Sharpness);
        DefinitionRow.SetValueSilent(a.Definition);
        NoiseRow.SetValueSilent(a.Noise);
        VignetteRow.SetValueSilent(a.Vignette);
        RotateZRow.SetValueSilent(a.RotateZ);
        TiltXRow.SetValueSilent(a.TiltX);
        TiltYRow.SetValueSilent(a.TiltY);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // AUTO ENHANCE
    //
    // Behaviour (matches iPhone Photos):
    //   • Tap Auto: analyze image, compute target params, show slider at 0 (= full auto).
    //   • Slider at  0  → full enhancement (the "default" auto-enhanced look).
    //   • Slider at +100 → push every param ~1.7× stronger than auto.
    //   • Slider at -100 → smoothly return to exactly what was there BEFORE auto.
    //   • Tap Auto again (toggle off): restore the pre-auto state entirely.
    // ─────────────────────────────────────────────────────────────────────────

    private void AutoBtn_Click(object sender, RoutedEventArgs e)
    {
        if (!_imageLoaded || _sourcePixels == null) return;

        if (!_autoEnhanceOn)
        {
            // ── Turning auto ON ──
            // 1. Snapshot current adj state so we can restore it if turned off
            _autoPreState = _adj.Clone();

            if (_autoBaseParams == null)
            {
                var analysis    = ImageProcessor.Analyze(_sourcePixels, _sourceW, _sourceH);
                _autoAnalysis   = analysis;
                _autoScene      = analysis.Scene;
                _autoBaseParams = ImageProcessor.ComputeAutoParams(analysis);
                if (_autoNNWeights.HasValue)
                {
                    _autoBaseParams = ImageProcessor.RefineWithNN(
                        _autoBaseParams, analysis, _autoNNWeights.Value);
                    _autoScene = _autoNNWeights.Value.Dominant;
                }
            }

            _autoEnhanceOn = true;
            AutoBtn.Content    = "⚡ On";
            AutoBtn.Background = new SolidColorBrush(Color.FromRgb(0x27, 0xB8, 0x4A));

            AutoSubLabel.Text = SceneLabel(_autoScene, "✓");
            AutoIntensityPanel.Visibility = Visibility.Visible;

            AutoIntensitySlider.Value = 0;
            AutoIntensityLabel.Text   = "Default";
            ApplyAutoAtSliderValue(0f);

            // TODO: re-enable once FiveK-trained models (expert_a/c/e.onnx) are ready.
            // _ = RunNeuralEnhancerAsync();
        }
        else
        {
            // ── Turning auto OFF — restore pre-auto state ──
            // Capture local ref BEFORE TurnOffAutoEnhanceUI nulls _autoPreState
            var preState = _autoPreState;
            TurnOffAutoEnhanceUI();
            // Restore the exact sliders and adj that existed before auto was turned on
            if (preState != null)
            {
                ApplyAdjToSliders(preState);
                _adj.Exposure   = preState.Exposure;
                _adj.Brilliance = preState.Brilliance;
                _adj.Highlights = preState.Highlights;
                _adj.Shadows    = preState.Shadows;
                _adj.Contrast   = preState.Contrast;
                _adj.Brightness = preState.Brightness;
                _adj.BlackPoint = preState.BlackPoint;
                _adj.Saturation = preState.Saturation;
                _adj.Vibrance   = preState.Vibrance;
                _adj.Warmth     = preState.Warmth;
                _adj.Tint       = preState.Tint;
                _adj.Sharpness  = preState.Sharpness;
                _adj.Definition = preState.Definition;
                _adj.Noise      = preState.Noise;
                _adj.Vignette   = preState.Vignette;
            }
            ScheduleRender();
        }
    }

    /// <summary>
    /// Only reset the UI chrome — don't touch sliders or adj values.
    /// </summary>
    private void TurnOffAutoEnhanceUI()
    {
        _autoEnhanceOn = false;
        _autoBaseParams = null;
        _autoPreState   = null;
        _autoAnalysis   = null;
        _autoScene      = SceneType.General;
        AutoBtn.Content    = "⚡ Auto";
        AutoBtn.Background = new SolidColorBrush(Color.FromRgb(0x0A, 0x84, 0xFF));
        AutoSubLabel.Text  = "Smart analysis of your photo";
        AutoIntensityPanel.Visibility = Visibility.Collapsed;
        AutoIntensitySlider.Value     = 0;
        AutoIntensityLabel.Text       = "Default";
    }

    private static string SceneLabel(SceneType scene, string suffix) => scene switch
    {
        SceneType.Portrait  => $"Portrait · {suffix}",
        SceneType.Landscape => $"Landscape · {suffix}",
        SceneType.LowLight  => $"Low-light · {suffix}",
        SceneType.Night     => $"Night · {suffix}",
        SceneType.HDR       => $"HDR · {suffix}",
        SceneType.Sunset    => $"Sunset · {suffix}",
        SceneType.HighKey   => $"High-key · {suffix}",
        SceneType.Daylight  => $"Daylight · {suffix}",
        _                   => $"Scene · {suffix}",
    };

    private async Task RunNeuralEnhancerAsync()
    {
        if (_sourcePixels == null || _autoAnalysis == null) return;

        var pixels   = _sourcePixels;
        var w        = _sourceW;
        var h        = _sourceH;
        var analysis = _autoAnalysis;
        var enhancer = _neuralEnhancer;

        if (enhancer?.HasParamModel == true)
        {
            var directParams = await Task.Run(() => enhancer.PredictParams(pixels, w, h));
            if (directParams == null)
            {
                await Dispatcher.InvokeAsync(() =>
                {
                    if (!_autoEnhanceOn) return;
                    AutoSubLabel.Text = SceneLabel(_autoScene, "✓");
                    ApplyAutoAtSliderValue((float)AutoIntensitySlider.Value);
                });
                return;
            }
            await Dispatcher.InvokeAsync(() =>
            {
                if (!_autoEnhanceOn) return;
                if (_autoBaseParams != null)
                {
                    // PPR10K is portrait-only, so the NN is only reliable for Portrait scenes.
                    // For all other scene types the NN extrapolates outside its training
                    // distribution and produces extreme values — skip the blend entirely.
                    if (_autoScene == SceneType.Portrait)
                    {
                        const float nn = 0.40f, rb = 0.60f;
                        _autoBaseParams.Exposure    = rb * _autoBaseParams.Exposure    + nn * directParams.Exposure;
                        _autoBaseParams.Brilliance  = rb * _autoBaseParams.Brilliance  + nn * directParams.Brilliance;
                        _autoBaseParams.Highlights  = rb * _autoBaseParams.Highlights  + nn * directParams.Highlights;
                        _autoBaseParams.Shadows     = rb * _autoBaseParams.Shadows     + nn * directParams.Shadows;
                        _autoBaseParams.Contrast    = rb * _autoBaseParams.Contrast    + nn * directParams.Contrast;
                        _autoBaseParams.Brightness  = rb * _autoBaseParams.Brightness  + nn * directParams.Brightness;
                        _autoBaseParams.BlackPoint  = rb * _autoBaseParams.BlackPoint  + nn * directParams.BlackPoint;
                        _autoBaseParams.Saturation  = rb * _autoBaseParams.Saturation  + nn * directParams.Saturation;
                        _autoBaseParams.Vibrance    = rb * _autoBaseParams.Vibrance    + nn * directParams.Vibrance;
                        _autoBaseParams.Warmth      = rb * _autoBaseParams.Warmth      + nn * directParams.Warmth;
                        _autoBaseParams.Tint        = rb * _autoBaseParams.Tint        + nn * directParams.Tint;
                        _autoBaseParams.Sharpness   = Math.Max(directParams.Sharpness,  _autoBaseParams.Sharpness);
                        _autoBaseParams.Definition  = Math.Max(directParams.Definition, _autoBaseParams.Definition);
                        _autoBaseParams.Noise       = Math.Max(directParams.Noise,      _autoBaseParams.Noise);
                    }
                    // Non-portrait: NN output ignored — rule-based params kept as-is.
                }
                AutoSubLabel.Text = SceneLabel(_autoScene, "AI ✓");
                ApplyAutoAtSliderValue((float)AutoIntensitySlider.Value);
            });
            return;
        }

        var weights = await Task.Run(() => enhancer?.Analyze(pixels, w, h));

        // No model files available — fall back gracefully to rule-based result
        if (weights == null)
        {
            await Dispatcher.InvokeAsync(() =>
            {
                if (!_autoEnhanceOn) return;
                AutoSubLabel.Text = SceneLabel(_autoScene, "✓");
                ApplyAutoAtSliderValue((float)AutoIntensitySlider.Value);
            });
            return;
        }

        _autoNNWeights = weights;
        await Dispatcher.InvokeAsync(() =>
        {
            if (!_autoEnhanceOn || _autoBaseParams == null || _autoAnalysis == null) return;
            _autoBaseParams = ImageProcessor.RefineWithNN(_autoBaseParams, _autoAnalysis, weights.Value);
            _autoScene = weights.Value.Dominant;
            AutoSubLabel.Text = SceneLabel(_autoScene, "AI ✓");
            ApplyAutoAtSliderValue((float)AutoIntensitySlider.Value);
        });
    }

    private void AutoIntensity_Changed(object sender, RoutedPropertyChangedEventArgs<double> e)
    {
        if (!_autoEnhanceOn || _autoBaseParams == null) return;
        float v = (float)e.NewValue;
        if (AutoIntensityLabel != null)
        {
            if (v < -90)       AutoIntensityLabel.Text = "Original";
            else if (v == 0)   AutoIntensityLabel.Text = "Default";
            else               AutoIntensityLabel.Text = (v > 0 ? "+" : "") + v.ToString("0.0");
        }
        ApplyAutoAtSliderValue(v);
    }

    /// <summary>
    /// Map the intensity slider (-100 … +100) to a blend and apply.
    ///
    /// Slider = 0   → t = 1.0  → full auto enhancement (the "Default" look)
    /// Slider = +100 → t = 1.7  → 70 % stronger than auto
    /// Slider = -100 → t = 0.0  → exactly the pre-auto state (original)
    ///
    /// The blend for each param is:
    ///   value = lerp(preState, autoTarget, t)
    ///
    /// This means at t=0 the image looks exactly as it did before auto was
    /// toggled on, and at t=1 it looks exactly as auto computed, just like
    /// iPhone's slider behaviour.
    /// </summary>
    private void ApplyAutoAtSliderValue(float sliderValue)
    {
        if (_autoBaseParams == null || _autoPreState == null) return;

        // Map slider -100…+100 → blend factor 0…1.7
        //   -100 → 0.0 (original / pre-auto)
        //      0 → 1.0 (full auto)
        //   +100 → 1.7 (pushed)
        float t = sliderValue < 0
            ? (sliderValue + 100f) / 100f          // -100→0, 0→1
            : 1f + (sliderValue / 100f) * 0.7f;    //   0→1, +100→1.7

        var pre = _autoPreState;
        var tgt = _autoBaseParams;

        // Interpolate every parameter from pre-auto → auto-target
        _adj.Exposure   = Lerp(pre.Exposure,   tgt.Exposure,   t);
        _adj.Brilliance = Lerp(pre.Brilliance, tgt.Brilliance, t);
        _adj.Highlights = Lerp(pre.Highlights, tgt.Highlights, t);
        _adj.Shadows    = Lerp(pre.Shadows,    tgt.Shadows,    t);
        _adj.Contrast   = Lerp(pre.Contrast,   tgt.Contrast,   t);
        _adj.Brightness = Lerp(pre.Brightness, tgt.Brightness, t);
        _adj.BlackPoint = Lerp(pre.BlackPoint, tgt.BlackPoint, t);
        _adj.Saturation = Lerp(pre.Saturation, tgt.Saturation, t);
        _adj.Vibrance   = Lerp(pre.Vibrance,   tgt.Vibrance,   t);
        _adj.Warmth     = Lerp(pre.Warmth,     tgt.Warmth,     t);
        _adj.Tint       = Lerp(pre.Tint,       tgt.Tint,       t);
        _adj.Sharpness  = Lerp(pre.Sharpness,  tgt.Sharpness,  t);
        _adj.Definition = Lerp(pre.Definition, tgt.Definition, t);
        _adj.Noise      = Lerp(pre.Noise,      tgt.Noise,      t);
        _adj.Vignette   = Lerp(pre.Vignette,   tgt.Vignette,   t);

        // Clamp to valid slider ranges
        _adj.Exposure   = Math.Clamp(_adj.Exposure,   -100, 100);
        _adj.Brilliance = Math.Clamp(_adj.Brilliance, -100, 100);
        _adj.Highlights = Math.Clamp(_adj.Highlights, -100, 100);
        _adj.Shadows    = Math.Clamp(_adj.Shadows,    -100, 100);
        _adj.Contrast   = Math.Clamp(_adj.Contrast,   -100, 100);
        _adj.Brightness = Math.Clamp(_adj.Brightness, -100, 100);
        _adj.BlackPoint = Math.Clamp(_adj.BlackPoint, -100, 100);
        _adj.Saturation = Math.Clamp(_adj.Saturation, -100, 100);
        _adj.Vibrance   = Math.Clamp(_adj.Vibrance,   -100, 100);
        _adj.Warmth     = Math.Clamp(_adj.Warmth,     -100, 100);
        _adj.Tint       = Math.Clamp(_adj.Tint,       -100, 100);
        _adj.Sharpness  = Math.Clamp(_adj.Sharpness,     0, 100);
        _adj.Definition = Math.Clamp(_adj.Definition,    0, 100);
        _adj.Noise      = Math.Clamp(_adj.Noise,         0, 100);
        _adj.Vignette   = Math.Clamp(_adj.Vignette,      0, 100);

        ApplyAdjToSliders(_adj);
        ScheduleRender();
    }

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    private static float Lerp(float from, float to, float t)
        => from + (to - from) * t;

    // ── Filters ──
    private void BuildFilterGrid()
    {
        FilterGrid.Children.Clear();
        foreach (var (type, name, c1, c2) in Filters)
        {
            var outerBorder = new Border
            {
                CornerRadius    = new CornerRadius(10),
                BorderThickness = new Thickness(1.5),
                BorderBrush     = new SolidColorBrush(Color.FromArgb(0x22, 0xFF, 0xFF, 0xFF)),
                ClipToBounds    = true,
                Margin          = new Thickness(0, 0, 0, 8),
                Height          = 80,
                Cursor          = Cursors.Hand,
                Tag             = type,
                Background      = new LinearGradientBrush(c1, c2, 45)
            };
            var grid = new Grid();
            var label = new Border
            {
                VerticalAlignment   = VerticalAlignment.Bottom,
                HorizontalAlignment = HorizontalAlignment.Left,
                Background          = new SolidColorBrush(Color.FromArgb(0x88, 0, 0, 0)),
                CornerRadius        = new CornerRadius(5),
                Margin              = new Thickness(8, 0, 0, 8),
                Padding             = new Thickness(6, 2, 6, 2),
                Child               = new TextBlock
                {
                    Text = name, Foreground = Brushes.White,
                    FontSize = 12, FontWeight = FontWeights.Medium
                }
            };
            grid.Children.Add(label);
            outerBorder.Child = grid;
            outerBorder.MouseDown += FilterCard_MouseDown;
            FilterGrid.Children.Add(outerBorder);
        }
        HighlightFilter(FilterType.None);
    }

    private void FilterCard_MouseDown(object sender, MouseButtonEventArgs e)
    {
        if (!_imageLoaded) return;
        if (sender is not Border border || border.Tag is not FilterType ft) return;
        PushHistory();
        bool wasActive = _adj.Filter == ft;
        if (wasActive && ft != FilterType.None)
        {
            _adj.Filter = FilterType.None;
            _adj.FilterIntensity = 100;
            FilterIntensitySlider.Value     = 100;
            FilterIntensityPanel.Visibility = Visibility.Collapsed;
            HighlightFilter(FilterType.None);
        }
        else
        {
            _adj.Filter = ft;
            HighlightFilter(ft);
            FilterIntensityPanel.Visibility = ft != FilterType.None
                ? Visibility.Visible : Visibility.Collapsed;
            if (ft == FilterType.None) _adj.FilterIntensity = 100;
        }
        ScheduleRender();
    }

    private void HighlightFilter(FilterType active)
    {
        foreach (var child in FilterGrid.Children)
            if (child is Border b && b.Tag is FilterType ft)
                b.BorderBrush = ft == active
                    ? new SolidColorBrush(Color.FromRgb(0x0A, 0x84, 0xFF))
                    : new SolidColorBrush(Color.FromArgb(0x22, 0xFF, 0xFF, 0xFF));
    }

    private void FilterIntensity_Changed(object sender, RoutedPropertyChangedEventArgs<double> e)
    {
        _adj.FilterIntensity = (float)e.NewValue;
        if (FilterIntensityLabel != null)
            FilterIntensityLabel.Text = $"{e.NewValue:0.0}%";
        if (_imageLoaded) ScheduleRender();
    }

    // ── Rotation sliders (non-destructive Z/X/Y axis) ──
    private void Rotation_Changed(object sender, double value)
    {
        if (!_imageLoaded) return;
        SyncAdjFromSliders();
        ScheduleRender();
    }

    // ── Transform ──
    // Bake the pending crop into _sourcePixels so rotate/flip operate on the cropped image.
    private void FlushPendingCrop()
    {
        if (!_hasPendingCrop || _sourcePixels == null) return;
        (_sourcePixels, _sourceW, _sourceH) = ImageProcessor.Crop(
            _sourcePixels, _sourceW, _sourceH, _pendCropX, _pendCropY, _pendCropW, _pendCropH);
        _hasPendingCrop = false;
        FileMetaText.Text = $"{_fileName}  ·  {_sourceW} × {_sourceH}";
    }

    private void CommitRenderedToSource()
    {
        if (_sourcePixels == null) return;
        FlushPendingCrop();
        var rendered = ImageProcessor.Render(_sourcePixels, _sourceW, _sourceH, _adj);
        int stride = _sourceW * 4;
        byte[] newPixels = new byte[_sourceH * stride];
        rendered.CopyPixels(new Int32Rect(0, 0, _sourceW, _sourceH), newPixels, stride, 0);
        _sourcePixels = newPixels;
        _adj.Reset();
        foreach (var sr in AllSliderRows()) sr.SetValueSilent(0);
        HighlightFilter(FilterType.None);
        FilterIntensityPanel.Visibility = Visibility.Collapsed;
        DrawCurves();             // curve was baked in → reset handles to identity
        SyncHslSlidersFromAdj();  // HSL was baked in → reset sliders to 0
        // Also clear any auto state since pixels changed
        TurnOffAutoEnhanceUI();
        _autoBaseParams = null;
        _autoPreState   = null;
    }

    private void RotateLeft_Click(object sender, RoutedEventArgs e)  => DoRotate(-90);
    private void RotateRight_Click(object sender, RoutedEventArgs e) => DoRotate(90);

    private void DoRotate(int deg)
    {
        if (!_imageLoaded || _sourcePixels == null) return;
        PushHistory();
        CommitRenderedToSource();
        (_sourcePixels, _sourceW, _sourceH) = ImageProcessor.Rotate(_sourcePixels, _sourceW, _sourceH, deg);
        FileMetaText.Text = $"{_fileName}  ·  {_sourceW} × {_sourceH}";
        DoRender();
    }

    private void FlipH_Click(object sender, RoutedEventArgs e) => DoFlip(true);
    private void FlipV_Click(object sender, RoutedEventArgs e) => DoFlip(false);

    private void DoFlip(bool horizontal)
    {
        if (!_imageLoaded || _sourcePixels == null) return;
        PushHistory();
        CommitRenderedToSource();
        _sourcePixels = ImageProcessor.Flip(_sourcePixels, _sourceW, _sourceH, horizontal);
        DoRender();
    }

    // ── Crop ──
    private void StartCropOverlay()
    {
        if (!_imageLoaded) return;
        _cropping = true;
        ApplyCropBtn.IsEnabled  = true;
        CancelCropBtn.IsEnabled = true;
        var imgBounds = GetImageDisplayBounds();

        if (_hasPendingCrop)
        {
            // Restore the overlay to exactly where the user last placed it
            double scaleX = imgBounds.Width  / _sourceW;
            double scaleY = imgBounds.Height / _sourceH;
            _cropRect = new Rect(
                imgBounds.X + _pendCropX * scaleX,
                imgBounds.Y + _pendCropY * scaleY,
                _pendCropW * scaleX,
                _pendCropH * scaleY);
            ClampCropToImage();
        }
        else
        {
            _cropRect = imgBounds;
            ApplyAspectConstraint();
        }

        ShowCropOverlay();
    }

    private void CommitCrop()
    {
        if (!_cropping || _sourcePixels == null) return;
        var imgBounds = GetImageDisplayBounds();

        // If crop is effectively full image, treat it as clearing the pending crop
        if (_cropRect.Width / imgBounds.Width > 0.99 && _cropRect.Height / imgBounds.Height > 0.99)
        {
            if (_hasPendingCrop) { PushHistory(); _hasPendingCrop = false; }
            CancelCrop();
            return;
        }

        PushHistory();

        double scaleX = _sourceW / imgBounds.Width;
        double scaleY = _sourceH / imgBounds.Height;
        int cx = Math.Max(0, (int)((_cropRect.X - imgBounds.X) * scaleX));
        int cy = Math.Max(0, (int)((_cropRect.Y - imgBounds.Y) * scaleY));
        int cw = Math.Max(1, (int)(_cropRect.Width  * scaleX));
        int ch = Math.Max(1, (int)(_cropRect.Height * scaleY));
        cw = Math.Min(cw, _sourceW - cx);
        ch = Math.Min(ch, _sourceH - cy);

        _pendCropX = cx; _pendCropY = cy; _pendCropW = cw; _pendCropH = ch;
        _hasPendingCrop = true;

        CancelCrop();
        // Tab_Click schedules a render right after this returns (wasOnCropTab != _isCropTabActive)
    }

    private Rect GetImageDisplayBounds()
    {
        if (PhotoDisplay.ActualWidth <= 0)
            return new Rect(0, 0, StageGrid.ActualWidth, StageGrid.ActualHeight);
        double stageW = StageGrid.ActualWidth;
        double stageH = StageGrid.ActualHeight;
        double imgRatio   = (double)_sourceW / _sourceH;
        double stageRatio = stageW / stageH;
        double renderW, renderH;
        if (imgRatio > stageRatio) { renderW = stageW; renderH = stageW / imgRatio; }
        else                       { renderH = stageH; renderW = stageH * imgRatio; }
        renderW = Math.Min(renderW, stageW);
        renderH = Math.Min(renderH, stageH);
        return new Rect((stageW - renderW) / 2, (stageH - renderH) / 2, renderW, renderH);
    }

    private void ShowCropOverlay()  { CropCanvas.Visibility = Visibility.Visible; RefreshCropOverlay(); }

    private void RefreshCropOverlay()
    {
        if (!_cropping) return;
        CropCanvas.Width  = StageGrid.ActualWidth;
        CropCanvas.Height = StageGrid.ActualHeight;
        CropDimOverlay.Width  = StageGrid.ActualWidth;
        CropDimOverlay.Height = StageGrid.ActualHeight;
        Canvas.SetLeft(CropDimOverlay, 0); Canvas.SetTop(CropDimOverlay, 0);
        double cx = _cropRect.X, cy = _cropRect.Y, cw = _cropRect.Width, ch = _cropRect.Height;
        Canvas.SetLeft(CropBox, cx); Canvas.SetTop(CropBox, cy);
        CropBox.Width = cw; CropBox.Height = ch;
        PlaceHandle(HandleNW, cx - 7,          cy - 7);
        PlaceHandle(HandleNE, cx + cw - 7,    cy - 7);
        PlaceHandle(HandleSW, cx - 7,          cy + ch - 7);
        PlaceHandle(HandleSE, cx + cw - 7,    cy + ch - 7);
        PlaceHandle(HandleN,  cx + cw / 2 - 7, cy - 7);
        PlaceHandle(HandleS,  cx + cw / 2 - 7, cy + ch - 7);
        PlaceHandle(HandleE,  cx + cw - 7,     cy + ch / 2 - 7);
        PlaceHandle(HandleW,  cx - 7,           cy + ch / 2 - 7);
    }

    private static void PlaceHandle(Border h, double x, double y)
    { Canvas.SetLeft(h, x); Canvas.SetTop(h, y); }

    private void ApplyAspectConstraint()
    {
        if (!_cropping) return;
        var sel = (AspectCombo.SelectedItem as ComboBoxItem)?.Content?.ToString() ?? "Free";
        if (sel == "Free") return;

        double ratio = sel switch
        {
            "Original"     => (double)_sourceW / _sourceH,
            "Square (1:1)" => 1.0,
            "16:9"  => 16.0 / 9,  "9:16"  => 9.0 / 16,
            "10:8"  => 10.0 / 8,  "8:10"  => 8.0 / 10,
            "7:5"   => 7.0  / 5,  "5:7"   => 5.0 / 7,
            "4:3"   => 4.0  / 3,  "3:4"   => 3.0 / 4,
            "5:4"   => 5.0  / 4,  "4:5"   => 4.0 / 5,
            "3:2"   => 3.0  / 2,  "2:3"   => 2.0 / 3,
            _       => 0
        };
        if (ratio <= 0) return;

        var b = GetImageDisplayBounds();

        // Drive from height (N/S handles) or width (all other handles / combo change)
        double newW, newH;
        if (_aspectDriveFromHeight)
        { newH = _cropRect.Height; newW = newH * ratio; }
        else
        { newW = _cropRect.Width;  newH = newW / ratio; }

        // Scale down proportionally so the rect always fits inside the image
        if (newW > b.Width || newH > b.Height)
        {
            double scale = Math.Min(b.Width / newW, b.Height / newH);
            newW *= scale;
            newH *= scale;
        }

        // Anchor the rect to the fixed corner/edge of the active handle
        // so dragging one corner never moves the opposite corner
        double x, y;
        switch (_cropHandle)
        {
            case "se": x = _cropRect.X;                                    y = _cropRect.Y;                                    break; // NW fixed
            case "nw": x = _cropRect.Right  - newW;                        y = _cropRect.Bottom - newH;                        break; // SE fixed
            case "ne": x = _cropRect.X;                                    y = _cropRect.Bottom - newH;                        break; // SW fixed
            case "sw": x = _cropRect.Right  - newW;                        y = _cropRect.Y;                                    break; // NE fixed
            case "e":  x = _cropRect.X;                                    y = _cropRect.Y + (_cropRect.Height - newH) / 2;    break; // left edge, center V
            case "w":  x = _cropRect.Right  - newW;                        y = _cropRect.Y + (_cropRect.Height - newH) / 2;    break; // right edge, center V
            case "n":  x = _cropRect.X + (_cropRect.Width  - newW) / 2;   y = _cropRect.Bottom - newH;                        break; // bottom edge, center H
            case "s":  x = _cropRect.X + (_cropRect.Width  - newW) / 2;   y = _cropRect.Y;                                    break; // top edge, center H
            default:   // ratio/orientation change from combo — keep centered
                       x = _cropRect.X + (_cropRect.Width  - newW) / 2;
                       y = _cropRect.Y + (_cropRect.Height - newH) / 2;   break;
        }

        _cropRect = new Rect(x, y, newW, newH);
        ClampCropToImage();
    }

    // Build a Rect from any two corners — handles negative drag without throwing
    private static Rect MakeRect(double x1, double y1, double x2, double y2) =>
        new Rect(Math.Min(x1, x2), Math.Min(y1, y2),
                 Math.Abs(x2 - x1), Math.Abs(y2 - y1));

    private void ClampCropToImage()
    {
        var b = GetImageDisplayBounds();
        double w = Math.Max(36, Math.Min(_cropRect.Width,  b.Width));
        double h = Math.Max(36, Math.Min(_cropRect.Height, b.Height));
        // Guard against float precision making min > max in Clamp
        double maxX = Math.Max(b.X, b.X + b.Width  - w);
        double maxY = Math.Max(b.Y, b.Y + b.Height - h);
        double x = Math.Clamp(_cropRect.X, b.X, maxX);
        double y = Math.Clamp(_cropRect.Y, b.Y, maxY);
        _cropRect = new Rect(x, y, w, h);
    }

    private void CropBox_MouseDown(object sender, MouseButtonEventArgs e)
    {
        _cropHandle = "move";
        _aspectDriveFromHeight = false;
        _cropDragStart   = e.GetPosition(CropCanvas);
        _cropDragInitial = _cropRect;
        (sender as UIElement)?.CaptureMouse();
        e.Handled = true;
    }

    private void Handle_MouseDown(object sender, MouseButtonEventArgs e)
    {
        var h = (Border)sender;
        _cropHandle = h == HandleNW ? "nw" : h == HandleNE ? "ne" :
                      h == HandleSW ? "sw" : h == HandleSE ? "se" :
                      h == HandleN  ? "n"  : h == HandleS  ? "s"  :
                      h == HandleE  ? "e"  : "w";
        _aspectDriveFromHeight = (h == HandleN || h == HandleS);
        _cropDragStart   = e.GetPosition(CropCanvas);
        _cropDragInitial = _cropRect;
        h.CaptureMouse();
        e.Handled = true;
    }

    private void Aspect_Changed(object sender, SelectionChangedEventArgs e)
    { if (!_suppressAspectChanged && _cropping) { _aspectDriveFromHeight = false; ApplyAspectConstraint(); RefreshCropOverlay(); } }

    private void CropOrient_Click(object sender, RoutedEventArgs e)
    {
        if (sender is not System.Windows.Controls.Primitives.ToggleButton clicked) return;
        _cropOrientation = (string)clicked.Tag;
        PortraitOrientBtn.IsChecked  = _cropOrientation == "Portrait";
        LandscapeOrientBtn.IsChecked = _cropOrientation == "Landscape";
        _aspectDriveFromHeight = false;
        PopulateAspectRatios();
        if (_cropping) { ApplyAspectConstraint(); RefreshCropOverlay(); }
    }

    // Rebuild the AspectCombo items so labels reflect the current orientation
    private void PopulateAspectRatios()
    {
        _suppressAspectChanged = true;
        bool portrait = _cropOrientation == "Portrait";
        string[] items = portrait
            ? new[] { "Free", "Original", "Square (1:1)", "9:16", "8:10", "5:7", "3:4", "4:5", "2:3" }
            : new[] { "Free", "Original", "Square (1:1)", "16:9", "10:8", "7:5",  "4:3", "5:4", "3:2" };
        AspectCombo.Items.Clear();
        foreach (var s in items)
            AspectCombo.Items.Add(new ComboBoxItem { Content = s });
        AspectCombo.SelectedIndex = 0;
        _suppressAspectChanged = false;
    }

    private void CropCanvas_MouseMove(object o, MouseEventArgs e)
    {
        if (_cropHandle == "" || !_cropping) return;
        var pos = e.GetPosition(CropCanvas);
        double dx = pos.X - _cropDragStart.X, dy = pos.Y - _cropDragStart.Y;
        var r = _cropDragInitial;
        // Use MakeRect for handle drags so dragging past the opposite corner
        // never produces a negative-dimension Rect (which would throw).
        _cropRect = _cropHandle switch
        {
            "move" => new Rect(r.X + dx, r.Y + dy, r.Width, r.Height),
            "nw"   => MakeRect(r.X + dx,            r.Y + dy,            r.X + r.Width,      r.Y + r.Height),
            "ne"   => MakeRect(r.X,                  r.Y + dy,            r.X + r.Width + dx, r.Y + r.Height),
            "sw"   => MakeRect(r.X + dx,             r.Y,                 r.X + r.Width,      r.Y + r.Height + dy),
            "se"   => MakeRect(r.X,                  r.Y,                 r.X + r.Width + dx, r.Y + r.Height + dy),
            "n"    => MakeRect(r.X,                  r.Y + dy,            r.X + r.Width,      r.Y + r.Height),
            "s"    => MakeRect(r.X,                  r.Y,                 r.X + r.Width,      r.Y + r.Height + dy),
            "e"    => MakeRect(r.X,                  r.Y,                 r.X + r.Width + dx, r.Y + r.Height),
            "w"    => MakeRect(r.X + dx,             r.Y,                 r.X + r.Width,      r.Y + r.Height),
            _      => _cropRect
        };
        // Enforce minimum size
        _cropRect = new Rect(_cropRect.X, _cropRect.Y,
            Math.Max(36, _cropRect.Width), Math.Max(36, _cropRect.Height));
        ApplyAspectConstraint();
        ClampCropToImage();
        RefreshCropOverlay();
    }

    private void CropCanvas_MouseUp(object o, MouseButtonEventArgs e)
    { _cropHandle = ""; Mouse.Capture(null); }

    private void ApplyCrop_Click(object sender, RoutedEventArgs e)
    {
        if (!_cropping || _sourcePixels == null) return;

        var imgBounds = GetImageDisplayBounds();

        // If the selection covers the whole image, treat as no-op
        if (_cropRect.Width / imgBounds.Width > 0.99 && _cropRect.Height / imgBounds.Height > 0.99)
        {
            CancelCrop();
            return;
        }

        // Compute pixel-space crop rect
        double scaleX = _sourceW / imgBounds.Width;
        double scaleY = _sourceH / imgBounds.Height;
        int cx = Math.Max(0, (int)((_cropRect.X - imgBounds.X) * scaleX));
        int cy = Math.Max(0, (int)((_cropRect.Y - imgBounds.Y) * scaleY));
        int cw = Math.Max(1, (int)(_cropRect.Width  * scaleX));
        int ch = Math.Max(1, (int)(_cropRect.Height * scaleY));
        cw = Math.Min(cw, _sourceW - cx);
        ch = Math.Min(ch, _sourceH - cy);

        // Save pre-crop state for dedicated Undo Crop button
        _preCropSnapshot = new HistorySnapshot(
            (byte[])_sourcePixels.Clone(), _sourceW, _sourceH,
            _adj.Clone(), CaptureMarkupStrokes());

        // Also push to global history so Ctrl+Z works normally
        PushHistory();

        // Bake the crop into source pixels immediately (destructive)
        (_sourcePixels, _sourceW, _sourceH) = ImageProcessor.Crop(
            _sourcePixels, _sourceW, _sourceH, cx, cy, cw, ch);
        _hasPendingCrop = false;
        FileMetaText.Text = $"{_fileName}  ·  {_sourceW} × {_sourceH}";

        CancelCrop();
        UndoCropBtn.IsEnabled = true;
        DoRender();
    }

    private void UndoCrop_Click(object sender, RoutedEventArgs e)
    {
        if (_preCropSnapshot == null) return;
        var snap = _preCropSnapshot;
        _preCropSnapshot = null;

        _sourcePixels = (byte[])snap.Pixels.Clone();
        _sourceW = snap.W; _sourceH = snap.H;
        _hasPendingCrop = false;
        SetAdjFromSnapshot(snap.Adj);
        ApplyAdjToSliders(snap.Adj);
        HighlightFilter(snap.Adj.Filter);
        FilterIntensitySlider.Value = snap.Adj.FilterIntensity;
        FilterIntensityPanel.Visibility = snap.Adj.Filter != FilterType.None
            ? Visibility.Visible : Visibility.Collapsed;
        RestoreMarkupStrokes(snap.Strokes);
        DrawCurves();
        SyncHslSlidersFromAdj();
        FileMetaText.Text = $"{_fileName}  ·  {_sourceW} × {_sourceH}";

        if (_cropping) CancelCrop();
        UndoCropBtn.IsEnabled = false;
        DoRender();

        // Re-start the crop overlay if still on the Crop tab
        if (_isCropTabActive && _imageLoaded)
            Dispatcher.BeginInvoke(System.Windows.Threading.DispatcherPriority.Render,
                new Action(StartCropOverlay));
    }

    private void CancelCrop_Click(object sender, RoutedEventArgs e)
    {
        if (_hasPendingCrop) PushHistory();
        _hasPendingCrop = false;
        CancelCrop();
        ScheduleRender();
    }

    private void CancelCrop()
    {
        _cropping = false;
        _cropHandle = "";
        CropCanvas.Visibility   = Visibility.Collapsed;
        ApplyCropBtn.IsEnabled  = false;
        CancelCropBtn.IsEnabled = false;
    }

    protected override void OnMouseDown(MouseButtonEventArgs e)
    {
        base.OnMouseDown(e);
        // Double-click toggles between fit-to-window and 100% (actual-pixel) zoom
        if (e.ClickCount == 2 && _imageLoaded && !_cropping && !_drawing)
        {
            double curScale = Math.Sqrt(_zoomMatrix.M11 * _zoomMatrix.M11 + _zoomMatrix.M12 * _zoomMatrix.M12);
            if (curScale > 1.01) ResetZoom();
            else                 ZoomToActualSize(e.GetPosition(ImageZoomGroup));
        }
    }

    protected override void OnMouseMove(MouseEventArgs e)
    {
        base.OnMouseMove(e);
        if (_cropping) CropCanvas_MouseMove(this, e);
        if (_drawing || _draggedStroke != null) Markup_MouseMove(this, e);
        UpdateFilmstripToggleFade(e.GetPosition(StageGrid).Y);
    }

    protected override void OnMouseUp(MouseButtonEventArgs e)
    {
        base.OnMouseUp(e);
        if (_cropping) CropCanvas_MouseUp(this, e);
        if (_drawing || _draggedStroke != null) Markup_MouseUp(this, e);
    }

    // ── Markup ──
    private void BuildSwatches()
    {
        var colors = new[] { "#FF453A","#FF9F0A","#FFD60A","#30D158","#0A84FF","#BF5AF2","#FFFFFF","#636366","#1C1C1E" };
        foreach (var hex in colors)
        {
            var c = (Color)ColorConverter.ConvertFromString(hex);
            var btn = new Border
            {
                Width = 30, Height = 30, CornerRadius = new CornerRadius(7),
                Background = new SolidColorBrush(c),
                BorderBrush = new SolidColorBrush(Color.FromArgb(0x40, 0xFF, 0xFF, 0xFF)),
                BorderThickness = new Thickness(2), Margin = new Thickness(0, 0, 6, 6),
                Cursor = Cursors.Hand, Tag = c
            };
            btn.MouseDown += Swatch_Click;
            SwatchPanel.Children.Add(btn);
        }
        ((Border)SwatchPanel.Children[0]).BorderBrush = new SolidColorBrush(Color.FromRgb(0xFF, 0xD6, 0x0A));
    }

    private void Swatch_Click(object sender, MouseButtonEventArgs e)
    {
        if (sender is not Border b || b.Tag is not Color c) return;
        _markupColor = c;
        foreach (Border s in SwatchPanel.Children)
            s.BorderBrush = new SolidColorBrush(Color.FromArgb(0x40, 0xFF, 0xFF, 0xFF));
        b.BorderBrush = new SolidColorBrush(Color.FromRgb(0xFF, 0xD6, 0x0A));
    }

    private void MarkupTool_Click(object sender, RoutedEventArgs e)
    {
        if (sender is not ToggleButton tb) return;
        var tools = new[] { PenTool, BrushTool, EraserTool, LineTool, RectTool, ArrowTool, RulerTool, TextTool };

        // Re-tapping the active tool deselects → hand/move mode
        if (tb.IsChecked == false)
        {
            CommitTextBox();
            foreach (var t in tools) t.IsChecked = false;
            _markupTool = "hand";
            MarkupCanvas.Cursor = Cursors.Hand;
            return;
        }

        CommitTextBox();
        foreach (var t in tools) t.IsChecked = t == tb;
        _markupTool = (string)tb.Tag;
        MarkupCanvas.Cursor = Cursors.Cross;
    }

    private void BrushSize_Changed(object sender, RoutedPropertyChangedEventArgs<double> e)
    {
        _brushSize = (float)e.NewValue;
        if (BrushLabel != null) BrushLabel.Text = e.NewValue.ToString("0.0");
    }

    private void BuildFontCombo()
    {
        var fonts = new[]
        {
            "Arial", "Arial Black", "Calibri", "Cambria", "Comic Sans MS",
            "Courier New", "Georgia", "Impact", "Segoe UI", "Tahoma",
            "Times New Roman", "Trebuchet MS", "Verdana"
        };
        foreach (var f in fonts)
            FontFamilyCombo.Items.Add(new ComboBoxItem
            {
                Content    = f,
                FontFamily = new FontFamily(f),
            });
        FontFamilyCombo.SelectedIndex = 0;
    }

    private void FontFamily_Changed(object sender, SelectionChangedEventArgs e)
    {
        if (FontFamilyCombo.SelectedItem is ComboBoxItem item)
            _fontFamily = item.Content.ToString() ?? "Arial";
    }

    private void FontSize_Changed(object sender, RoutedPropertyChangedEventArgs<double> e)
    {
        _fontSize = (float)e.NewValue;
        if (FontSizeLabel != null) FontSizeLabel.Text = ((int)e.NewValue).ToString();
    }

    private void TextBox_KeyDown(object sender, KeyEventArgs e)
    {
        if (e.Key == Key.Return)
        {
            if (Keyboard.Modifiers.HasFlag(ModifierKeys.Control))
            {
                CommitTextBox();
                e.Handled = true;
            }
            // Enter and Shift+Enter both insert newlines (AcceptsReturn=true handles it)
        }
        else if (e.Key == Key.Escape)
        {
            if (_activeTextBox != null)
            {
                MarkupCanvas.Children.Remove(_activeTextBox);
                _activeTextBox = null;
                if (_history.Count > 0) _history.Pop();
            }
            e.Handled = true;
        }
    }

    private void TextBox_LostFocus(object sender, RoutedEventArgs e) => CommitTextBox();

    private void CommitTextBox()
    {
        if (_activeTextBox == null) return;
        var tb   = _activeTextBox;
        _activeTextBox = null;
        tb.KeyDown   -= TextBox_KeyDown;
        tb.LostFocus -= TextBox_LostFocus;

        if (string.IsNullOrWhiteSpace(tb.Text))
        {
            MarkupCanvas.Children.Remove(tb);
            if (_history.Count > 0) _history.Pop();
            return;
        }

        // Replace the editable TextBox with a frozen TextBlock
        double left = Canvas.GetLeft(tb);
        double top  = Canvas.GetTop(tb);
        MarkupCanvas.Children.Remove(tb);

        var label = new TextBlock
        {
            Text             = tb.Text,
            Foreground       = tb.Foreground,
            FontSize         = tb.FontSize,
            FontWeight       = tb.FontWeight,
            FontFamily       = tb.FontFamily,
            TextWrapping     = TextWrapping.Wrap,
            IsHitTestVisible = false,
            Effect = new System.Windows.Media.Effects.DropShadowEffect
                { Color = Colors.Black, BlurRadius = 4, ShadowDepth = 1, Opacity = 0.7 }
        };
        Canvas.SetLeft(label, left);
        Canvas.SetTop(label,  top);
        MarkupCanvas.Children.Add(label);
        _markupStrokes.Add(label);
        UndoMarkupBtn.IsEnabled = true;
    }

    private void PhotoDisplay_MouseDown(object sender, MouseButtonEventArgs e) { }

    private void Markup_MouseDown(object sender, MouseButtonEventArgs e)
    {
        if (!_imageLoaded || MarkupCanvas.Visibility != Visibility.Visible) return;
        _drawing   = true;
        _drawStart = e.GetPosition(MarkupCanvas);

        // Text: commit any existing text box first, then start a new one
        if (_markupTool == "text")
        {
            _drawing = false;
            CommitTextBox();
            PushHistory();
            var pos = e.GetPosition(MarkupCanvas);
            _activeTextBox = new TextBox
            {
                Background       = Brushes.Transparent,
                BorderThickness  = new Thickness(1),
                BorderBrush      = new SolidColorBrush(Color.FromArgb(140, 255, 255, 255)),
                Foreground       = new SolidColorBrush(_markupColor),
                CaretBrush       = new SolidColorBrush(_markupColor),
                FontSize         = _fontSize,
                FontWeight       = FontWeights.SemiBold,
                FontFamily       = new FontFamily(_fontFamily),
                MinWidth         = 60,
                AcceptsReturn    = true,
                TextWrapping     = TextWrapping.Wrap,
                IsHitTestVisible = true,
            };
            Canvas.SetLeft(_activeTextBox, pos.X);
            Canvas.SetTop(_activeTextBox,  pos.Y);
            _activeTextBox.KeyDown      += TextBox_KeyDown;
            _activeTextBox.LostFocus    += TextBox_LostFocus;
            MarkupCanvas.Children.Add(_activeTextBox);
            _activeTextBox.Focus();
            e.Handled = true;
            return;
        }

        // Hand/move: short tap → edit text  |  long press / drag → move stroke
        if (_markupTool == "hand")
        {
            _drawing = false;
            _draggedStroke = _markupStrokes.LastOrDefault(s => StrokeHitsPoint(s, _drawStart, 14f));
            if (_draggedStroke != null)
            {
                _dragAnchor    = _drawStart;
                _handMouseStart = _drawStart;
                _handDragging  = false;
                _longPressTimer.Start();
                MarkupCanvas.CaptureMouse();
            }
            e.Handled = true;
            return;
        }

        // Eraser: no history push, no new stroke — just start erasing
        if (_markupTool == "eraser")
        {
            float r = _brushSize * 1.5f;
            _eraserCursor = new Ellipse
            {
                Width  = r * 2, Height = r * 2,
                Stroke = new SolidColorBrush(Color.FromArgb(200, 255, 255, 255)),
                StrokeThickness = 1.5,
                StrokeDashArray = new DoubleCollection { 4, 2 },
                IsHitTestVisible = false
            };
            Canvas.SetLeft(_eraserCursor, _drawStart.X - r);
            Canvas.SetTop(_eraserCursor,  _drawStart.Y - r);
            MarkupCanvas.Children.Add(_eraserCursor);
            EraseAtPoint(_drawStart);
            MarkupCanvas.CaptureMouse();
            e.Handled = true;
            return;
        }

        PushHistory();
        var brush = new SolidColorBrush(_markupColor);
        switch (_markupTool)
        {
            case "pen":
                _currentPoly = new Polyline
                {
                    Stroke = brush, StrokeThickness = _brushSize,
                    StrokeLineJoin = PenLineJoin.Round,
                    StrokeStartLineCap = PenLineCap.Round,
                    StrokeEndLineCap = PenLineCap.Round
                };
                _currentPoly.Points.Add(_drawStart);
                MarkupCanvas.Children.Add(_currentPoly);
                break;
            case "brush":
                // Semi-transparent for layered paint feel
                var brushColor = Color.FromArgb(178, _markupColor.R, _markupColor.G, _markupColor.B);
                _currentPoly = new Polyline
                {
                    Stroke = new SolidColorBrush(brushColor),
                    StrokeThickness = _brushSize * 1.6,
                    StrokeLineJoin = PenLineJoin.Round,
                    StrokeStartLineCap = PenLineCap.Round,
                    StrokeEndLineCap = PenLineCap.Round
                };
                _currentPoly.Points.Add(_drawStart);
                MarkupCanvas.Children.Add(_currentPoly);
                break;
            case "line":
            case "arrow":
                _currentShape = new Line
                {
                    Stroke = brush, StrokeThickness = _brushSize,
                    X1 = _drawStart.X, Y1 = _drawStart.Y,
                    X2 = _drawStart.X, Y2 = _drawStart.Y
                };
                MarkupCanvas.Children.Add(_currentShape);
                break;
            case "ruler":
                _currentShape = new Line
                {
                    Stroke = brush, StrokeThickness = Math.Max(1, _brushSize * 0.6),
                    X1 = _drawStart.X, Y1 = _drawStart.Y,
                    X2 = _drawStart.X, Y2 = _drawStart.Y,
                    StrokeStartLineCap = PenLineCap.Round, StrokeEndLineCap = PenLineCap.Round
                };
                MarkupCanvas.Children.Add(_currentShape);
                _rulerLabel = new TextBlock
                {
                    Text = "0 px", Foreground = brush, FontSize = 11,
                    FontWeight = FontWeights.SemiBold,
                    Effect = new System.Windows.Media.Effects.DropShadowEffect
                        { Color = Colors.Black, BlurRadius = 4, ShadowDepth = 1, Opacity = 0.8 }
                };
                Canvas.SetLeft(_rulerLabel, _drawStart.X + 6);
                Canvas.SetTop(_rulerLabel,  _drawStart.Y - 8);
                MarkupCanvas.Children.Add(_rulerLabel);
                break;
            case "rect":
                _currentShape = new Rectangle
                { Stroke = brush, StrokeThickness = _brushSize, Fill = Brushes.Transparent };
                Canvas.SetLeft(_currentShape, _drawStart.X);
                Canvas.SetTop(_currentShape,  _drawStart.Y);
                MarkupCanvas.Children.Add(_currentShape);
                break;
        }
        MarkupCanvas.CaptureMouse();
        e.Handled = true;
    }

    private void Markup_MouseMove(object sender, MouseEventArgs e)
    {
        var pos = e.GetPosition(MarkupCanvas);

        // Eraser cursor follows mouse even when not dragging
        if (_markupTool == "eraser" && _eraserCursor != null)
        {
            float r = _brushSize * 1.5f;
            Canvas.SetLeft(_eraserCursor, pos.X - r);
            Canvas.SetTop(_eraserCursor,  pos.Y - r);
        }

        if (!_drawing && _draggedStroke == null) return;

        switch (_markupTool)
        {
            case "pen":
            case "brush":   _currentPoly?.Points.Add(pos); break;
            case "eraser":  EraseAtPoint(pos); break;
            case "line":    if (_currentShape is Line ln) { ln.X2 = pos.X; ln.Y2 = pos.Y; } break;
            case "arrow":   if (_currentShape is Line al) { al.X2 = pos.X; al.Y2 = pos.Y; } break;
            case "ruler":
                if (_currentShape is Line rl)
                {
                    rl.X2 = pos.X; rl.Y2 = pos.Y;
                    double d = Math.Sqrt(Math.Pow(pos.X - _drawStart.X, 2) + Math.Pow(pos.Y - _drawStart.Y, 2));
                    if (_rulerLabel != null)
                    {
                        _rulerLabel.Text = $"{d:0} px";
                        Canvas.SetLeft(_rulerLabel, pos.X + 6);
                        Canvas.SetTop(_rulerLabel,  pos.Y - 8);
                    }
                }
                break;
            case "rect":
                if (_currentShape is Rectangle rc)
                {
                    Canvas.SetLeft(rc, Math.Min(_drawStart.X, pos.X));
                    Canvas.SetTop(rc,  Math.Min(_drawStart.Y, pos.Y));
                    rc.Width  = Math.Abs(pos.X - _drawStart.X);
                    rc.Height = Math.Abs(pos.Y - _drawStart.Y);
                }
                break;
            case "hand":
                if (_draggedStroke == null) break;
                if (!_handDragging && Dist(pos, _handMouseStart) > 6)
                {
                    _longPressTimer.Stop();
                    _handDragging = true;
                    MarkupCanvas.Cursor = Cursors.SizeAll;
                }
                if (_handDragging)
                    _draggedStroke.RenderTransform = new TranslateTransform(
                        pos.X - _dragAnchor.X, pos.Y - _dragAnchor.Y);
                break;
        }
    }

    private void Markup_MouseUp(object sender, MouseButtonEventArgs e)
    {
        // ── Hand: separate path — doesn't use _drawing ──
        if (_markupTool == "hand" && _draggedStroke != null)
        {
            _longPressTimer.Stop();
            MarkupCanvas.ReleaseMouseCapture();
            if (_handDragging)
            {
                if (_draggedStroke.RenderTransform is TranslateTransform tx)
                    BakeTranslate(_draggedStroke, tx.X, tx.Y);
            }
            else if (_draggedStroke is TextBlock tbEdit)
            {
                // Short tap on text → reopen for editing
                OpenTextEditor(tbEdit);
            }
            _draggedStroke = null;
            _handDragging  = false;
            MarkupCanvas.Cursor = Cursors.Hand;
            return;
        }

        if (!_drawing) return;
        _drawing = false;
        MarkupCanvas.ReleaseMouseCapture();

        // ── Eraser: remove cursor ring, nothing to add ──
        if (_markupTool == "eraser")
        {
            if (_eraserCursor != null)
            {
                MarkupCanvas.Children.Remove(_eraserCursor);
                _eraserCursor = null;
            }
            UndoMarkupBtn.IsEnabled = _markupStrokes.Count > 0 && _imageLoaded;
            return;
        }

        // ── Ruler: replace preview line+label with committed ruler group ──
        if (_markupTool == "ruler" && _currentShape is Line rulerLine)
        {
            MarkupCanvas.Children.Remove(rulerLine);
            if (_rulerLabel != null) { MarkupCanvas.Children.Remove(_rulerLabel); _rulerLabel = null; }
            var col  = ((SolidColorBrush)rulerLine.Stroke).Color;
            double d = Math.Sqrt(Math.Pow(rulerLine.X2 - rulerLine.X1, 2) + Math.Pow(rulerLine.Y2 - rulerLine.Y1, 2));
            var ruler = BuildRulerCanvas(col, rulerLine.StrokeThickness,
                rulerLine.X1, rulerLine.Y1, rulerLine.X2, rulerLine.Y2, $"{d:0} px");
            MarkupCanvas.Children.Add(ruler);
            _markupStrokes.Add(ruler);
            _currentShape = null;
            UndoMarkupBtn.IsEnabled = true;
            return;
        }

        // ── Arrow: wrap shaft + head into a Canvas group ──
        if (_markupTool == "arrow" && _currentShape is Line arrow)
        {
            MarkupCanvas.Children.Remove(arrow);
            double adx = arrow.X2 - arrow.X1, ady = arrow.Y2 - arrow.Y1;
            double angle = Math.Atan2(ady, adx);
            double alen  = Math.Max(16, _brushSize * 3);
            var head = new Polygon
            {
                Fill = new SolidColorBrush(_markupColor),
                Points = new PointCollection(new[]
                {
                    new Point(arrow.X2, arrow.Y2),
                    new Point(arrow.X2 - alen * Math.Cos(angle - Math.PI/7), arrow.Y2 - alen * Math.Sin(angle - Math.PI/7)),
                    new Point(arrow.X2 - alen * Math.Cos(angle + Math.PI/7), arrow.Y2 - alen * Math.Sin(angle + Math.PI/7)),
                })
            };
            var group = new Canvas { IsHitTestVisible = false };
            group.Children.Add(arrow);
            group.Children.Add(head);
            MarkupCanvas.Children.Add(group);
            _markupStrokes.Add(group);
        }
        else
        {
            if (_currentPoly  != null) _markupStrokes.Add(_currentPoly);
            if (_currentShape != null) _markupStrokes.Add(_currentShape);
        }
        _currentPoly  = null;
        _currentShape = null;
        UndoMarkupBtn.IsEnabled = _markupStrokes.Count > 0 && _imageLoaded;
    }

    private void UndoMarkup_Click(object sender, RoutedEventArgs e)
    {
        CommitTextBox();
        if (_markupStrokes.Count == 0) return;
        MarkupCanvas.Children.Remove(_markupStrokes[^1]);
        _markupStrokes.RemoveAt(_markupStrokes.Count - 1);
        UndoMarkupBtn.IsEnabled = _markupStrokes.Count > 0;
    }

    private void ClearMarkup_Click(object sender, RoutedEventArgs e)
    {
        CommitTextBox();
        MarkupCanvas.Children.Clear();
        _markupStrokes.Clear();
        UndoMarkupBtn.IsEnabled = false;
    }

    // ── Zoom / Pan ──
    // Ctrl + scroll  → zoom centred on cursor
    // Scroll         → pan vertically (forward = up, back = down)
    // Double-click   → reset to fit

    private void Stage_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
    {
        if (!_imageLoaded || _cropping || _drawing || e.ClickCount != 1) return;
        _isPanning = true;
        _panAnchor = e.GetPosition(StageGrid);
        StageGrid.CaptureMouse();
        StageGrid.Cursor = Cursors.ScrollAll;
    }

    private void Stage_PanMove(object sender, MouseEventArgs e)
    {
        if (!_isPanning) return;
        var pos = e.GetPosition(StageGrid);
        _zoomMatrix.Translate(pos.X - _panAnchor.X, pos.Y - _panAnchor.Y);
        _panAnchor = pos;
        ImageZoomGroup.RenderTransform = new MatrixTransform(_zoomMatrix);
    }

    private void Stage_PanEnd(object sender, MouseButtonEventArgs e)
    {
        if (!_isPanning) return;
        _isPanning = false;
        StageGrid.ReleaseMouseCapture();
        StageGrid.Cursor = null;
    }

    private void FilmstripToggle_Click(object sender, MouseButtonEventArgs e)
    {
        bool show = FilmstripBar.Visibility != Visibility.Visible;
        FilmstripBar.Visibility  = show ? Visibility.Visible   : Visibility.Collapsed;
        FilmstripRowDef.Height   = show ? new GridLength(96)   : new GridLength(0);
        FilmstripToggleIcon.Text = show ? "▴" : "▾";
    }

    private void UpdateFilmstripToggleFade(double mouseYInStage)
    {
        if (!_imageLoaded) return;
        bool near = mouseYInStage >= StageGrid.ActualHeight - 90;
        FilmstripToggleBtn.Opacity          = near ? 0.92 : 0.0;
        FilmstripToggleBtn.IsHitTestVisible = near;
    }

    private void ResetZoom()
    {
        _zoomMatrix = Matrix.Identity;
        ImageZoomGroup.RenderTransform = Transform.Identity;
        UpdateStatusBar();
    }

    private void Stage_MouseWheel(object sender, MouseWheelEventArgs e)
    {
        if (!_imageLoaded) return;

        if (Keyboard.IsKeyDown(Key.LeftCtrl) || Keyboard.IsKeyDown(Key.RightCtrl))
        {
            // Zoom — scale around the cursor position in ImageZoomGroup coordinates
            double factor     = e.Delta > 0 ? 1.12 : 1.0 / 1.12;
            var    cursorPt   = e.GetPosition(ImageZoomGroup);
            var    m          = _zoomMatrix;
            m.ScaleAt(factor, factor, cursorPt.X, cursorPt.Y);

            // Clamp scale between 0.2× and 20×
            double scaleX = Math.Sqrt(m.M11 * m.M11 + m.M12 * m.M12);
            if (scaleX >= 0.2 && scaleX <= 20.0)
                _zoomMatrix = m;
        }
        else
        {
            // Pan vertically only when zoomed in — otherwise the image would
            // scroll off-screen at 1× with no scrollbar to bring it back.
            double curScale = Math.Sqrt(_zoomMatrix.M11 * _zoomMatrix.M11 + _zoomMatrix.M12 * _zoomMatrix.M12);
            if (curScale <= 1.001) return;
            double pan = e.Delta > 0 ? 80 : -80;
            _zoomMatrix.Translate(0, pan);
        }

        ImageZoomGroup.RenderTransform = new MatrixTransform(_zoomMatrix);
        UpdateStatusBar();
        e.Handled = true;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // HISTOGRAM
    // ═══════════════════════════════════════════════════════════════════════

    private System.Windows.Threading.DispatcherTimer? _histTimer;
    private CancellationTokenSource? _filmstripCts;
    private CancellationTokenSource? _loadCts;

    private void ScheduleHistogram()
    {
        if (_histTimer == null)
        {
            _histTimer = new System.Windows.Threading.DispatcherTimer
                { Interval = TimeSpan.FromMilliseconds(120) };
            _histTimer.Tick += async (_, _) =>
            {
                _histTimer.Stop();
                await DrawHistogramAsync();
            };
        }
        _histTimer.Stop();
        _histTimer.Start();
    }

    private async Task DrawHistogramAsync()
    {
        if (!_imageLoaded || _sourcePixels == null) return;

        // Render a half-resolution copy for speed
        var pixels = _sourcePixels;
        var w = _sourceW; var h = _sourceH;
        var adj = _adj.Clone();

        var (rH, gH, bH, _) = await Task.Run(() =>
        {
            var rendered = ImageProcessor.Render(pixels, w, h, adj);
            int sw = rendered.PixelWidth, sh = rendered.PixelHeight;
            int stride = sw * 4;
            byte[] px = new byte[sh * stride];
            rendered.CopyPixels(new Int32Rect(0, 0, sw, sh), px, stride, 0);
            return ImageProcessor.ComputeHistogram(px, sw, sh);
        });

        DrawHistogram(rH, gH, bH);
    }

    private void DrawHistogram(int[] rH, int[] gH, int[] bH)
    {
        const double W = 268, H = 68;
        HistogramCanvas.Children.Clear();

        // Background
        HistogramCanvas.Children.Add(new System.Windows.Shapes.Rectangle
        {
            Width = W, Height = H,
            Fill = new SolidColorBrush(Color.FromRgb(0x0D, 0x0D, 0x0F))
        });

        // Subtle grid lines
        for (int i = 1; i < 4; i++)
        {
            double x = W * i / 4;
            HistogramCanvas.Children.Add(new System.Windows.Shapes.Line
            {
                X1 = x, Y1 = 0, X2 = x, Y2 = H,
                Stroke = new SolidColorBrush(Color.FromArgb(0x18, 0xFF, 0xFF, 0xFF)),
                StrokeThickness = 1
            });
        }

        int max = 1;
        for (int i = 2; i < 254; i++)
            max = Math.Max(max, Math.Max(rH[i], Math.Max(gH[i], bH[i])));

        // Draw each channel as a filled polygon
        void DrawChannel(int[] hist, Color col)
        {
            var pts = new PointCollection();
            pts.Add(new Point(0, H));
            for (int i = 0; i < 256; i++)
            {
                double x = i * W / 255.0;
                double y = H - (hist[i] / (double)max) * (H - 2);
                pts.Add(new Point(x, y));
            }
            pts.Add(new Point(W, H));
            HistogramCanvas.Children.Add(new System.Windows.Shapes.Polygon
            {
                Points = pts,
                Fill   = new SolidColorBrush(Color.FromArgb(0x55, col.R, col.G, col.B)),
                Stroke = new SolidColorBrush(Color.FromArgb(0xCC, col.R, col.G, col.B)),
                StrokeThickness = 1
            });
        }

        DrawChannel(bH, Color.FromRgb(0x0A, 0x84, 0xFF));
        DrawChannel(gH, Color.FromRgb(0x30, 0xD1, 0x58));
        DrawChannel(rH, Color.FromRgb(0xFF, 0x45, 0x3A));

        // Bottom border
        HistogramCanvas.Children.Add(new System.Windows.Shapes.Line
        {
            X1 = 0, Y1 = H - 1, X2 = W, Y2 = H - 1,
            Stroke = new SolidColorBrush(Color.FromRgb(0x1A, 0x1A, 0x1F)),
            StrokeThickness = 1
        });
    }

    // ═══════════════════════════════════════════════════════════════════════
    // STATUS BAR
    // ═══════════════════════════════════════════════════════════════════════

    private void UpdateStatusBar()
    {
        if (!_imageLoaded) { StatusBar.Visibility = Visibility.Collapsed; return; }

        // Dimensions — show cropped size when a crop is pending
        int dw = _hasPendingCrop ? _pendCropW : _sourceW;
        int dh = _hasPendingCrop ? _pendCropH : _sourceH;
        StatusDimText.Text = $"{dw} × {dh} px";

        // Zoom — M11 is the horizontal scale factor applied to ImageZoomGroup.
        // At identity (M11=1) the Image element itself applies a Uniform stretch to fit.
        // Actual pixel zoom = M11 * fitScale.
        if (PhotoDisplay.Source is BitmapSource bs && StageGrid.ActualWidth > 0 && StageGrid.ActualHeight > 0)
        {
            double fitScale = Math.Min(StageGrid.ActualWidth  / bs.PixelWidth,
                                       StageGrid.ActualHeight / bs.PixelHeight);
            double zoom = _zoomMatrix.M11 * fitScale * 100.0;
            StatusZoomText.Text = Math.Abs(_zoomMatrix.M11 - 1.0) < 0.002
                ? "Fit"
                : $"{zoom:0}%";
        }

        StatusBar.Visibility = Visibility.Visible;
    }

    private void JpegQuality_Changed(object sender, RoutedPropertyChangedEventArgs<double> e)
    {
        _jpegQuality = (int)e.NewValue;
        if (JpegQualityLabel != null)
            JpegQualityLabel.Text = $"{_jpegQuality}";
    }

    // ═══════════════════════════════════════════════════════════════════════
    // CURVES EDITOR
    // ═══════════════════════════════════════════════════════════════════════

    private void InitCurvesCanvas()
    {
        _curveHandles = new System.Windows.Shapes.Ellipse[CurvePtCount];
        for (int i = 0; i < CurvePtCount; i++)
        {
            _curveHandles[i] = new System.Windows.Shapes.Ellipse
            {
                Width = 12, Height = 12,
                Fill   = new SolidColorBrush(Color.FromRgb(0x0A, 0x84, 0xFF)),
                Stroke = Brushes.White, StrokeThickness = 1.5,
                Tag = i, Cursor = Cursors.Hand
            };
        }
        DrawCurves();
    }

    private void DrawCurves()
    {
        if (_curveHandles == null) return;
        const double W = 236, H = 180;

        // Rebuild canvas from scratch each call (fast — only ~20 children)
        CurvesCanvas.Children.Clear();

        // Background
        CurvesCanvas.Children.Add(new System.Windows.Shapes.Rectangle
        {
            Width = W, Height = H,
            Fill = new SolidColorBrush(Color.FromRgb(0x0D, 0x0D, 0x0F)),
            IsHitTestVisible = false
        });
        // Grid lines
        for (int i = 1; i < 4; i++)
        {
            double gx = W * i / 4, gy = H * i / 4;
            CurvesCanvas.Children.Add(new System.Windows.Shapes.Line
            {
                X1 = gx, Y1 = 0, X2 = gx, Y2 = H,
                Stroke = new SolidColorBrush(Color.FromArgb(0x20, 0xFF, 0xFF, 0xFF)),
                StrokeThickness = 1, IsHitTestVisible = false
            });
            CurvesCanvas.Children.Add(new System.Windows.Shapes.Line
            {
                X1 = 0, Y1 = gy, X2 = W, Y2 = gy,
                Stroke = new SolidColorBrush(Color.FromArgb(0x20, 0xFF, 0xFF, 0xFF)),
                StrokeThickness = 1, IsHitTestVisible = false
            });
        }
        // Diagonal identity reference
        CurvesCanvas.Children.Add(new System.Windows.Shapes.Line
        {
            X1 = 0, Y1 = H, X2 = W, Y2 = 0,
            Stroke = new SolidColorBrush(Color.FromArgb(0x30, 0xFF, 0xFF, 0xFF)),
            StrokeThickness = 1, StrokeDashArray = new DoubleCollection { 4, 4 },
            IsHitTestVisible = false
        });

        // Curve polyline
        var lut = ImageProcessor.ComputeCurveLUT(_adj);
        var pts = new PointCollection(256);
        for (int i = 0; i < 256; i++)
        {
            double outVal = lut != null ? lut[i] : i;
            pts.Add(new Point(i * W / 255.0, H - outVal * H / 255.0));
        }
        CurvesCanvas.Children.Add(new System.Windows.Shapes.Polyline
        {
            Points = pts,
            Stroke = new SolidColorBrush(Color.FromRgb(0x0A, 0x84, 0xFF)),
            StrokeThickness = 2, IsHitTestVisible = false
        });

        // Control-point handles
        float[] inputs  = { 0, 64, 128, 192, 255 };
        float[] outVals = { _adj.Curve0, _adj.Curve64, _adj.Curve128, _adj.Curve192, _adj.Curve255 };
        for (int i = 0; i < CurvePtCount; i++)
        {
            double cx = inputs[i] * W / 255.0;
            double cy = H - outVals[i] * H / 255.0;
            System.Windows.Controls.Canvas.SetLeft(_curveHandles[i], cx - 6);
            System.Windows.Controls.Canvas.SetTop(_curveHandles[i],  cy - 6);
            CurvesCanvas.Children.Add(_curveHandles[i]);
        }
    }

    private void CurvesHeader_MouseDown(object sender, MouseButtonEventArgs e)
    {
        _curvesOpen = !_curvesOpen;
        CurvesPanel.Visibility  = _curvesOpen ? Visibility.Visible : Visibility.Collapsed;
        CurvesChevron.Text      = _curvesOpen ? "▾" : "▸";
        if (_curvesOpen) DrawCurves();
        e.Handled = true;
    }

    private void CurvesCanvas_MouseDown(object sender, MouseButtonEventArgs e)
    {
        if (!_imageLoaded) return;
        var pos = e.GetPosition(CurvesCanvas);
        const double W = 236, H = 180;

        // Find nearest handle within 16px
        float[] inputs = { 0, 64, 128, 192, 255 };
        float[] ys     = { _adj.Curve0, _adj.Curve64, _adj.Curve128, _adj.Curve192, _adj.Curve255 };
        double best = 16 * 16;
        _curveDragIdx = -1;
        for (int i = 0; i < CurvePtCount; i++)
        {
            double cx = inputs[i] * W / 255.0;
            double cy = H - ys[i] * H / 255.0;
            double d  = (pos.X - cx) * (pos.X - cx) + (pos.Y - cy) * (pos.Y - cy);
            if (d < best) { best = d; _curveDragIdx = i; }
        }
        if (_curveDragIdx >= 0)
        {
            PushHistory();
            CurvesCanvas.CaptureMouse();
        }
        e.Handled = true;
    }

    private void CurvesCanvas_MouseMove(object sender, MouseEventArgs e)
    {
        if (_curveDragIdx < 0) return;
        var pos = e.GetPosition(CurvesCanvas);
        const double H = 180;
        float outVal = (float)Math.Clamp((H - pos.Y) * 255.0 / H, 0, 255);
        switch (_curveDragIdx)
        {
            case 0: _adj.Curve0   = outVal; break;
            case 1: _adj.Curve64  = outVal; break;
            case 2: _adj.Curve128 = outVal; break;
            case 3: _adj.Curve192 = outVal; break;
            case 4: _adj.Curve255 = outVal; break;
        }
        DrawCurves();
        ScheduleRender();
    }

    private void CurvesCanvas_MouseUp(object sender, MouseButtonEventArgs e)
    {
        _curveDragIdx = -1;
        CurvesCanvas.ReleaseMouseCapture();
    }

    private void ResetCurve_Click(object sender, RoutedEventArgs e)
    {
        if (!_imageLoaded) return;
        PushHistory();
        _adj.Curve0 = 0; _adj.Curve64 = 64; _adj.Curve128 = 128;
        _adj.Curve192 = 192; _adj.Curve255 = 255;
        DrawCurves();
        ScheduleRender();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // HSL MIXER
    // ═══════════════════════════════════════════════════════════════════════

    private static readonly (string name, Color dot)[] HslChannelDefs =
    {
        ("R",  Color.FromRgb(0xFF, 0x45, 0x3A)),
        ("O",  Color.FromRgb(0xFF, 0x9F, 0x0A)),
        ("Y",  Color.FromRgb(0xFF, 0xD6, 0x0A)),
        ("G",  Color.FromRgb(0x30, 0xD1, 0x58)),
        ("C",  Color.FromRgb(0x5A, 0xC8, 0xFA)),
        ("B",  Color.FromRgb(0x0A, 0x84, 0xFF)),
        ("P",  Color.FromRgb(0xBF, 0x5A, 0xF2)),
    };

    private void BuildHslChannelGrid()
    {
        _hslChannelBtns = new Border[7];
        for (int i = 0; i < 7; i++)
        {
            int idx = i;
            var (name, dot) = HslChannelDefs[i];
            var btn = new Border
            {
                CornerRadius = new CornerRadius(6),
                Height = 32, Cursor = Cursors.Hand,
                BorderBrush = new SolidColorBrush(Color.FromArgb(0x40, 0xFF, 0xFF, 0xFF)),
                BorderThickness = new Thickness(1.5),
                Background = new SolidColorBrush(Color.FromArgb(0x18, dot.R, dot.G, dot.B)),
                Child = new System.Windows.Shapes.Ellipse
                {
                    Width = 10, Height = 10,
                    Fill = new SolidColorBrush(dot),
                    HorizontalAlignment = HorizontalAlignment.Center,
                    VerticalAlignment   = VerticalAlignment.Center
                }
            };
            btn.MouseDown += (_, _) => SelectHslChannel(idx);
            _hslChannelBtns[i] = btn;
            HslChannelGrid.Children.Add(btn);
        }
        SelectHslChannel(0);
    }

    private void SelectHslChannel(int ch)
    {
        _hslChannel = ch;
        for (int i = 0; i < 7; i++)
        {
            var (_, dot) = HslChannelDefs[i];
            _hslChannelBtns![i].BorderBrush = i == ch
                ? new SolidColorBrush(dot)
                : new SolidColorBrush(Color.FromArgb(0x40, 0xFF, 0xFF, 0xFF));
        }
        SyncHslSlidersFromAdj();
    }

    private void HslHeader_MouseDown(object sender, MouseButtonEventArgs e)
    {
        _hslOpen = !_hslOpen;
        HslPanel.Visibility = _hslOpen ? Visibility.Visible : Visibility.Collapsed;
        HslChevron.Text     = _hslOpen ? "▾" : "▸";
        e.Handled = true;
    }

    private void SyncHslSlidersFromAdj()
    {
        _suppressHslSliders = true;
        var (h, s) = GetHslAdjForChannel(_hslChannel);
        HslHueSlider.Value = h;
        HslSatSlider.Value = s;
        HslHueLabel.Text   = ((int)h).ToString();
        HslSatLabel.Text   = ((int)s).ToString();
        _suppressHslSliders = false;
    }

    private (float h, float s) GetHslAdjForChannel(int ch) => ch switch
    {
        0 => (_adj.HslRH, _adj.HslRS),
        1 => (_adj.HslOH, _adj.HslOS),
        2 => (_adj.HslYH, _adj.HslYS),
        3 => (_adj.HslGH, _adj.HslGS),
        4 => (_adj.HslCH, _adj.HslCS),
        5 => (_adj.HslBH, _adj.HslBS),
        _ => (_adj.HslPH, _adj.HslPS),
    };

    private void SetHslAdjForChannel(int ch, float h, float s)
    {
        switch (ch)
        {
            case 0: _adj.HslRH = h; _adj.HslRS = s; break;
            case 1: _adj.HslOH = h; _adj.HslOS = s; break;
            case 2: _adj.HslYH = h; _adj.HslYS = s; break;
            case 3: _adj.HslGH = h; _adj.HslGS = s; break;
            case 4: _adj.HslCH = h; _adj.HslCS = s; break;
            case 5: _adj.HslBH = h; _adj.HslBS = s; break;
            default: _adj.HslPH = h; _adj.HslPS = s; break;
        }
    }

    private void HslHue_Changed(object sender, RoutedPropertyChangedEventArgs<double> e)
    {
        if (_suppressHslSliders || !_imageLoaded) return;
        HslHueLabel.Text = ((int)e.NewValue).ToString();
        var (_, s) = GetHslAdjForChannel(_hslChannel);
        SetHslAdjForChannel(_hslChannel, (float)e.NewValue, s);
        ScheduleRender();
    }

    private void HslSat_Changed(object sender, RoutedPropertyChangedEventArgs<double> e)
    {
        if (_suppressHslSliders || !_imageLoaded) return;
        HslSatLabel.Text = ((int)e.NewValue).ToString();
        var (h, _) = GetHslAdjForChannel(_hslChannel);
        SetHslAdjForChannel(_hslChannel, h, (float)e.NewValue);
        ScheduleRender();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // KEYBOARD SHORTCUTS
    // ═══════════════════════════════════════════════════════════════════════

    private void Window_KeyDown(object sender, KeyEventArgs e)
    {
        bool ctrl  = Keyboard.Modifiers.HasFlag(ModifierKeys.Control);
        bool shift = Keyboard.Modifiers.HasFlag(ModifierKeys.Shift);

        // While editing a text field (slider value boxes, dialog inputs), let the
        // textbox handle its own keys (Ctrl+Z/A/C/V etc.) instead of firing shortcuts.
        if (Keyboard.FocusedElement is System.Windows.Controls.TextBox) return;

        switch (e.Key)
        {
            case Key.Z when ctrl && shift:
            case Key.Y when ctrl:
                if (RedoBtn.IsEnabled) { RedoBtn_Click(this, new RoutedEventArgs()); e.Handled = true; }
                break;

            case Key.Z when ctrl:
                if (UndoBtn.IsEnabled) { UndoBtn_Click(this, new RoutedEventArgs()); e.Handled = true; }
                break;

            case Key.S when ctrl && shift:
                if (BatchBtn.IsEnabled) BatchBtn_Click(this, new RoutedEventArgs());
                e.Handled = true;
                break;

            case Key.S when ctrl:
                if (ExportBtn.IsEnabled) ExportBtn_Click(this, new RoutedEventArgs());
                e.Handled = true;
                break;

            case Key.O when ctrl:
                OpenBtn_Click(this, new RoutedEventArgs());
                e.Handled = true;
                break;

            case Key.F11:
                ToggleFullScreen();
                e.Handled = true;
                break;

            case Key.D0 when ctrl:
            case Key.NumPad0 when ctrl:
                ResetZoom();
                e.Handled = true;
                break;

            case Key.C when ctrl && shift:
                CopyLook_Click(this, new RoutedEventArgs());
                e.Handled = true;
                break;

            case Key.V when ctrl && shift:
                ApplyCopiedLook();
                e.Handled = true;
                break;

            case Key.L when ctrl:
                if (_imageLoaded) ShowLooksDialog();
                e.Handled = true;
                break;

            case Key.R when ctrl:
                ShowRecentMenu();
                e.Handled = true;
                break;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // FULL-SCREEN MODE
    // ═══════════════════════════════════════════════════════════════════════

    private void FullScreenBtn_Click(object sender, RoutedEventArgs e) => ToggleFullScreen();

    private void ToggleFullScreen()
    {
        _fullScreen = !_fullScreen;
        if (_fullScreen)
        {
            _preFullScreenState = this.WindowState;
            this.WindowStyle    = WindowStyle.None;
            this.WindowState    = System.Windows.WindowState.Maximized;
            FullScreenBtn.Content = "⊡";
            FullScreenBtn.ToolTip = "Exit full-screen (F11)";
        }
        else
        {
            this.WindowStyle  = WindowStyle.SingleBorderWindow;
            this.WindowState  = _preFullScreenState;
            FullScreenBtn.Content = "⛶";
            FullScreenBtn.ToolTip = "Toggle full-screen (F11)";
            ApplyDarkTitleBar();
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // SPLIT VIEW (before / after side-by-side)
    // ═══════════════════════════════════════════════════════════════════════

    private void SplitViewBtn_Click(object sender, RoutedEventArgs e)
    {
        if (!_imageLoaded) return;
        _splitViewOn = !_splitViewOn;

        if (_splitViewOn)
        {
            SplitViewBtn.Content    = "⊢  Split ●";
            SplitViewBtn.Background = new SolidColorBrush(Color.FromRgb(0x0A, 0x84, 0xFF));
            SplitViewBtn.Foreground = Brushes.White;
            _compareMode = false;
            ScheduleRender(); // render triggers UpdateSplitView after rendering
        }
        else
        {
            SplitViewBtn.Content    = "⊢  Split";
            SplitViewBtn.Background = new SolidColorBrush(Color.FromRgb(0x28, 0x28, 0x2F));
            SplitViewBtn.Foreground = new SolidColorBrush(Color.FromRgb(0xF0, 0xF0, 0xF2));
            SplitCanvas.Visibility = Visibility.Collapsed;
            SplitCanvas.Children.Clear();
            ScheduleRender();
        }
    }

    private void UpdateSplitView()
    {
        if (!_splitViewOn || !_imageLoaded || _sourcePixels == null) return;

        var bounds = GetImageDisplayBounds();
        double splitX = _splitRatio * bounds.Width;  // pixels from left edge of image
        double divX   = bounds.X + splitX;

        // "After" side is already in PhotoDisplay.Source (rendered by DoRenderAsync)
        // "Before" overlay clips the original to the left portion
        var before = PixelsToImage(_sourcePixels, _sourceW, _sourceH);

        SplitCanvas.Children.Clear();
        SplitCanvas.Width  = StageGrid.ActualWidth;
        SplitCanvas.Height = StageGrid.ActualHeight;

        var beforeImg = new System.Windows.Controls.Image
        {
            Source  = before,
            Stretch = System.Windows.Media.Stretch.Uniform,
            Width   = bounds.Width,
            Height  = bounds.Height
        };
        Canvas.SetLeft(beforeImg, bounds.X);
        Canvas.SetTop(beforeImg,  bounds.Y);
        // Clip is in element-local coordinates: beforeImg origin is (0,0), not (bounds.X, bounds.Y)
        beforeImg.Clip = new RectangleGeometry(new Rect(0, 0, splitX, bounds.Height));
        SplitCanvas.Children.Add(beforeImg);

        // Divider line
        var divider = new System.Windows.Shapes.Line
        {
            X1 = divX, Y1 = bounds.Y,
            X2 = divX, Y2 = bounds.Y + bounds.Height,
            Stroke = Brushes.White, StrokeThickness = 2,
            Cursor = Cursors.SizeWE,
            Effect = new System.Windows.Media.Effects.DropShadowEffect
                { Color = Colors.Black, BlurRadius = 8, ShadowDepth = 0, Opacity = 0.6 }
        };
        SplitCanvas.Children.Add(divider);

        // Drag handle circle at mid-height on the divider
        var handleBg = new Ellipse
        {
            Width = 30, Height = 30, Fill = Brushes.White,
            Stroke = new SolidColorBrush(Color.FromRgb(0x55, 0x55, 0x6A)), StrokeThickness = 1,
            Cursor = Cursors.SizeWE,
            Effect = new System.Windows.Media.Effects.DropShadowEffect
                { Color = Colors.Black, BlurRadius = 10, ShadowDepth = 2, Opacity = 0.45 }
        };
        var handleTxt = new TextBlock
        {
            Text = "◁▷", Foreground = new SolidColorBrush(Color.FromRgb(0x44, 0x44, 0x44)),
            FontSize = 10, HorizontalAlignment = HorizontalAlignment.Center,
            VerticalAlignment = VerticalAlignment.Center, IsHitTestVisible = false
        };
        var handle = new Grid { Width = 30, Height = 30, Cursor = Cursors.SizeWE };
        handle.Children.Add(handleBg);
        handle.Children.Add(handleTxt);
        Canvas.SetLeft(handle, divX - 15);
        Canvas.SetTop(handle,  bounds.Y + bounds.Height / 2 - 15);
        SplitCanvas.Children.Add(handle);

        // Labels
        SplitCanvas.Children.Add(MakeSplitLabel("Before", bounds.X + 8, bounds.Y + 8));
        SplitCanvas.Children.Add(MakeSplitLabel("After",  divX + 8,     bounds.Y + 8));

        SplitCanvas.Visibility = Visibility.Visible;
        ScheduleHistogram();
    }

    private static UIElement MakeSplitLabel(string text, double x, double y)
    {
        var tb = new TextBlock
        {
            Text = text, Foreground = Brushes.White, FontSize = 12,
            FontWeight = FontWeights.SemiBold,
            Effect = new System.Windows.Media.Effects.DropShadowEffect
                { Color = Colors.Black, BlurRadius = 6, ShadowDepth = 1, Opacity = 0.9 }
        };
        Canvas.SetLeft(tb, x);
        Canvas.SetTop(tb, y);
        return tb;
    }

    // ── Split divider drag handlers ──
    private void SplitCanvas_MouseDown(object sender, MouseButtonEventArgs e)
    {
        if (!_splitViewOn) return;
        var bounds = GetImageDisplayBounds();
        var pos    = e.GetPosition(SplitCanvas);
        double divX = bounds.X + _splitRatio * bounds.Width;

        if (Math.Abs(pos.X - divX) <= 22 &&
            pos.Y >= bounds.Y - 5 && pos.Y <= bounds.Y + bounds.Height + 5)
        {
            _splitDragging = true;
            SplitCanvas.CaptureMouse();
            e.Handled = true;
        }
    }

    private void SplitCanvas_MouseMove(object sender, MouseEventArgs e)
    {
        if (!_splitDragging) return;
        var bounds = GetImageDisplayBounds();
        var pos    = e.GetPosition(SplitCanvas);
        _splitRatio = Math.Max(0.01, Math.Min(0.99, (pos.X - bounds.X) / bounds.Width));
        UpdateSplitView();
        e.Handled = true;
    }

    private void SplitCanvas_MouseUp(object sender, MouseButtonEventArgs e)
    {
        if (!_splitDragging) return;
        _splitDragging = false;
        SplitCanvas.ReleaseMouseCapture();
        e.Handled = true;
    }

    private void SplitCanvas_MouseLeave(object sender, MouseEventArgs e)
    {
        if (_splitDragging)
        {
            _splitDragging = false;
            SplitCanvas.ReleaseMouseCapture();
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // HAMBURGER MENU
    // ═══════════════════════════════════════════════════════════════════════

    private void HamburgerBtn_Click(object sender, RoutedEventArgs e)
    {
        _inspectorVisible = !_inspectorVisible;
        WorkspaceGrid.ColumnDefinitions[1].Width =
            _inspectorVisible ? new GridLength(292) : new GridLength(0);
        InspectorBorder.Visibility =
            _inspectorVisible ? Visibility.Visible : Visibility.Collapsed;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // EXPORT PRESETS
    // ═══════════════════════════════════════════════════════════════════════

    private void PresetsBtn_Click(object sender, RoutedEventArgs e)
    {
        if (!_imageLoaded || _sourcePixels == null) return;

        // Build a styled presets dialog
        var dlg = new Window
        {
            Width = 340, SizeToContent = SizeToContent.Height,
            WindowStyle = WindowStyle.None, AllowsTransparency = true,
            Background = Brushes.Transparent, ResizeMode = ResizeMode.NoResize,
            WindowStartupLocation = WindowStartupLocation.CenterOwner,
            Owner = this, ShowInTaskbar = false
        };

        var card = new Border
        {
            Background   = new SolidColorBrush(Color.FromRgb(0x1C, 0x1C, 0x1E)),
            CornerRadius  = new CornerRadius(14),
            Padding      = new Thickness(20, 18, 20, 18),
            Effect        = new System.Windows.Media.Effects.DropShadowEffect
                { Color = Colors.Black, Opacity = 0.75, BlurRadius = 28, ShadowDepth = 4 }
        };

        var stack = new StackPanel();
        stack.Children.Add(new TextBlock
        {
            Text = "Export Preset", FontSize = 15, FontWeight = FontWeights.SemiBold,
            Foreground = new SolidColorBrush(Color.FromRgb(0x30, 0xD1, 0x58)),
            Margin = new Thickness(0, 0, 0, 14)
        });

        void AddPreset(string label, string desc, string fmt, int? maxW, double upscale = 1.0)
        {
            var btn = new Button
            {
                Height = 44, HorizontalContentAlignment = HorizontalAlignment.Left,
                Padding = new Thickness(14, 0, 14, 0), Margin = new Thickness(0, 0, 0, 8),
                Cursor = Cursors.Hand
            };
            var tpl   = new ControlTemplate(typeof(Button));
            var bdFac = new FrameworkElementFactory(typeof(Border));
            bdFac.SetValue(Border.BackgroundProperty, new SolidColorBrush(Color.FromRgb(0x28, 0x28, 0x2F)));
            bdFac.SetValue(Border.CornerRadiusProperty, new CornerRadius(9));
            bdFac.SetValue(Border.BorderBrushProperty, new SolidColorBrush(Color.FromRgb(0x3A, 0x3A, 0x48)));
            bdFac.SetValue(Border.BorderThicknessProperty, new Thickness(1));
            var inner = new FrameworkElementFactory(typeof(StackPanel));
            inner.SetValue(StackPanel.OrientationProperty, Orientation.Vertical);
            inner.SetValue(StackPanel.MarginProperty, new Thickness(14, 0, 14, 0));
            var t1 = new FrameworkElementFactory(typeof(TextBlock));
            t1.SetValue(TextBlock.TextProperty, label);
            t1.SetValue(TextBlock.ForegroundProperty, Brushes.White);
            t1.SetValue(TextBlock.FontWeightProperty, FontWeights.Medium);
            t1.SetValue(TextBlock.FontSizeProperty, 13.0);
            var t2 = new FrameworkElementFactory(typeof(TextBlock));
            t2.SetValue(TextBlock.TextProperty, desc);
            t2.SetValue(TextBlock.ForegroundProperty, new SolidColorBrush(Color.FromRgb(0x8E, 0x8E, 0xA0)));
            t2.SetValue(TextBlock.FontSizeProperty, 11.0);
            inner.AppendChild(t1);
            inner.AppendChild(t2);
            bdFac.AppendChild(inner);
            tpl.VisualTree = bdFac;
            btn.Template = tpl;
            btn.Click += (_, _) =>
            {
                dlg.Close();
                QuickExport(fmt, maxW, upscale);
            };
            stack.Children.Add(btn);
        }

        AddPreset("Web — JPEG",     "Max 1920 px wide · 85% quality",  "jpg",  1920);
        AddPreset("Web — PNG",      "Max 1920 px wide · lossless",      "png",  1920);
        AddPreset("Print — JPEG",   "Full resolution · 95% quality",    "jpg",  null);
        AddPreset("Full — PNG",     "Full resolution · lossless",        "png",  null);
        AddPreset("Upscale 2× — PNG", "2× resolution · high-quality resample", "png", null, 2.0);
        AddPreset("Upscale 4× — PNG", "4× resolution · high-quality resample", "png", null, 4.0);

        var cancelBtn = new Button
        {
            Content = "Cancel", Width = 80, Height = 32,
            HorizontalAlignment = HorizontalAlignment.Right,
            Foreground = new SolidColorBrush(Color.FromRgb(0x8E, 0x8E, 0xA0)),
            FontSize = 12, Cursor = Cursors.Hand, Margin = new Thickness(0, 6, 0, 0)
        };
        var cTpl   = new ControlTemplate(typeof(Button));
        var cBd    = new FrameworkElementFactory(typeof(Border));
        cBd.SetValue(Border.BackgroundProperty, Brushes.Transparent);
        cBd.SetValue(Border.CornerRadiusProperty, new CornerRadius(7));
        var cCp = new FrameworkElementFactory(typeof(ContentPresenter));
        cCp.SetValue(ContentPresenter.HorizontalAlignmentProperty, HorizontalAlignment.Center);
        cCp.SetValue(ContentPresenter.VerticalAlignmentProperty, VerticalAlignment.Center);
        cBd.AppendChild(cCp);
        cTpl.VisualTree = cBd;
        cancelBtn.Template = cTpl;
        cancelBtn.Click += (_, _) => dlg.Close();
        stack.Children.Add(cancelBtn);

        card.Child  = stack;
        dlg.Content = card;
        dlg.KeyDown += (_, ke) => { if (ke.Key == Key.Escape) dlg.Close(); };
        dlg.ShowDialog();
    }

    private void QuickExport(string format, int? maxWidth, double upscale = 1.0)
    {
        if (!_imageLoaded || _sourcePixels == null) return;

        var saveDlg = new SaveFileDialog
        {
            Title    = "Quick Export",
            FileName = $"{_fileName}-luma",
            Filter   = format == "png" ? "PNG Image|*.png" : "JPEG Image|*.jpg"
        };
        if (saveDlg.ShowDialog() != true) return;

        try
        {
            var rendered = ImageProcessor.Render(_sourcePixels, _sourceW, _sourceH, _adj);
            BitmapSource exportSrc = _hasPendingCrop
                ? new CroppedBitmap(rendered, new Int32Rect(_pendCropX, _pendCropY, _pendCropW, _pendCropH))
                : (BitmapSource)rendered;

            if (_markupStrokes.Count > 0)
                exportSrc = CompositeMarkupOnExport(exportSrc);

            // Resize if maxWidth specified
            if (maxWidth.HasValue && exportSrc.PixelWidth > maxWidth.Value)
            {
                double scale = (double)maxWidth.Value / exportSrc.PixelWidth;
                int tw = maxWidth.Value;
                int th = (int)(exportSrc.PixelHeight * scale);
                var tb = new TransformedBitmap(exportSrc, new ScaleTransform(scale, scale));
                exportSrc = tb;
            }

            // High-quality upscale (no-model "super-resolution" via Fant resampling)
            if (upscale > 1.0)
                exportSrc = UpscaleBitmap(exportSrc, upscale);

            SaveBitmapAs(exportSrc, saveDlg.FileName, format);
            ShowSaveSuccessDialog(saveDlg.FileName);
        }
        catch (Exception ex)
        {
            ShowToast("Export Failed", ex.Message, success: false);
        }
    }

    private void SaveBitmapAs(BitmapSource src, string path, string format)
    {
        BitmapEncoder enc = format switch
        {
            "png"  => new PngBitmapEncoder(),
            _      => new JpegBitmapEncoder { QualityLevel = format == "jpg_print" ? 95 : _jpegQuality }
        };
        enc.Frames.Add(BitmapFrame.Create(src));
        using var fs = File.Create(path);
        enc.Save(fs);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // FILMSTRIP
    // ═══════════════════════════════════════════════════════════════════════

    private async Task LoadFilmstripAsync(string folder)
    {
        // Cancel any in-flight filmstrip load
        _filmstripCts?.Cancel();
        _filmstripCts?.Dispose();
        _filmstripCts = new CancellationTokenSource();
        var token = _filmstripCts.Token;

        if (string.IsNullOrEmpty(folder) || !System.IO.Directory.Exists(folder)) return;

        var supportedExts = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
            { ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif", ".heic", ".heif" };

        string[] files;
        try
        {
            files = System.IO.Directory.GetFiles(folder)
                .Where(f => supportedExts.Contains(System.IO.Path.GetExtension(f)))
                .OrderBy(f => f)
                .ToArray();
        }
        catch { return; }

        if (files.Length < 2)
        {
            if (!token.IsCancellationRequested)
                await Dispatcher.InvokeAsync(() =>
                {
                    FilmstripBar.Visibility = Visibility.Collapsed;
                    FilmstripRowDef.Height  = new GridLength(0);
                });
            return;
        }

        // Show filmstrip immediately with placeholder tiles — no waiting for decodes
        var placeholders = new Border[files.Length];
        await Dispatcher.InvokeAsync(() =>
        {
            FilmstripPanel.Children.Clear();
            for (int i = 0; i < files.Length; i++)
            {
                bool isCurrent = string.Equals(files[i], _currentFilePath, StringComparison.OrdinalIgnoreCase);
                var border = new Border
                {
                    Width = 72, Height = 72, CornerRadius = new CornerRadius(6),
                    Margin = new Thickness(0, 0, 6, 0),
                    BorderThickness = new Thickness(isCurrent ? 2 : 0),
                    BorderBrush = new SolidColorBrush(Color.FromRgb(0x0A, 0x84, 0xFF)),
                    Background = new SolidColorBrush(Color.FromRgb(0x2A, 0x2A, 0x2E)),
                    Cursor = Cursors.Hand, Tag = files[i],
                    ClipToBounds = true,
                    Opacity = isCurrent ? 1.0 : 0.65
                };
                string capturedPath = files[i];
                border.MouseDown += (s, _) =>
                {
                    if (s is Border b && b.Tag is string p) LoadImageFile(p);
                };
                placeholders[i] = border;
                FilmstripPanel.Children.Add(border);
            }
            FilmstripBar.Visibility = Visibility.Visible;
            FilmstripRowDef.Height  = new GridLength(96);
        });

        if (token.IsCancellationRequested) return;

        // Decode thumbnails in parallel (max 4 at once) and fill placeholders as they finish
        var sem = new SemaphoreSlim(4);
        var tasks = Enumerable.Range(0, files.Length).Select(i => Task.Run(async () =>
        {
            await sem.WaitAsync(token).ConfigureAwait(false);
            try
            {
                if (token.IsCancellationRequested) return;
                BitmapImage? bmi = null;
                try
                {
                    var b = new BitmapImage();
                    b.BeginInit();
                    b.UriSource         = new Uri(files[i]);
                    b.DecodePixelHeight = 72;
                    b.CacheOption       = BitmapCacheOption.OnLoad;
                    b.EndInit();
                    b.Freeze();
                    bmi = b;
                }
                catch { /* unsupported or corrupt — leave placeholder grey */ }

                if (bmi != null && !token.IsCancellationRequested)
                {
                    var thumb  = bmi;
                    var border = placeholders[i];
                    await Dispatcher.InvokeAsync(() =>
                    {
                        if (token.IsCancellationRequested) return;
                        border.Child = new System.Windows.Controls.Image
                        {
                            Source  = thumb,
                            Stretch = System.Windows.Media.Stretch.UniformToFill
                        };
                    });
                }
            }
            finally { sem.Release(); }
        }, token)).ToArray();

        await Task.WhenAll(tasks);
        sem.Dispose();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // EXIF METADATA VIEWER
    // ═══════════════════════════════════════════════════════════════════════

    private void ExifHeader_MouseDown(object sender, MouseButtonEventArgs e)
    {
        _exifOpen = !_exifOpen;
        ExifPanel.Visibility = _exifOpen ? Visibility.Visible : Visibility.Collapsed;
        ExifChevron.Text     = _exifOpen ? "▾" : "▸";
    }

    private async Task ReadExifAsync(string path)
    {
        string info = await Task.Run(() =>
        {
            try
            {
                BitmapMetadata? meta = null;
                using var stream = File.OpenRead(path);
                var decoder = BitmapDecoder.Create(stream,
                    BitmapCreateOptions.PreservePixelFormat, BitmapCacheOption.OnLoad);
                meta = decoder.Frames[0].Metadata as BitmapMetadata;
                if (meta == null) return "No EXIF data";

                string? Q(string q) { try { return meta.GetQuery(q)?.ToString()?.Trim(); } catch { return null; } }

                var sb = new System.Text.StringBuilder();

                string? make  = meta.CameraManufacturer?.Trim();
                string? model = meta.CameraModel?.Trim();
                if (!string.IsNullOrEmpty(make) || !string.IsNullOrEmpty(model))
                    sb.AppendLine($"Camera: {(make + " " + model).Trim()}");

                string? lens = Q("/app1/ifd/exif/{ushort=42036}");
                if (!string.IsNullOrEmpty(lens)) sb.AppendLine($"Lens: {lens}");

                string? exp = Q("/app1/ifd/exif/{ushort=33434}");
                if (!string.IsNullOrEmpty(exp)) sb.AppendLine($"Shutter: {exp}s");

                string? fn = Q("/app1/ifd/exif/{ushort=33437}");
                if (!string.IsNullOrEmpty(fn)) sb.AppendLine($"Aperture: f/{fn}");

                string? iso = Q("/app1/ifd/exif/{ushort=34855}");
                if (!string.IsNullOrEmpty(iso)) sb.AppendLine($"ISO: {iso}");

                string? fl = Q("/app1/ifd/exif/{ushort=37386}");
                if (!string.IsNullOrEmpty(fl)) sb.AppendLine($"Focal: {fl}mm");

                string? dt = meta.DateTaken?.Trim() ?? Q("/app1/ifd/exif/{ushort=36867}");
                if (!string.IsNullOrEmpty(dt)) sb.AppendLine($"Date: {dt}");

                string result = sb.ToString().Trim();
                return string.IsNullOrEmpty(result) ? "No EXIF data" : result;
            }
            catch { return "No EXIF data"; }
        });

        await Dispatcher.InvokeAsync(() => ExifText.Text = info);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // BATCH EXPORT (enhanced)
    // ═══════════════════════════════════════════════════════════════════════

    private async void BatchBtn_Click(object sender, RoutedEventArgs e)
    {
        if (!_imageLoaded) return;

        // ── Options dialog ──
        var optDlg = new Window
        {
            Width = 380, SizeToContent = SizeToContent.Height,
            WindowStyle = WindowStyle.None, AllowsTransparency = true,
            Background = Brushes.Transparent, ResizeMode = ResizeMode.NoResize,
            WindowStartupLocation = WindowStartupLocation.CenterOwner,
            Owner = this, ShowInTaskbar = false
        };

        var card = new Border
        {
            Background   = new SolidColorBrush(Color.FromRgb(0x1C, 0x1C, 0x1E)),
            CornerRadius  = new CornerRadius(14),
            Padding      = new Thickness(22, 18, 22, 18),
            Effect        = new System.Windows.Media.Effects.DropShadowEffect
                { Color = Colors.Black, Opacity = 0.75, BlurRadius = 28, ShadowDepth = 4 }
        };

        var sp = new StackPanel();
        sp.Children.Add(new TextBlock
        {
            Text = "⊞  Batch Export", FontSize = 15, FontWeight = FontWeights.SemiBold,
            Foreground = new SolidColorBrush(Color.FromRgb(0x30, 0xD1, 0x58)),
            Margin = new Thickness(0, 0, 0, 14)
        });

        // Format picker
        sp.Children.Add(new TextBlock { Text = "Output Format", Foreground = new SolidColorBrush(Color.FromRgb(0x8E,0x8E,0xA0)), FontSize=12, Margin=new Thickness(0,0,0,4) });
        var fmtCombo = new ComboBox { Height=32, Margin=new Thickness(0,0,0,12), Foreground=Brushes.White,
            Background=new SolidColorBrush(Color.FromRgb(0x28,0x28,0x2F)), FontSize=13, SelectedIndex=0 };
        fmtCombo.Items.Add(new ComboBoxItem { Content="JPEG (95%)",    Tag="jpg",  Foreground=Brushes.Black });
        fmtCombo.Items.Add(new ComboBoxItem { Content="PNG (lossless)", Tag="png",  Foreground=Brushes.Black });
        sp.Children.Add(fmtCombo);

        // Suffix
        sp.Children.Add(new TextBlock { Text = "Filename suffix", Foreground = new SolidColorBrush(Color.FromRgb(0x8E,0x8E,0xA0)), FontSize=12, Margin=new Thickness(0,0,0,4) });
        var suffixBox = new TextBox { Text="-luma", Height=32, Padding=new Thickness(10,0,10,0),
            Background=new SolidColorBrush(Color.FromRgb(0x28,0x28,0x2F)), Foreground=Brushes.White,
            FontSize=13, BorderBrush=new SolidColorBrush(Color.FromRgb(0x3A,0x3A,0x48)), BorderThickness=new Thickness(1),
            Margin=new Thickness(0,0,0,12), VerticalContentAlignment=VerticalAlignment.Center };
        sp.Children.Add(suffixBox);

        // Max width
        sp.Children.Add(new TextBlock { Text = "Max width (px, blank = original)", Foreground = new SolidColorBrush(Color.FromRgb(0x8E,0x8E,0xA0)), FontSize=12, Margin=new Thickness(0,0,0,4) });
        var maxWBox = new TextBox { Text="", Height=32, Padding=new Thickness(10,0,10,0),
            Background=new SolidColorBrush(Color.FromRgb(0x28,0x28,0x2F)), Foreground=Brushes.White,
            FontSize=13, BorderBrush=new SolidColorBrush(Color.FromRgb(0x3A,0x3A,0x48)), BorderThickness=new Thickness(1),
            Margin=new Thickness(0,0,0,12), VerticalContentAlignment=VerticalAlignment.Center };
        sp.Children.Add(maxWBox);

        // Watermark text
        sp.Children.Add(new TextBlock { Text = "Watermark text (blank = none)", Foreground = new SolidColorBrush(Color.FromRgb(0x8E,0x8E,0xA0)), FontSize=12, Margin=new Thickness(0,0,0,4) });
        var wmBox = new TextBox { Text="", Height=32, Padding=new Thickness(10,0,10,0),
            Background=new SolidColorBrush(Color.FromRgb(0x28,0x28,0x2F)), Foreground=Brushes.White,
            FontSize=13, BorderBrush=new SolidColorBrush(Color.FromRgb(0x3A,0x3A,0x48)), BorderThickness=new Thickness(1),
            Margin=new Thickness(0,0,0,12), VerticalContentAlignment=VerticalAlignment.Center };
        sp.Children.Add(wmBox);

        // Auto enhance per image
        var autoChk = new CheckBox
        {
            Content = new TextBlock { Text="Auto enhance each image individually", Foreground=Brushes.White, FontSize=13 },
            Foreground=Brushes.White, IsChecked=false, Margin=new Thickness(0,0,0,18), Cursor=Cursors.Hand
        };
        sp.Children.Add(autoChk);

        // Buttons
        bool proceed = false;
        var btnRow = new StackPanel { Orientation=Orientation.Horizontal, HorizontalAlignment=HorizontalAlignment.Right };

        var cancelB = MakeDialogBtn("Cancel", Color.FromRgb(0x28,0x28,0x2F));
        cancelB.Click += (_, _) => optDlg.Close();
        btnRow.Children.Add(cancelB);

        var goB = MakeDialogBtn("Continue →", Color.FromRgb(0x0A,0x84,0xFF));
        goB.Foreground = Brushes.White;
        goB.Margin = new Thickness(8,0,0,0);
        goB.Click += (_, _) => { proceed = true; optDlg.Close(); };
        btnRow.Children.Add(goB);
        sp.Children.Add(btnRow);

        card.Child  = sp;
        optDlg.Content = card;
        optDlg.KeyDown += (_, ke) => { if (ke.Key == Key.Escape) optDlg.Close(); };
        optDlg.ShowDialog();

        if (!proceed) return;

        string suffix     = string.IsNullOrWhiteSpace(suffixBox.Text) ? "-luma" : suffixBox.Text;
        string format     = ((fmtCombo.SelectedItem as ComboBoxItem)?.Tag as string) ?? "jpg";
        string ext        = format == "png" ? ".png" : ".jpg";
        bool   autoEnh    = autoChk.IsChecked == true;
        string wmText     = wmBox.Text.Trim();
        int?   maxW       = int.TryParse(maxWBox.Text.Trim(), out int mw) && mw > 0 ? mw : (int?)null;

        // Pick folders
        var inputDlg = new Microsoft.Win32.OpenFolderDialog { Title = "Select folder with photos to process" };
        if (inputDlg.ShowDialog() != true) return;

        var outputDlg = new Microsoft.Win32.OpenFolderDialog { Title = "Select output folder" };
        if (outputDlg.ShowDialog() != true) return;

        string inputDir  = inputDlg.FolderName;
        string outputDir = outputDlg.FolderName;
        var    adjSnap   = _adj.Clone();

        var supportedExts = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
            { ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif" };

        var files = System.IO.Directory.GetFiles(inputDir, "*.*")
            .Where(f => supportedExts.Contains(System.IO.Path.GetExtension(f)))
            .ToArray();

        if (files.Length == 0)
        {
            ShowToast("Batch Export", "No supported images found in the selected folder.", success: false);
            return;
        }

        // Progress dialog
        var prog      = BuildProgressDialog(files.Length, out var statusText, out var bar);
        prog.Show();

        int done = 0, failed = 0;
        var neuralRef = _neuralEnhancer;  // capture for background thread

        await Task.Run(() =>
        {
            foreach (var file in files)
            {
                try
                {
                    var (pixels, w, h) = ImageProcessor.LoadImageFile(file);

                    AdjustmentState adj = adjSnap;
                    if (autoEnh)
                    {
                        var analysis = ImageProcessor.Analyze(pixels, w, h);
                        adj = ImageProcessor.ComputeAutoParams(analysis);
                    }

                    var rendered = ImageProcessor.Render(pixels, w, h, adj);
                    BitmapSource src = rendered;

                    if (maxW.HasValue && src.PixelWidth > maxW.Value)
                    {
                        double scale = (double)maxW.Value / src.PixelWidth;
                        src = new TransformedBitmap(rendered, new ScaleTransform(scale, scale));
                        src.Freeze();
                    }

                    if (!string.IsNullOrEmpty(wmText))
                        src = ApplyWatermarkText(src, wmText);

                    string name    = System.IO.Path.GetFileNameWithoutExtension(file);
                    string outPath = System.IO.Path.Combine(outputDir, name + suffix + ext);

                    BitmapEncoder enc = format == "png"
                        ? (BitmapEncoder)new PngBitmapEncoder()
                        : new JpegBitmapEncoder { QualityLevel = _jpegQuality };
                    enc.Frames.Add(BitmapFrame.Create(src));
                    using var fs = System.IO.File.Create(outPath);
                    enc.Save(fs);
                    done++;
                }
                catch { failed++; }

                Dispatcher.Invoke(() =>
                {
                    bar.Value       = done + failed;
                    statusText.Text = $"Processing {done + failed} / {files.Length}…";
                });
            }
        });

        prog.Close();
        string resultBody = $"{done} exported, {failed} failed.\nOutput folder: {outputDir}";
        ShowToast("✓  Batch Export Complete", resultBody, success: failed == 0);
    }

    private static BitmapSource ApplyWatermarkText(BitmapSource src, string text)
    {
        int pw = src.PixelWidth, ph = src.PixelHeight;
        double fontSize = Math.Max(14, pw / 40.0);

        var dv = new DrawingVisual();
        using (var dc = dv.RenderOpen())
        {
            dc.DrawImage(src, new Rect(0, 0, pw, ph));

            var tf  = new Typeface(new FontFamily("Segoe UI"), FontStyles.Normal, FontWeights.SemiBold, FontStretches.Normal);
            var ft  = new FormattedText(text, System.Globalization.CultureInfo.InvariantCulture,
                          FlowDirection.LeftToRight, tf, fontSize,
                          new SolidColorBrush(Color.FromArgb(180, 255, 255, 255)), 96);

            // Shadow
            dc.DrawText(new FormattedText(text, System.Globalization.CultureInfo.InvariantCulture,
                          FlowDirection.LeftToRight, tf, fontSize,
                          new SolidColorBrush(Color.FromArgb(140, 0, 0, 0)), 96),
                new Point(pw - ft.Width - 18 + 2, ph - ft.Height - 14 + 2));
            dc.DrawText(ft, new Point(pw - ft.Width - 18, ph - ft.Height - 14));
        }

        var rtb = new RenderTargetBitmap(pw, ph, 96, 96, PixelFormats.Pbgra32);
        rtb.Render(dv);
        rtb.Freeze();
        return rtb;
    }

    private Window BuildProgressDialog(int total, out TextBlock statusText, out System.Windows.Controls.ProgressBar bar)
    {
        var prog = new Window
        {
            Width = 380, SizeToContent = SizeToContent.Height,
            WindowStyle = WindowStyle.None, AllowsTransparency = true,
            Background = Brushes.Transparent, ResizeMode = ResizeMode.NoResize,
            WindowStartupLocation = WindowStartupLocation.CenterOwner, Owner = this, ShowInTaskbar = false
        };

        var card = new Border
        {
            Background   = new SolidColorBrush(Color.FromRgb(0x1C, 0x1C, 0x1E)),
            CornerRadius  = new CornerRadius(14),
            Padding      = new Thickness(22, 18, 22, 18),
            Effect        = new System.Windows.Media.Effects.DropShadowEffect
                { Color = Colors.Black, Opacity = 0.75, BlurRadius = 28, ShadowDepth = 4 }
        };

        var sp = new StackPanel();
        sp.Children.Add(new TextBlock
        {
            Text = "⊞  Batch Export", FontSize = 15, FontWeight = FontWeights.SemiBold,
            Foreground = new SolidColorBrush(Color.FromRgb(0x30, 0xD1, 0x58)),
            Margin = new Thickness(0, 0, 0, 12)
        });

        statusText = new TextBlock
        {
            Text = "Starting…", FontSize = 13,
            Foreground = new SolidColorBrush(Color.FromRgb(0x8E, 0x8E, 0xA0)),
            Margin = new Thickness(0, 0, 0, 10)
        };
        sp.Children.Add(statusText);

        bar = new System.Windows.Controls.ProgressBar
        {
            Height = 6, Minimum = 0, Maximum = total,
            Background = new SolidColorBrush(Color.FromRgb(0x28, 0x28, 0x2F)),
            Foreground = new SolidColorBrush(Color.FromRgb(0x30, 0xD1, 0x58)),
            BorderThickness = new Thickness(0)
        };
        sp.Children.Add(bar);

        card.Child  = sp;
        prog.Content = card;
        return prog;
    }

    private static Button MakeDialogBtn(string label, Color bg)
    {
        var btn = new Button
        {
            Content = label, Height = 34, Padding = new Thickness(16, 0, 16, 0),
            Cursor = Cursors.Hand,
            Foreground = new SolidColorBrush(Color.FromRgb(0xF0, 0xF0, 0xF2)),
            FontSize = 13
        };
        var tpl   = new ControlTemplate(typeof(Button));
        var bdFac = new FrameworkElementFactory(typeof(Border));
        bdFac.SetValue(Border.BackgroundProperty, new SolidColorBrush(bg));
        bdFac.SetValue(Border.CornerRadiusProperty, new CornerRadius(9));
        bdFac.SetValue(Border.PaddingProperty, new Thickness(16, 0, 16, 0));
        var cp = new FrameworkElementFactory(typeof(ContentPresenter));
        cp.SetValue(ContentPresenter.HorizontalAlignmentProperty, HorizontalAlignment.Center);
        cp.SetValue(ContentPresenter.VerticalAlignmentProperty,   VerticalAlignment.Center);
        bdFac.AppendChild(cp);
        tpl.VisualTree = bdFac;
        btn.Template = tpl;
        return btn;
    }

    // ── In-app update ──────────────────────────────────────────────────────────

    private CancellationTokenSource? _updateCts;

    /// <summary>
    /// Runs on startup (background). Shows the update banner if a newer
    /// GitHub Release exists. Silently does nothing when offline.
    /// </summary>
    private async Task CheckForUpdateAsync()
    {
        var info = await UpdateChecker.CheckAsync().ConfigureAwait(false);
        if (info == null) return;

        await Dispatcher.InvokeAsync(() =>
        {
            UpdateBannerText.Text = $"LumaPhoto {info.Version} is available";
            UpdateBanner.Visibility = Visibility.Visible;

            // Store update info on the button tag so the click handler can reach it.
            UpdateNowBtn.Tag = info;

            UpdateNowBtn.MouseLeftButtonUp  += UpdateNowBtn_Click;
            UpdateDismissBtn.MouseLeftButtonUp += (_, _) =>
                UpdateBanner.Visibility = Visibility.Collapsed;

            // Subtle hover effects
            UpdateNowBtn.MouseEnter += (_, _) =>
                UpdateNowBtn.Background = new SolidColorBrush(Color.FromRgb(0x00, 0x6C, 0xD9));
            UpdateNowBtn.MouseLeave += (_, _) =>
                UpdateNowBtn.Background = new SolidColorBrush(Color.FromRgb(0x0A, 0x84, 0xFF));
        });
    }

    private async void UpdateNowBtn_Click(object sender, System.Windows.Input.MouseButtonEventArgs e)
    {
        if (UpdateNowBtn.Tag is not UpdateInfo info) return;

        // If it's a webpage URL (no .exe asset found), just open the browser.
        if (!info.DownloadUrl.EndsWith(".exe", StringComparison.OrdinalIgnoreCase))
        {
            System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo
            {
                FileName = info.DownloadUrl, UseShellExecute = true
            });
            return;
        }

        // Disable the button during download.
        UpdateNowBtn.MouseLeftButtonUp -= UpdateNowBtn_Click;
        UpdateNowBtn.Cursor = System.Windows.Input.Cursors.Wait;

        _updateCts = new CancellationTokenSource();
        var progress = new Progress<int>(pct =>
            UpdateNowText.Text = $"Downloading… {pct}%");

        string? installer = await UpdateChecker
            .DownloadInstallerAsync(info.DownloadUrl, progress, _updateCts.Token)
            .ConfigureAwait(true);   // resume on UI thread

        if (installer == null)
        {
            // Download failed — open the releases page instead.
            UpdateNowText.Text = "Update now";
            UpdateNowBtn.Cursor = System.Windows.Input.Cursors.Hand;
            UpdateNowBtn.MouseLeftButtonUp += UpdateNowBtn_Click;
            UpdateNowBtn.Tag = info with { DownloadUrl =
                $"https://github.com/{AppVersion.GitHubRepo}/releases/latest" };
            return;
        }

        // Launch installer silently then close this instance so the exe can be replaced.
        System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo
        {
            FileName = installer, Arguments = "/SILENT /NORESTART", UseShellExecute = true
        });
        Application.Current.Shutdown();
    }

}
