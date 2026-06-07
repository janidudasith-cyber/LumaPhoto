using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace LumaPhoto;

// Quick-win + neural-style features split out of the MainWindow monolith:
// copy/paste look · custom + scene "Looks" · smart zoom · recent files · upscale export.
public partial class MainWindow
{
    // ── Paths ──────────────────────────────────────────────────────────────────
    private static readonly string AppDataDir =
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "LumaPhoto");
    private static string LooksDir       => Path.Combine(AppDataDir, "looks");
    private static string RecentFilePath => Path.Combine(AppDataDir, "recent.txt");

    // ── Copy / Paste look ────────────────────────────────────────────────────────
    private AdjustmentState? _copiedLook;

    // Copies tonal + colour fields only — geometry (rotation/tilt) and crop are left
    // untouched, so a look can be pasted onto a differently-framed photo.
    private static void CopyLookFields(AdjustmentState src, AdjustmentState dst)
    {
        dst.Exposure = src.Exposure; dst.Brilliance = src.Brilliance; dst.Highlights = src.Highlights;
        dst.Shadows = src.Shadows; dst.Contrast = src.Contrast; dst.Brightness = src.Brightness;
        dst.BlackPoint = src.BlackPoint; dst.Saturation = src.Saturation; dst.Vibrance = src.Vibrance;
        dst.Warmth = src.Warmth; dst.Tint = src.Tint; dst.Sharpness = src.Sharpness;
        dst.Definition = src.Definition; dst.Noise = src.Noise; dst.Vignette = src.Vignette;
        dst.Filter = src.Filter; dst.FilterIntensity = src.FilterIntensity;
        dst.Curve0 = src.Curve0; dst.Curve64 = src.Curve64; dst.Curve128 = src.Curve128;
        dst.Curve192 = src.Curve192; dst.Curve255 = src.Curve255;
        dst.HslRH = src.HslRH; dst.HslRS = src.HslRS; dst.HslOH = src.HslOH; dst.HslOS = src.HslOS;
        dst.HslYH = src.HslYH; dst.HslYS = src.HslYS; dst.HslGH = src.HslGH; dst.HslGS = src.HslGS;
        dst.HslCH = src.HslCH; dst.HslCS = src.HslCS; dst.HslBH = src.HslBH; dst.HslBS = src.HslBS;
        dst.HslPH = src.HslPH; dst.HslPS = src.HslPS;
    }

    private void CopyLook_Click(object sender, RoutedEventArgs e)
    {
        if (!_imageLoaded) return;
        _copiedLook = _adj.Clone();
        PasteSettingsBtn.IsEnabled = true;
        ShowToast("Copied", "Adjustments copied — paste onto any photo");
    }

    private void PasteLook_Click(object sender, RoutedEventArgs e) => ApplyCopiedLook();

    private void ApplyCopiedLook()
    {
        if (_copiedLook == null || !_imageLoaded) return;
        PushHistory();
        CopyLookFields(_copiedLook, _adj);
        RefreshAllFromAdj();
        ShowToast("Pasted", "Adjustments applied");
    }

    // Apply a full look (tonal only) with one undo entry + full UI refresh.
    private void ApplyLook(AdjustmentState look)
    {
        if (!_imageLoaded) return;
        PushHistory();
        CopyLookFields(look, _adj);
        RefreshAllFromAdj();
    }

    // Push _adj back out to every UI control and re-render. Mirrors RestoreSnapshot
    // minus the pixel/crop restore.
    private void RefreshAllFromAdj()
    {
        ApplyAdjToSliders(_adj);
        HighlightFilter(_adj.Filter);
        FilterIntensitySlider.Value = _adj.FilterIntensity;
        FilterIntensityLabel.Text   = $"{_adj.FilterIntensity:0.0}%";
        FilterIntensityPanel.Visibility = _adj.Filter != FilterType.None
            ? Visibility.Visible : Visibility.Collapsed;
        DrawCurves();
        SyncHslSlidersFromAdj();
        TurnOffAutoEnhanceUI();
        DoRender();
    }

    // ── Smart zoom — jump to 100% (actual-pixel) centred on the cursor ─────────────
    private void ZoomToActualSize(Point center)
    {
        if (PhotoDisplay.Source is not BitmapSource bs) return;
        double aw = PhotoDisplay.ActualWidth, ah = PhotoDisplay.ActualHeight;
        if (aw <= 0 || ah <= 0) return;
        // Uniform-stretch fit scale = min over both axes; 100% needs zoom = 1 / fitScale.
        double fitScale = System.Math.Min(aw / bs.PixelWidth, ah / bs.PixelHeight);
        if (fitScale <= 0) return;
        double target = 1.0 / fitScale;
        if (target <= 1.01) return;   // image already shown at ≥100% when fit
        var m = Matrix.Identity;
        m.ScaleAt(target, target, center.X, center.Y);
        _zoomMatrix = m;
        ImageZoomGroup.RenderTransform = new MatrixTransform(_zoomMatrix);
        UpdateStatusBar();
    }

    // ── High-quality upscale (no-model super-resolution via Fant resampling) ───────
    private static BitmapSource UpscaleBitmap(BitmapSource src, double factor)
    {
        int tw = (int)System.Math.Round(src.PixelWidth  * factor);
        int th = (int)System.Math.Round(src.PixelHeight * factor);
        var dv = new DrawingVisual();
        RenderOptions.SetBitmapScalingMode(dv, BitmapScalingMode.HighQuality);
        using (var dc = dv.RenderOpen())
            dc.DrawImage(src, new Rect(0, 0, tw, th));
        var rtb = new RenderTargetBitmap(tw, th, 96, 96, PixelFormats.Pbgra32);
        rtb.Render(dv);
        rtb.Freeze();
        return rtb;
    }

    // ── Recent files ──────────────────────────────────────────────────────────────
    private List<string> LoadRecentFiles()
    {
        try
        {
            if (File.Exists(RecentFilePath))
                return File.ReadAllLines(RecentFilePath).Where(File.Exists).Take(12).ToList();
        }
        catch { }
        return new List<string>();
    }

    private void AddRecentFile(string path)
    {
        try
        {
            Directory.CreateDirectory(AppDataDir);
            var list = LoadRecentFiles();
            list.RemoveAll(p => string.Equals(p, path, StringComparison.OrdinalIgnoreCase));
            list.Insert(0, path);
            if (list.Count > 12) list = list.Take(12).ToList();
            File.WriteAllLines(RecentFilePath, list);
        }
        catch { }
    }

    private void RecentBtn_Click(object sender, RoutedEventArgs e) => ShowRecentMenu();

    private void ShowRecentMenu()
    {
        var recents = LoadRecentFiles();
        if (recents.Count == 0)
        {
            ShowToast("Recent", "No recently opened photos yet", success: false);
            return;
        }

        var (dlg, stack) = BuildCardDialog("Recent Photos", Color.FromRgb(0x0A, 0x84, 0xFF));
        foreach (var path in recents)
        {
            string p = path;
            var btn = DarkListButton(Path.GetFileName(p), Path.GetDirectoryName(p) ?? "", Brushes.White, null);
            btn.Click += (_, _) => { dlg.Close(); LoadImageFile(p); };
            stack.Children.Add(btn);
        }
        AddCancelButton(dlg, stack);
        dlg.ShowDialog();
    }

    // ── Looks (custom presets + built-in scene looks) ──────────────────────────────
    private void LooksBtn_Click(object sender, RoutedEventArgs e)
    {
        if (_imageLoaded) ShowLooksDialog();
    }

    private static List<(string name, string desc, AdjustmentState adj)> BuiltInLooks()
    {
        AdjustmentState A(System.Action<AdjustmentState> set) { var a = new AdjustmentState(); set(a); return a; }
        return new()
        {
            ("Natural",        "Gentle, true-to-life lift",        A(a => { a.Exposure=3; a.Brilliance=8; a.Shadows=10; a.Contrast=6; a.Vibrance=12; a.Definition=8; })),
            ("Vivid Pop",      "Punchy colour and contrast",       A(a => { a.Contrast=18; a.Saturation=20; a.Vibrance=25; a.Brilliance=12; a.Definition=14; a.Sharpness=10; })),
            ("Portrait",       "Soft skin, warm tone",             A(a => { a.Exposure=4; a.Shadows=14; a.Highlights=-8; a.Contrast=6; a.Warmth=8; a.Vibrance=10; a.Definition=-6; a.Sharpness=6; })),
            ("Landscape",      "Crisp detail, rich greens & sky",  A(a => { a.Contrast=14; a.Vibrance=22; a.Saturation=8; a.Definition=16; a.Sharpness=12; a.Shadows=8; a.Highlights=-6; })),
            ("Night",          "Lift shadows, tame noise",         A(a => { a.Exposure=12; a.Shadows=25; a.BlackPoint=6; a.Noise=30; a.Contrast=8; a.Highlights=-10; })),
            ("Food",           "Warm, appetising saturation",      A(a => { a.Warmth=10; a.Saturation=16; a.Vibrance=18; a.Contrast=10; a.Brightness=4; a.Definition=10; })),
            ("B&W",            "Monochrome with contrast",         A(a => { a.Saturation=-100; a.Contrast=16; a.Brilliance=10; a.Definition=10; })),
            ("Warm Sunset",    "Golden-hour warmth",               A(a => { a.Warmth=22; a.Tint=6; a.Saturation=12; a.Contrast=8; a.Highlights=-8; a.Shadows=10; })),
            ("Cool Cinematic", "Teal shadows, faded highlights",   A(a => { a.Warmth=-16; a.Tint=-4; a.Contrast=16; a.Shadows=18; a.Highlights=-10; a.Saturation=-6; a.Vignette=20; })),
            ("Matte Fade",     "Lifted blacks, film look",         A(a => { a.Curve0=32; a.Curve255=235; a.Contrast=-6; a.Saturation=-8; a.Warmth=4; })),
        };
    }

    private List<(string name, AdjustmentState adj)> LoadUserLooks()
    {
        var result = new List<(string, AdjustmentState)>();
        try
        {
            if (!Directory.Exists(LooksDir)) return result;
            foreach (var f in Directory.GetFiles(LooksDir, "*.json").OrderBy(f => f))
            {
                try
                {
                    var a = JsonSerializer.Deserialize<AdjustmentState>(File.ReadAllText(f), _jsonOpts);
                    if (a != null) result.Add((Path.GetFileNameWithoutExtension(f), a));
                }
                catch { }
            }
        }
        catch { }
        return result;
    }

    private static readonly JsonSerializerOptions _jsonOpts = new() { IncludeFields = true, WriteIndented = true };

    private void SaveCurrentLook(string name)
    {
        try
        {
            Directory.CreateDirectory(LooksDir);
            string safe = string.Join("_", name.Split(Path.GetInvalidFileNameChars()));
            if (string.IsNullOrWhiteSpace(safe)) safe = "Look";
            File.WriteAllText(Path.Combine(LooksDir, safe + ".json"),
                JsonSerializer.Serialize(_adj, _jsonOpts));
            ShowToast("Saved", $"Look “{safe}” saved");
        }
        catch (Exception ex) { ShowToast("Save failed", ex.Message, success: false); }
    }

    private void DeleteUserLook(string name)
    {
        try
        {
            string path = Path.Combine(LooksDir, name + ".json");
            if (File.Exists(path)) File.Delete(path);
        }
        catch { }
    }

    private void ShowLooksDialog()
    {
        var (dlg, stack) = BuildCardDialog("Looks", Color.FromRgb(0xBF, 0x5A, 0xF2));

        // Save current
        var saveBtn = DarkListButton("＋  Save current as a Look…", "Store the current adjustments", Brushes.White, null);
        saveBtn.Click += (_, _) =>
        {
            dlg.Close();
            string? name = PromptText("Save Look", "Look name:", "My Look");
            if (!string.IsNullOrWhiteSpace(name)) SaveCurrentLook(name!.Trim());
        };
        stack.Children.Add(saveBtn);

        // My Looks
        var userLooks = LoadUserLooks();
        if (userLooks.Count > 0)
        {
            stack.Children.Add(SectionLabel("MY LOOKS"));
            foreach (var (name, adj) in userLooks)
            {
                string n = name; var a = adj;
                stack.Children.Add(CustomLookRow(n, a,
                    onApply:  () => { dlg.Close(); ApplyLook(a);  ShowToast("Applied",  $"“{n}” applied"); },
                    onDelete: () => { DeleteUserLook(n); dlg.Close(); ShowLooksDialog(); }));
            }
        }

        // Built-in
        stack.Children.Add(SectionLabel("BUILT-IN"));
        foreach (var (name, desc, adj) in BuiltInLooks())
        {
            var a = adj;
            string n = name;
            var btn = DarkListButton(n, desc, Brushes.White, null);
            btn.Click += (_, _) => { dlg.Close(); ApplyLook(a); ShowToast("Applied", $"“{n}” applied"); };
            stack.Children.Add(btn);
        }

        AddCancelButton(dlg, stack);
        dlg.ShowDialog();
    }

    // ── Small dark-themed dialog toolkit ───────────────────────────────────────────
    private (Window dlg, StackPanel stack) BuildCardDialog(string title, Color titleColor)
    {
        var dlg = new Window
        {
            Width = 360, MaxHeight = 600, SizeToContent = SizeToContent.Height,
            WindowStyle = WindowStyle.None, AllowsTransparency = true,
            Background = Brushes.Transparent, ResizeMode = ResizeMode.NoResize,
            WindowStartupLocation = WindowStartupLocation.CenterOwner,
            Owner = this, ShowInTaskbar = false
        };
        var card = new Border
        {
            Background  = new SolidColorBrush(Color.FromRgb(0x1C, 0x1C, 0x1E)),
            CornerRadius = new CornerRadius(14),
            Padding     = new Thickness(20, 18, 20, 18),
            Effect      = new System.Windows.Media.Effects.DropShadowEffect
                { Color = Colors.Black, Opacity = 0.75, BlurRadius = 28, ShadowDepth = 4 }
        };
        var outer = new StackPanel();
        outer.Children.Add(new TextBlock
        {
            Text = title, FontSize = 15, FontWeight = FontWeights.SemiBold,
            Foreground = new SolidColorBrush(titleColor), Margin = new Thickness(0, 0, 0, 14)
        });
        var scroll = new ScrollViewer
        {
            VerticalScrollBarVisibility = ScrollBarVisibility.Auto,
            MaxHeight = 460, HorizontalScrollBarVisibility = ScrollBarVisibility.Disabled
        };
        var stack = new StackPanel();
        scroll.Content = stack;
        outer.Children.Add(scroll);
        card.Child = outer;
        dlg.Content = card;
        dlg.KeyDown += (_, ke) => { if (ke.Key == Key.Escape) dlg.Close(); };
        return (dlg, stack);
    }

    private static TextBlock SectionLabel(string text) => new()
    {
        Text = text, FontSize = 10, FontWeight = FontWeights.SemiBold,
        Foreground = new SolidColorBrush(Color.FromRgb(0x8E, 0x8E, 0xA0)),
        Margin = new Thickness(2, 10, 0, 6)
    };

    // A templated dark list row with an optional trailing delete (✕) glyph.
    private Button DarkListButton(string title, string sub, Brush titleColor, System.Action? onDelete)
    {
        var btn = new Button
        {
            Height = 46, HorizontalContentAlignment = HorizontalAlignment.Stretch,
            Margin = new Thickness(0, 0, 0, 8), Cursor = Cursors.Hand
        };
        var tpl = new ControlTemplate(typeof(Button));
        var bd  = new FrameworkElementFactory(typeof(Border));
        bd.SetValue(Border.BackgroundProperty, new SolidColorBrush(Color.FromRgb(0x28, 0x28, 0x2F)));
        bd.SetValue(Border.CornerRadiusProperty, new CornerRadius(9));
        bd.SetValue(Border.BorderBrushProperty, new SolidColorBrush(Color.FromRgb(0x3A, 0x3A, 0x48)));
        bd.SetValue(Border.BorderThicknessProperty, new Thickness(1));
        bd.SetValue(Border.PaddingProperty, new Thickness(14, 0, 10, 0));

        var grid = new FrameworkElementFactory(typeof(Grid));
        var col0 = new FrameworkElementFactory(typeof(ColumnDefinition));
        col0.SetValue(ColumnDefinition.WidthProperty, new GridLength(1, GridUnitType.Star));
        var col1 = new FrameworkElementFactory(typeof(ColumnDefinition));
        col1.SetValue(ColumnDefinition.WidthProperty, GridLength.Auto);
        grid.AppendChild(col0);
        grid.AppendChild(col1);

        var inner = new FrameworkElementFactory(typeof(StackPanel));
        inner.SetValue(StackPanel.OrientationProperty, Orientation.Vertical);
        inner.SetValue(StackPanel.VerticalAlignmentProperty, VerticalAlignment.Center);
        var t1 = new FrameworkElementFactory(typeof(TextBlock));
        t1.SetValue(TextBlock.TextProperty, title);
        t1.SetValue(TextBlock.ForegroundProperty, titleColor);
        t1.SetValue(TextBlock.FontWeightProperty, FontWeights.Medium);
        t1.SetValue(TextBlock.FontSizeProperty, 13.0);
        t1.SetValue(TextBlock.TextTrimmingProperty, TextTrimming.CharacterEllipsis);
        var t2 = new FrameworkElementFactory(typeof(TextBlock));
        t2.SetValue(TextBlock.TextProperty, sub);
        t2.SetValue(TextBlock.ForegroundProperty, new SolidColorBrush(Color.FromRgb(0x8E, 0x8E, 0xA0)));
        t2.SetValue(TextBlock.FontSizeProperty, 11.0);
        t2.SetValue(TextBlock.TextTrimmingProperty, TextTrimming.CharacterEllipsis);
        inner.AppendChild(t1);
        inner.AppendChild(t2);
        inner.SetValue(Grid.ColumnProperty, 0);
        grid.AppendChild(inner);

        bd.AppendChild(grid);
        tpl.VisualTree = bd;
        btn.Template = tpl;

        if (onDelete != null)
        {
            // A separate ✕ button overlaid via a Grid wrapper isn't possible inside
            // a single templated Button, so attach delete to right-click instead.
            btn.ToolTip = "Click to apply · right-click to delete";
            btn.MouseRightButtonUp += (_, ev) => { ev.Handled = true; onDelete(); };
        }
        return btn;
    }

    // Custom look row — left area applies, small × button on right deletes.
    private Border CustomLookRow(string name, AdjustmentState adj,
        System.Action onApply, System.Action onDelete)
    {
        var outer = new Border
        {
            Height = 46, Margin = new Thickness(0, 0, 0, 8),
            CornerRadius = new CornerRadius(9),
            Background = new SolidColorBrush(Color.FromRgb(0x28, 0x28, 0x2F)),
            BorderBrush = new SolidColorBrush(Color.FromRgb(0x3A, 0x3A, 0x48)),
            BorderThickness = new Thickness(1), Cursor = Cursors.Hand
        };
        var grid = new Grid();
        grid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        grid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });

        // Text panel
        var textStack = new StackPanel
        {
            Orientation = Orientation.Vertical, VerticalAlignment = VerticalAlignment.Center,
            Margin = new Thickness(14, 0, 8, 0)
        };
        textStack.Children.Add(new TextBlock
        {
            Text = name, FontSize = 13, FontWeight = FontWeights.Medium,
            Foreground = new SolidColorBrush(Color.FromRgb(0x30, 0xD1, 0x58)),
            TextTrimming = TextTrimming.CharacterEllipsis
        });
        textStack.Children.Add(new TextBlock
        {
            Text = "Custom look", FontSize = 11,
            Foreground = new SolidColorBrush(Color.FromRgb(0x8E, 0x8E, 0xA0)),
        });
        Grid.SetColumn(textStack, 0);
        grid.Children.Add(textStack);

        // Delete button — small red × on the right
        var delBorder = new Border
        {
            Width = 24, Height = 24, CornerRadius = new CornerRadius(5),
            Background = new SolidColorBrush(Color.FromRgb(0x3A, 0x1A, 0x1A)),
            Margin = new Thickness(0, 0, 12, 0), Cursor = Cursors.Hand,
            VerticalAlignment = VerticalAlignment.Center,
            Child = new TextBlock
            {
                Text = "✕", FontSize = 10, FontWeight = FontWeights.Bold,
                Foreground = new SolidColorBrush(Color.FromRgb(0xFF, 0x45, 0x3A)),
                HorizontalAlignment = HorizontalAlignment.Center,
                VerticalAlignment   = VerticalAlignment.Center
            },
            ToolTip = "Delete look"
        };
        delBorder.MouseLeftButtonUp += (_, e) => { e.Handled = true; onDelete(); };
        delBorder.MouseEnter += (_, _) => delBorder.Background = new SolidColorBrush(Color.FromRgb(0x55, 0x20, 0x20));
        delBorder.MouseLeave += (_, _) => delBorder.Background = new SolidColorBrush(Color.FromRgb(0x3A, 0x1A, 0x1A));
        Grid.SetColumn(delBorder, 1);
        grid.Children.Add(delBorder);

        outer.Child = grid;

        // Hover effect on the row
        outer.MouseEnter += (_, _) => outer.Background = new SolidColorBrush(Color.FromRgb(0x32, 0x32, 0x3A));
        outer.MouseLeave += (_, _) => outer.Background = new SolidColorBrush(Color.FromRgb(0x28, 0x28, 0x2F));
        // Left click on row = apply (× button handles its own click and stops propagation)
        outer.MouseLeftButtonUp += (_, e) => { if (!e.Handled) onApply(); };

        return outer;
    }

    private void AddCancelButton(Window dlg, StackPanel stack)
    {
        var cancel = new Button
        {
            Content = "Close", Width = 80, Height = 32,
            HorizontalAlignment = HorizontalAlignment.Right,
            Foreground = new SolidColorBrush(Color.FromRgb(0x8E, 0x8E, 0xA0)),
            FontSize = 12, Cursor = Cursors.Hand, Margin = new Thickness(0, 6, 0, 0), Background = Brushes.Transparent
        };
        var tpl = new ControlTemplate(typeof(Button));
        var bd  = new FrameworkElementFactory(typeof(Border));
        bd.SetValue(Border.BackgroundProperty, Brushes.Transparent);
        bd.SetValue(Border.CornerRadiusProperty, new CornerRadius(7));
        var cp = new FrameworkElementFactory(typeof(ContentPresenter));
        cp.SetValue(ContentPresenter.HorizontalAlignmentProperty, HorizontalAlignment.Center);
        cp.SetValue(ContentPresenter.VerticalAlignmentProperty, VerticalAlignment.Center);
        bd.AppendChild(cp);
        tpl.VisualTree = bd;
        cancel.Template = tpl;
        cancel.Click += (_, _) => dlg.Close();
        stack.Children.Add(cancel);
    }

    // Minimal modal text-input prompt. Returns null if cancelled.
    private string? PromptText(string title, string label, string defaultValue)
    {
        var (dlg, stack) = BuildCardDialog(title, Color.FromRgb(0xBF, 0x5A, 0xF2));
        stack.Children.Add(new TextBlock
        {
            Text = label, Foreground = new SolidColorBrush(Color.FromRgb(0xE5, 0xE5, 0xEA)),
            FontSize = 12, Margin = new Thickness(2, 0, 0, 6)
        });
        var input = new TextBox
        {
            Text = defaultValue, FontSize = 13, Height = 32, Padding = new Thickness(8, 0, 8, 0),
            Background = new SolidColorBrush(Color.FromRgb(0x28, 0x28, 0x2F)),
            Foreground = Brushes.White, CaretBrush = Brushes.White,
            BorderBrush = new SolidColorBrush(Color.FromRgb(0x0A, 0x84, 0xFF)),
            BorderThickness = new Thickness(1), VerticalContentAlignment = VerticalAlignment.Center,
            Margin = new Thickness(0, 0, 0, 12)
        };
        stack.Children.Add(input);

        string? result = null;
        var ok = new Button
        {
            Content = "Save", Width = 90, Height = 34, HorizontalAlignment = HorizontalAlignment.Right,
            Cursor = Cursors.Hand, Foreground = Brushes.White, FontSize = 13, FontWeight = FontWeights.Medium
        };
        var tpl = new ControlTemplate(typeof(Button));
        var bd  = new FrameworkElementFactory(typeof(Border));
        bd.SetValue(Border.BackgroundProperty, new SolidColorBrush(Color.FromRgb(0x0A, 0x84, 0xFF)));
        bd.SetValue(Border.CornerRadiusProperty, new CornerRadius(8));
        var cp = new FrameworkElementFactory(typeof(ContentPresenter));
        cp.SetValue(ContentPresenter.HorizontalAlignmentProperty, HorizontalAlignment.Center);
        cp.SetValue(ContentPresenter.VerticalAlignmentProperty, VerticalAlignment.Center);
        bd.AppendChild(cp);
        tpl.VisualTree = bd;
        ok.Template = tpl;
        void Commit() { result = input.Text; dlg.Close(); }
        ok.Click += (_, _) => Commit();
        input.KeyDown += (_, ke) => { if (ke.Key == Key.Enter) { Commit(); ke.Handled = true; } };
        stack.Children.Add(ok);

        dlg.Loaded += (_, _) => { input.Focus(); input.SelectAll(); };
        dlg.ShowDialog();
        return result;
    }
}
