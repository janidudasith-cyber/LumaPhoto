using Microsoft.Win32;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;

namespace LumaPhoto;

public partial class MainWindow
{
    private bool _frameLayerVisible = true;
    private bool _watermarkLayerVisible = true;
    private bool _markupLayerVisible = true;
    private System.Windows.Threading.DispatcherTimer? _designRefreshTimer;
    private bool _layersRefreshPending;

    private string FrameStyle =>
        ((FrameStyleCombo?.SelectedItem as ComboBoxItem)?.Tag as string) ?? "None";

    private string WatermarkPosition =>
        ((WatermarkPositionCombo?.SelectedItem as ComboBoxItem)?.Tag as string) ?? "BottomRight";

    private bool HasFrame =>
        _frameLayerVisible && FrameStyle != "None" && FrameSizeSlider.Value > 0;

    private bool HasWatermark =>
        _watermarkLayerVisible && !string.IsNullOrWhiteSpace(WatermarkTextBox.Text);

    private bool ShouldExportMarkup =>
        _markupLayerVisible && _markupStrokes.Count > 0;

    private void Frame_Changed(object sender, RoutedEventArgs e)
    {
        if (FrameSizeLabel != null)
            FrameSizeLabel.Text = $"{FrameSizeSlider.Value:0}";
        ScheduleDesignRefresh(updateLayers: true);
    }

    private void Watermark_Changed(object sender, RoutedEventArgs e)
    {
        if (WatermarkOpacityLabel != null)
            WatermarkOpacityLabel.Text = $"{WatermarkOpacitySlider.Value:0}";
        if (WatermarkSizeLabel != null)
            WatermarkSizeLabel.Text = $"{WatermarkSizeSlider.Value:0}";
        ScheduleDesignRefresh(updateLayers: true);
    }

    private void ScheduleDesignRefresh(bool updateLayers)
    {
        _layersRefreshPending |= updateLayers;

        if (_designRefreshTimer == null)
        {
            _designRefreshTimer = new System.Windows.Threading.DispatcherTimer
            {
                Interval = TimeSpan.FromMilliseconds(55)
            };
            _designRefreshTimer.Tick += (_, _) =>
            {
                _designRefreshTimer.Stop();
                RefreshDesignOverlay();
                if (_layersRefreshPending || LayersPanel?.Visibility == Visibility.Visible)
                    UpdateLayerList();
                _layersRefreshPending = false;
            };
        }

        _designRefreshTimer.Stop();
        _designRefreshTimer.Start();
    }

    private void RefreshDesignOverlay()
    {
        if (DesignCanvas == null) return;

        DesignCanvas.Children.Clear();
        if (!_imageLoaded || (!HasFrame && !HasWatermark))
        {
            DesignCanvas.Visibility = Visibility.Collapsed;
            return;
        }

        var bounds = GetImageDisplayBounds();
        if (bounds.Width <= 1 || bounds.Height <= 1)
        {
            DesignCanvas.Visibility = Visibility.Collapsed;
            return;
        }

        DesignCanvas.Width = StageGrid.ActualWidth;
        DesignCanvas.Height = StageGrid.ActualHeight;

        if (HasFrame)
            AddFramePreview(bounds);
        if (HasWatermark)
            AddWatermarkPreview(bounds);

        DesignCanvas.Visibility = Visibility.Visible;
    }

    private void AddFramePreview(Rect bounds)
    {
        double t = FrameThickness(bounds.Width, bounds.Height);
        Brush brush = FrameBrush();

        if (FrameStyle == "Shadow")
        {
            var shade = new SolidColorBrush(Color.FromArgb(70, 0, 0, 0));
            AddFrameStrip(bounds.X, bounds.Y, bounds.Width, t, shade);
            AddFrameStrip(bounds.X, bounds.Bottom - t, bounds.Width, t, shade);
            AddFrameStrip(bounds.X, bounds.Y, t, bounds.Height, shade);
            AddFrameStrip(bounds.Right - t, bounds.Y, t, bounds.Height, shade);
            return;
        }

        if (FrameStyle == "RoundedWhite")
        {
            var rounded = new Rectangle
            {
                Width = Math.Max(0, bounds.Width - t),
                Height = Math.Max(0, bounds.Height - t),
                Stroke = brush,
                StrokeThickness = t,
                RadiusX = Math.Max(10, t * 1.6),
                RadiusY = Math.Max(10, t * 1.6)
            };
            Canvas.SetLeft(rounded, bounds.X + t / 2);
            Canvas.SetTop(rounded, bounds.Y + t / 2);
            DesignCanvas.Children.Add(rounded);
            return;
        }

        if (FrameStyle == "Double")
        {
            double ot = Math.Max(2, t * 0.55), it = Math.Max(1.5, t * 0.16);
            AddFrameStrip(bounds.X, bounds.Y, bounds.Width, ot, brush);
            AddFrameStrip(bounds.X, bounds.Bottom - ot, bounds.Width, ot, brush);
            AddFrameStrip(bounds.X, bounds.Y, ot, bounds.Height, brush);
            AddFrameStrip(bounds.Right - ot, bounds.Y, ot, bounds.Height, brush);
            var innerLine = new Rectangle
            {
                Width = Math.Max(0, bounds.Width - 2 * t),
                Height = Math.Max(0, bounds.Height - 2 * t),
                Stroke = brush, StrokeThickness = it
            };
            Canvas.SetLeft(innerLine, bounds.X + t);
            Canvas.SetTop(innerLine, bounds.Y + t);
            DesignCanvas.Children.Add(innerLine);
            return;
        }

        AddFrameStrip(bounds.X, bounds.Y, bounds.Width, t, brush);
        AddFrameStrip(bounds.X, bounds.Y + bounds.Height - t, bounds.Width, t, brush);
        AddFrameStrip(bounds.X, bounds.Y, t, bounds.Height, brush);
        AddFrameStrip(bounds.X + bounds.Width - t, bounds.Y, t, bounds.Height, brush);

        if (FrameStyle == "Polaroid")
            AddFrameStrip(bounds.X, bounds.Y + bounds.Height - t * 2.4, bounds.Width, t * 1.4, brush);
    }

    private void AddFrameStrip(double x, double y, double w, double h, Brush brush)
    {
        var strip = new Rectangle { Width = Math.Max(0, w), Height = Math.Max(0, h), Fill = brush };
        Canvas.SetLeft(strip, x);
        Canvas.SetTop(strip, y);
        DesignCanvas.Children.Add(strip);
    }

    private void AddWatermarkPreview(Rect bounds)
    {
        var text = WatermarkTextBox.Text.Trim();
        if (text.Length == 0) return;

        var tb = new TextBlock
        {
            Text = text,
            FontFamily = new FontFamily("Segoe UI"),
            FontWeight = FontWeights.SemiBold,
            FontSize = WatermarkSizeSlider.Value,
            Foreground = Brushes.White,
            Opacity = WatermarkOpacitySlider.Value / 100.0,
            Effect = new System.Windows.Media.Effects.DropShadowEffect
            {
                Color = Colors.Black,
                Opacity = 0.8,
                BlurRadius = 8,
                ShadowDepth = 1
            }
        };

        tb.Measure(new Size(double.PositiveInfinity, double.PositiveInfinity));
        var p = PositionWatermark(bounds, tb.DesiredSize.Width, tb.DesiredSize.Height, 16);
        Canvas.SetLeft(tb, p.X);
        Canvas.SetTop(tb, p.Y);
        DesignCanvas.Children.Add(tb);
    }

    private Brush FrameBrush() => FrameStyle switch
    {
        "Black" => new SolidColorBrush(Color.FromRgb(18, 18, 20)),
        "FilmBlack" => new SolidColorBrush(Color.FromRgb(8, 8, 10)),
        "WarmMat" => new SolidColorBrush(Color.FromRgb(236, 225, 207)),
        "VignetteEdge" => new SolidColorBrush(Color.FromArgb(120, 0, 0, 0)),
        "Gold" => new LinearGradientBrush(Color.FromRgb(0xE6, 0xC4, 0x5C), Color.FromRgb(0xA8, 0x7A, 0x1E), 45),
        "Silver" => new LinearGradientBrush(Color.FromRgb(0xE8, 0xE8, 0xEC), Color.FromRgb(0x96, 0x96, 0xA0), 45),
        "Walnut" => new LinearGradientBrush(Color.FromRgb(0x6B, 0x44, 0x26), Color.FromRgb(0x4A, 0x2E, 0x18), 45),
        "Navy" => new SolidColorBrush(Color.FromRgb(0x14, 0x2A, 0x4A)),
        "Forest" => new SolidColorBrush(Color.FromRgb(0x1E, 0x3D, 0x2A)),
        "Burgundy" => new SolidColorBrush(Color.FromRgb(0x5C, 0x1A, 0x24)),
        _ => Brushes.White
    };

    private double FrameThickness(double w, double h)
    {
        double baseThickness = Math.Max(2, Math.Min(w, h) * FrameSizeSlider.Value / 100.0);
        return FrameStyle switch
        {
            "ThinWhite" => Math.Clamp(baseThickness, 2, 8),
            "FilmBlack" => baseThickness * 1.2,
            "Polaroid" => baseThickness * 1.15,
            _ => baseThickness
        };
    }

    private Point PositionWatermark(Rect bounds, double w, double h, double margin)
    {
        double x = WatermarkPosition.Contains("Right") ? bounds.Right - w - margin : bounds.X + margin;
        double y = WatermarkPosition.Contains("Bottom") ? bounds.Bottom - h - margin : bounds.Y + margin;
        if (WatermarkPosition == "Center")
        {
            x = bounds.X + (bounds.Width - w) / 2;
            y = bounds.Y + (bounds.Height - h) / 2;
        }
        return new Point(Math.Max(bounds.X + 2, x), Math.Max(bounds.Y + 2, y));
    }

    private BitmapSource CompositeDesignOnExport(BitmapSource src)
    {
        if (!HasFrame && !HasWatermark) return src;

        int pw = src.PixelWidth;
        int ph = src.PixelHeight;
        var dv = new DrawingVisual();

        using (var dc = dv.RenderOpen())
        {
            dc.DrawImage(src, new Rect(0, 0, pw, ph));
            if (HasFrame)
                DrawFrame(dc, pw, ph);
            if (HasWatermark)
                DrawWatermark(dc, pw, ph);
        }

        var rtb = new RenderTargetBitmap(pw, ph, 96, 96, PixelFormats.Pbgra32);
        rtb.Render(dv);
        rtb.Freeze();
        return rtb;
    }

    private void DrawFrame(DrawingContext dc, int pw, int ph)
    {
        double t = FrameThickness(pw, ph);
        Brush brush = FrameBrush();

        if (FrameStyle == "Shadow")
        {
            dc.DrawRectangle(new SolidColorBrush(Color.FromArgb(65, 0, 0, 0)), null, new Rect(0, 0, pw, t));
            dc.DrawRectangle(new SolidColorBrush(Color.FromArgb(65, 0, 0, 0)), null, new Rect(0, ph - t, pw, t));
            dc.DrawRectangle(new SolidColorBrush(Color.FromArgb(65, 0, 0, 0)), null, new Rect(0, 0, t, ph));
            dc.DrawRectangle(new SolidColorBrush(Color.FromArgb(65, 0, 0, 0)), null, new Rect(pw - t, 0, t, ph));
            return;
        }

        if (FrameStyle == "RoundedWhite")
        {
            dc.DrawRoundedRectangle(null, new Pen(brush, t), new Rect(t / 2, t / 2, pw - t, ph - t),
                Math.Max(14, t * 1.8), Math.Max(14, t * 1.8));
            return;
        }

        if (FrameStyle == "Double")
        {
            double ot = Math.Max(2, t * 0.55), it = Math.Max(1.5, t * 0.16);
            dc.DrawRectangle(brush, null, new Rect(0, 0, pw, ot));
            dc.DrawRectangle(brush, null, new Rect(0, ph - ot, pw, ot));
            dc.DrawRectangle(brush, null, new Rect(0, 0, ot, ph));
            dc.DrawRectangle(brush, null, new Rect(pw - ot, 0, ot, ph));
            dc.DrawRectangle(null, new Pen(brush, it), new Rect(t, t, pw - 2 * t, ph - 2 * t));
            return;
        }

        dc.DrawRectangle(brush, null, new Rect(0, 0, pw, t));
        dc.DrawRectangle(brush, null, new Rect(0, ph - t, pw, t));
        dc.DrawRectangle(brush, null, new Rect(0, 0, t, ph));
        dc.DrawRectangle(brush, null, new Rect(pw - t, 0, t, ph));
        if (FrameStyle == "Polaroid")
            dc.DrawRectangle(brush, null, new Rect(0, ph - t * 2.4, pw, t * 1.4));
    }

    private void DrawWatermark(DrawingContext dc, int pw, int ph)
    {
        string text = WatermarkTextBox.Text.Trim();
        if (text.Length == 0) return;

        double fontSize = Math.Max(12, WatermarkSizeSlider.Value * pw / 1200.0);
        byte alpha = (byte)Math.Clamp(WatermarkOpacitySlider.Value * 255.0 / 100.0, 10, 255);
        var face = new Typeface(new FontFamily("Segoe UI"), FontStyles.Normal, FontWeights.SemiBold, FontStretches.Normal);
        var ft = new FormattedText(text, System.Globalization.CultureInfo.InvariantCulture,
            FlowDirection.LeftToRight, face, fontSize,
            new SolidColorBrush(Color.FromArgb(alpha, 255, 255, 255)), 96);

        var bounds = new Rect(0, 0, pw, ph);
        double margin = Math.Max(18, pw * 0.018);
        Point p = PositionWatermark(bounds, ft.Width, ft.Height, margin);

        var shadow = new FormattedText(text, System.Globalization.CultureInfo.InvariantCulture,
            FlowDirection.LeftToRight, face, fontSize,
            new SolidColorBrush(Color.FromArgb((byte)Math.Min(180, (int)alpha), 0, 0, 0)), 96);
        dc.DrawText(shadow, new Point(p.X + 2, p.Y + 2));
        dc.DrawText(ft, p);
    }

    private void UpdateLayerList()
    {
        if (LayerList == null) return;
        LayerList.Children.Clear();

        LayerList.Children.Add(MakeLayerToggle("Watermark", LayerSummary(WatermarkTextBox.Text.Trim().Length > 0 ? WatermarkTextBox.Text.Trim() : "No text yet", _watermarkLayerVisible),
            _watermarkLayerVisible, v => { _watermarkLayerVisible = v; RefreshDesignOverlay(); UpdateLayerList(); }));
        LayerList.Children.Add(MakeLayerToggle("Markup", LayerSummary($"{_markupStrokes.Count} item(s)", _markupLayerVisible),
            _markupLayerVisible, v => { _markupLayerVisible = v; MarkupCanvas.Opacity = v ? 1.0 : 0.0; UpdateLayerList(); }));
        LayerList.Children.Add(MakeLayerToggle("Frame", LayerSummary(FrameStyle == "None" ? "No frame selected" : FrameStyle, _frameLayerVisible),
            _frameLayerVisible, v => { _frameLayerVisible = v; RefreshDesignOverlay(); UpdateLayerList(); }));
        LayerList.Children.Add(MakeLayerRow("Photo", "Base image - always visible/exported", true));
    }

    private static string LayerSummary(string detail, bool isOn) =>
        $"{(isOn ? "Visible + exported" : "Hidden")} - {detail}";

    private UIElement MakeLayerToggle(string title, string sub, bool isOn, Action<bool> onChange)
    {
        var row = MakeLayerRow(title, sub, isOn);
        if (row is Border border && border.Child is Grid grid && grid.Children[0] is CheckBox cb)
            cb.Checked += (_, _) => onChange(true);
        if (row is Border border2 && border2.Child is Grid grid2 && grid2.Children[0] is CheckBox cb2)
            cb2.Unchecked += (_, _) => onChange(false);
        return row;
    }

    private UIElement MakeLayerRow(string title, string sub, bool isOn)
    {
        var row = new Border
        {
            Background = new SolidColorBrush(Color.FromRgb(0x20, 0x20, 0x26)),
            BorderBrush = new SolidColorBrush(Color.FromRgb(0x32, 0x32, 0x3A)),
            BorderThickness = new Thickness(1),
            CornerRadius = new CornerRadius(8),
            Padding = new Thickness(10, 8, 10, 8),
            Margin = new Thickness(0, 0, 0, 8),
            Opacity = isOn ? 1.0 : 0.62
        };

        var grid = new Grid();
        grid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        grid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });

        var cb = new CheckBox
        {
            IsChecked = isOn,
            IsEnabled = title != "Photo",
            VerticalAlignment = VerticalAlignment.Center,
            Margin = new Thickness(0, 0, 10, 0)
        };
        Grid.SetColumn(cb, 0);
        grid.Children.Add(cb);

        var text = new StackPanel();
        text.Children.Add(new TextBlock
        {
            Text = title,
            Foreground = Brushes.White,
            FontSize = 13,
            FontWeight = FontWeights.SemiBold
        });
        text.Children.Add(new TextBlock
        {
            Text = sub,
            Foreground = new SolidColorBrush(Color.FromRgb(0x8E, 0x8E, 0xA0)),
            FontSize = 11,
            TextTrimming = TextTrimming.CharacterEllipsis
        });
        Grid.SetColumn(text, 1);
        grid.Children.Add(text);

        row.Child = grid;
        return row;
    }

    private static int CollageSlotCount(string layout) => layout switch
    {
        "Split" => 2, "Stack" => 3, "Feature" => 3, _ => 4
    };

    private async void Collage_Click(object sender, RoutedEventArgs e)
    {
        if (!_imageLoaded || _sourcePixels == null || sender is not Button btn) return;
        var layout = (string)btn.Tag;
        int slots  = CollageSlotCount(layout);

        var dlg = new OpenFileDialog
        {
            Title = $"Choose up to {slots - 1} photo(s) — your current photo fills the first slot",
            Filter = ImageOpenFilter,
            Multiselect = true
        };
        if (dlg.ShowDialog() != true) return;

        var slotImages = new BitmapSource?[slots];
        try
        {
            Mouse.OverrideCursor = Cursors.Wait;
            var pixels = (byte[])_sourcePixels.Clone();
            int w = _sourceW, h = _sourceH;
            var adj = _adj.Clone();
            var files = dlg.FileNames.Take(slots - 1).ToArray();

            var loaded = await Task.Run(() =>
            {
                var list = new List<BitmapSource> { ImageProcessor.Render(pixels, w, h, adj) };
                foreach (var f in files)
                {
                    var (px, iw, ih) = ImageProcessor.LoadImageFile(f);
                    list.Add(ImageProcessor.BufferToBitmap(px, iw, ih));
                }
                return list;
            });
            for (int i = 0; i < loaded.Count && i < slots; i++) slotImages[i] = loaded[i];
        }
        catch (Exception ex)
        {
            ShowToast("Collage Failed", ex.Message, success: false);
            return;
        }
        finally { Mouse.OverrideCursor = null; }

        ShowCollageArrangeDialog(layout, slotImages);
    }

    // Arrange dialog — a live mini-preview of the layout. Click two photos to
    // swap their slots; click an empty slot to add a photo into it.
    private void ShowCollageArrangeDialog(string layout, BitmapSource?[] slotImages)
    {
        int n = slotImages.Length;
        const double refW = 1600;
        double refH = layout == "Stack" ? refW * 1.35 : refW;
        var rects = CollageSlots(layout, (int)refW, (int)refH).ToArray();

        var (dlg, stack) = BuildCardDialog("Arrange Collage", Color.FromRgb(0x30, 0xD1, 0x58));

        stack.Children.Add(new TextBlock
        {
            Text = "Click a photo, then another to swap places. Click an empty slot to add a photo.",
            Foreground = new SolidColorBrush(Color.FromRgb(0x8E, 0x8E, 0xA0)),
            FontSize = 11, TextWrapping = TextWrapping.Wrap, Margin = new Thickness(2, 0, 0, 10)
        });

        double previewW = 296;
        double scale = previewW / refW;
        var canvas = new Canvas
        {
            Width = previewW, Height = refH * scale,
            Background = new SolidColorBrush(Color.FromRgb(0xF5, 0xF5, 0xF7)),
            HorizontalAlignment = HorizontalAlignment.Center
        };

        int selected = -1;
        var cells = new Border[n];
        Button createBtn = null!;   // assigned below, used by RefreshCells

        void RefreshCells()
        {
            for (int i = 0; i < n; i++)
            {
                cells[i].BorderBrush = i == selected
                    ? new SolidColorBrush(Color.FromRgb(0x0A, 0x84, 0xFF))
                    : new SolidColorBrush(Color.FromArgb(0x55, 0x00, 0x00, 0x00));
                if (slotImages[i] != null)
                    cells[i].Child = new Image { Source = slotImages[i], Stretch = Stretch.UniformToFill };
                else
                    cells[i].Child = new TextBlock
                    {
                        Text = "＋", FontSize = 20,
                        Foreground = new SolidColorBrush(Color.FromRgb(0x8E, 0x8E, 0xA0)),
                        HorizontalAlignment = HorizontalAlignment.Center,
                        VerticalAlignment   = VerticalAlignment.Center
                    };
            }
            createBtn.IsEnabled = slotImages.All(s => s != null);
            createBtn.Opacity   = createBtn.IsEnabled ? 1.0 : 0.45;
        }

        for (int i = 0; i < n; i++)
        {
            int idx = i;
            var r = rects[i];
            var cell = new Border
            {
                Width = r.Width * scale, Height = r.Height * scale,
                Background = new SolidColorBrush(Color.FromRgb(0xE2, 0xE2, 0xE6)),
                BorderThickness = new Thickness(2), CornerRadius = new CornerRadius(3),
                ClipToBounds = true, Cursor = Cursors.Hand
            };
            Canvas.SetLeft(cell, r.X * scale);
            Canvas.SetTop(cell,  r.Y * scale);
            cell.MouseLeftButtonUp += (_, ev) =>
            {
                ev.Handled = true;
                if (slotImages[idx] == null)
                {
                    var pick = new OpenFileDialog { Title = "Choose a photo for this slot", Filter = ImageOpenFilter };
                    if (pick.ShowDialog() == true)
                    {
                        try
                        {
                            Mouse.OverrideCursor = Cursors.Wait;
                            var (px, iw, ih) = ImageProcessor.LoadImageFile(pick.FileName);
                            slotImages[idx] = ImageProcessor.BufferToBitmap(px, iw, ih);
                        }
                        catch (Exception ex) { ShowToast("Could not open", ex.Message, success: false); }
                        finally { Mouse.OverrideCursor = null; }
                    }
                }
                else if (selected == -1)  selected = idx;
                else if (selected == idx) selected = -1;
                else
                {
                    (slotImages[selected], slotImages[idx]) = (slotImages[idx], slotImages[selected]);
                    selected = -1;
                }
                RefreshCells();
            };
            cells[i] = cell;
            canvas.Children.Add(cell);
        }
        stack.Children.Add(canvas);

        createBtn = new Button
        {
            Content = "Create Collage", Height = 36, Margin = new Thickness(0, 12, 0, 0),
            Foreground = Brushes.White, FontSize = 13, FontWeight = FontWeights.Medium,
            Cursor = Cursors.Hand
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
        createBtn.Template = tpl;
        createBtn.Click += (_, _) =>
        {
            dlg.Close();
            try
            {
                Mouse.OverrideCursor = Cursors.Wait;
                PushHistory();
                var collage = BuildCollageBitmap(layout, slotImages.Select(s => s!).ToArray());
                ReplaceSourceWithBitmap(collage, "collage");
                ShowToast("Collage Created", "Your collage is now the editable photo.");
            }
            catch (Exception ex) { ShowToast("Collage Failed", ex.Message, success: false); }
            finally { Mouse.OverrideCursor = null; }
        };
        stack.Children.Add(createBtn);

        RefreshCells();
        AddCancelButton(dlg, stack);
        dlg.ShowDialog();
    }

    private BitmapSource BuildCollageBitmap(string layout, BitmapSource[] images)
    {
        int outW = Math.Clamp(images[0].PixelWidth, 1200, 2400);
        int outH = layout == "Stack" ? (int)(outW * 1.35) : outW;
        var slots = CollageSlots(layout, outW, outH).ToArray();

        var dv = new DrawingVisual();
        RenderOptions.SetBitmapScalingMode(dv, BitmapScalingMode.HighQuality);
        using (var dc = dv.RenderOpen())
        {
            dc.DrawRectangle(new SolidColorBrush(Color.FromRgb(245, 245, 247)), null, new Rect(0, 0, outW, outH));
            for (int i = 0; i < slots.Length && i < images.Length; i++)
                DrawImageCover(dc, images[i], slots[i], 10);
        }

        var rtb = new RenderTargetBitmap(outW, outH, 96, 96, PixelFormats.Pbgra32);
        rtb.Render(dv);
        rtb.Freeze();
        return rtb;
    }

    private static IEnumerable<Rect> CollageSlots(string layout, int w, int h)
    {
        double g = Math.Max(16, w * 0.018);
        if (layout == "Split")
        {
            yield return new Rect(g, g, (w - 3 * g) / 2, h - 2 * g);
            yield return new Rect(2 * g + (w - 3 * g) / 2, g, (w - 3 * g) / 2, h - 2 * g);
            yield break;
        }

        if (layout == "Stack")
        {
            double sh = (h - 4 * g) / 3;
            yield return new Rect(g, g, w - 2 * g, sh);
            yield return new Rect(g, 2 * g + sh, w - 2 * g, sh);
            yield return new Rect(g, 3 * g + sh * 2, w - 2 * g, sh);
            yield break;
        }

        if (layout == "Feature")
        {
            double leftW = (w - 3 * g) * 0.62;
            double rightW = w - 3 * g - leftW;
            yield return new Rect(g, g, leftW, h - 2 * g);
            yield return new Rect(2 * g + leftW, g, rightW, (h - 3 * g) / 2);
            yield return new Rect(2 * g + leftW, 2 * g + (h - 3 * g) / 2, rightW, (h - 3 * g) / 2);
            yield break;
        }

        double cell = (w - 3 * g) / 2;
        yield return new Rect(g, g, cell, cell);
        yield return new Rect(2 * g + cell, g, cell, cell);
        yield return new Rect(g, 2 * g + cell, cell, cell);
        yield return new Rect(2 * g + cell, 2 * g + cell, cell, cell);
    }

    private static void DrawImageCover(DrawingContext dc, BitmapSource src, Rect slot, double radius)
    {
        double scale = Math.Max(slot.Width / src.PixelWidth, slot.Height / src.PixelHeight);
        double dw = src.PixelWidth * scale;
        double dh = src.PixelHeight * scale;
        var dest = new Rect(slot.X - (dw - slot.Width) / 2, slot.Y - (dh - slot.Height) / 2, dw, dh);

        dc.PushClip(new RectangleGeometry(slot, radius, radius));
        dc.DrawImage(src, dest);
        dc.Pop();
    }

    private void ReplaceSourceWithBitmap(BitmapSource bitmap, string name)
    {
        var converted = new FormatConvertedBitmap(bitmap, PixelFormats.Bgra32, null, 0);
        int stride = converted.PixelWidth * 4;
        byte[] pixels = new byte[stride * converted.PixelHeight];
        converted.CopyPixels(pixels, stride, 0);

        _sourcePixels = pixels;
        _sourceW = converted.PixelWidth;
        _sourceH = converted.PixelHeight;
        _originalPixels = (byte[])pixels.Clone();
        _originalW = _sourceW;
        _originalH = _sourceH;
        _fileName = name;
        _currentFilePath = null;
        FileMetaText.Text = $"{name}  -  {_sourceW} x {_sourceH}";

        _future.Clear();
        ResetAll();
        DoRender();
        UpdateLayerList();
    }
}
