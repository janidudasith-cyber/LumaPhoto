using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;

namespace LumaPhoto;

public partial class SliderRow : UserControl
{
    public static readonly DependencyProperty LabelProperty =
        DependencyProperty.Register("Label", typeof(string), typeof(SliderRow),
            new PropertyMetadata("", (d, e) => ((SliderRow)d).LabelText.Text = (string)e.NewValue));

    public static readonly DependencyProperty MinProperty =
        DependencyProperty.Register("Min", typeof(double), typeof(SliderRow),
            new PropertyMetadata(-100.0, (d, e) => ((SliderRow)d).ValueSlider.Minimum = (double)e.NewValue));

    public static readonly DependencyProperty MaxProperty =
        DependencyProperty.Register("Max", typeof(double), typeof(SliderRow),
            new PropertyMetadata(100.0, (d, e) => ((SliderRow)d).ValueSlider.Maximum = (double)e.NewValue));

    public static readonly DependencyProperty ValueProperty =
        DependencyProperty.Register("Value", typeof(double), typeof(SliderRow),
            new PropertyMetadata(0.0, (d, e) =>
            {
                var row = (SliderRow)d;
                row.ValueSlider.Value = (double)e.NewValue;
                row.ValueText.Text = ((int)(double)e.NewValue).ToString();
            }));

    public string Label  { get => (string)GetValue(LabelProperty);  set => SetValue(LabelProperty, value); }
    public double Min    { get => (double)GetValue(MinProperty);    set => SetValue(MinProperty, value); }
    public double Max    { get => (double)GetValue(MaxProperty);    set => SetValue(MaxProperty, value); }
    public double Value  { get => (double)GetValue(ValueProperty);  set => SetValue(ValueProperty, value); }

    public event EventHandler<double>? Changed;
    public event EventHandler? DragStarted;
    public event EventHandler? DragCompleted;
    public event EventHandler? CommitChange;

    public SliderRow()
    {
        InitializeComponent();
        ValueSlider.AddHandler(System.Windows.Controls.Primitives.Thumb.DragStartedEvent,
            new System.Windows.Controls.Primitives.DragStartedEventHandler(
                (_, _) => DragStarted?.Invoke(this, EventArgs.Empty)));
        ValueSlider.AddHandler(System.Windows.Controls.Primitives.Thumb.DragCompletedEvent,
            new System.Windows.Controls.Primitives.DragCompletedEventHandler(
                (_, _) => DragCompleted?.Invoke(this, EventArgs.Empty)));
    }

    private bool _suppressEvent;

    private void Slider_Changed(object sender, RoutedPropertyChangedEventArgs<double> e)
    {
        if (_suppressEvent) return;
        double v = Math.Round(e.NewValue, 1);
        if (!ValueText.IsFocused) ValueText.Text = v.ToString("0.0");
        Changed?.Invoke(this, v);
    }

    public void SetValueSilent(double v)
    {
        _suppressEvent = true;
        ValueSlider.Value = v;
        ValueText.Text = Math.Round(v, 1).ToString("0.0");
        _suppressEvent = false;
    }

    public void SetEnabled(bool on) => ValueSlider.IsEnabled = on;

    // ── Editable value label ──
    private void ValueText_PreviewMouseDown(object sender, MouseButtonEventArgs e)
    {
        if (!ValueText.IsFocused)
        {
            ValueText.Focus();
            e.Handled = true;
        }
    }

    private void ValueText_GotFocus(object sender, RoutedEventArgs e)
    {
        ValueText.Background       = new SolidColorBrush(Color.FromRgb(0x28, 0x28, 0x2F));
        ValueText.BorderThickness  = new Thickness(1);
        ValueText.BorderBrush      = new SolidColorBrush(Color.FromRgb(0x0A, 0x84, 0xFF));
        ValueText.Foreground       = new SolidColorBrush(Color.FromRgb(0xF0, 0xF0, 0xF2));
        // Defer SelectAll so the mouse-click doesn't immediately deselect
        Dispatcher.BeginInvoke(System.Windows.Threading.DispatcherPriority.Input,
            new Action(ValueText.SelectAll));
    }

    private void ValueText_LostFocus(object sender, RoutedEventArgs e)
    {
        ApplyTypedValue();
        ValueText.Background      = Brushes.Transparent;
        ValueText.BorderThickness = new Thickness(0);
        ValueText.Foreground      = new SolidColorBrush(Color.FromRgb(0x8E, 0x8E, 0xA0));
    }

    private void ValueText_KeyDown(object sender, KeyEventArgs e)
    {
        if (e.Key == Key.Enter || e.Key == Key.Tab)
        {
            ApplyTypedValue();
            Keyboard.ClearFocus();
            e.Handled = true;
        }
        else if (e.Key == Key.Escape)
        {
            ValueText.Text = Math.Round(ValueSlider.Value, 1).ToString("0.0");
            Keyboard.ClearFocus();
            e.Handled = true;
        }
    }

    private void ApplyTypedValue()
    {
        if (double.TryParse(ValueText.Text.Trim(),
                System.Globalization.NumberStyles.Float,
                System.Globalization.CultureInfo.CurrentCulture, out double parsed))
        {
            double clamped = Math.Clamp(Math.Round(parsed, 1), ValueSlider.Minimum, ValueSlider.Maximum);
            if (Math.Abs(clamped - ValueSlider.Value) > 0.05)
                CommitChange?.Invoke(this, EventArgs.Empty);
            _suppressEvent = true;
            ValueSlider.Value = clamped;
            _suppressEvent = false;
            ValueText.Text = clamped.ToString("0.0");
            Changed?.Invoke(this, clamped);
        }
        else
        {
            ValueText.Text = Math.Round(ValueSlider.Value, 1).ToString("0.0");
        }
    }
}
