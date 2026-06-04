using System.Windows;
using System.Windows.Threading;

namespace LumaPhoto;

public partial class App : Application
{
    protected override void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);

        DispatcherUnhandledException += OnUnhandled;
        AppDomain.CurrentDomain.UnhandledException += (_, ex) =>
        {
            var msg = ex.ExceptionObject is Exception err
                ? Unwrap(err) : ex.ExceptionObject?.ToString() ?? "Unknown error";
            MessageBox.Show(msg, "Luma — Startup Error", MessageBoxButton.OK, MessageBoxImage.Error);
        };
    }

    private void OnUnhandled(object sender, DispatcherUnhandledExceptionEventArgs e)
    {
        MessageBox.Show(Unwrap(e.Exception), "Luma — Error",
            MessageBoxButton.OK, MessageBoxImage.Error);
        e.Handled = true;
    }

    private static string Unwrap(Exception ex)
    {
        // Drill into InnerException to get the real cause
        var inner = ex;
        while (inner.InnerException != null) inner = inner.InnerException;
        return $"{inner.GetType().Name}:\n{inner.Message}\n\n--- Stack ---\n{inner.StackTrace}";
    }
}
