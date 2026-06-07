namespace LumaPhoto;

/// <summary>
/// Single source of truth for the app version.
/// Bump <see cref="Current"/> for every release, then rebuild + run Inno Setup.
/// </summary>
public static class AppVersion
{
    public const string Current    = "1.1";
    public const string GitHubRepo = "janidudasith-cyber/LumaPhoto";
}
