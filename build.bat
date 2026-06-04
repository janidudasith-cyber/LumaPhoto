@echo off
echo ╔══════════════════════════════════════╗
echo ║        Luma Photo Editor Builder       ║
echo ╚══════════════════════════════════════╝
echo.

where dotnet >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] .NET SDK not found.
    echo Please install .NET 8 SDK from: https://dotnet.microsoft.com/download
    pause
    exit /b 1
)

echo [1/3] Restoring packages...
dotnet restore LumaPhoto\LumaPhoto.csproj
if %ERRORLEVEL% NEQ 0 ( echo Restore failed. & pause & exit /b 1 )

echo [2/3] Building release...
dotnet build LumaPhoto\LumaPhoto.csproj -c Release --nologo -v minimal
if %ERRORLEVEL% NEQ 0 ( echo Build failed. & pause & exit /b 1 )

echo [3/3] Publishing single-file executable...
dotnet publish LumaPhoto\LumaPhoto.csproj -c Release -r win-x64 --self-contained true ^
    -p:PublishSingleFile=true -p:IncludeNativeLibrariesForSelfExtract=true ^
    -o publish --nologo -v minimal
if %ERRORLEVEL% NEQ 0 ( echo Publish failed. & pause & exit /b 1 )

echo.
echo ✅ Done! Executable at: publish\LumaPhoto.exe
echo.
set /p LAUNCH="Launch Luma now? (y/n): "
if /i "%LAUNCH%"=="y" start "" "publish\LumaPhoto.exe"
