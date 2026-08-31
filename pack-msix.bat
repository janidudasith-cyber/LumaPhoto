@echo off
echo ================================================
echo   LumaPhoto MSIX Packager (Store build)
echo ================================================
echo.

where dotnet >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] .NET SDK not found.
    echo Please install .NET 8 SDK from: https://dotnet.microsoft.com/download
    pause
    exit /b 1
)

echo [1/5] Fetching MSIX packaging tools (makeappx) via NuGet, if not already cached...
dotnet restore tools\MsixTools\MsixTools.csproj --nologo -v minimal
if %ERRORLEVEL% NEQ 0 ( echo Restore failed. & pause & exit /b 1 )

set "MAKEAPPX="
for /f "delims=" %%F in ('dir /b /s /a-d "%USERPROFILE%\.nuget\packages\microsoft.windows.sdk.buildtools\makeappx.exe" 2^>nul ^| findstr /i "\\x64\\"') do set "MAKEAPPX=%%F"
if "%MAKEAPPX%"=="" (
    echo [ERROR] makeappx.exe not found in the NuGet cache after restore.
    echo Expected under %%USERPROFILE%%\.nuget\packages\microsoft.windows.sdk.buildtools\...\bin\...\x64\
    pause
    exit /b 1
)
echo   Using: %MAKEAPPX%

echo [2/5] Publishing the Store build (no FiveK/PPR10K models, no self-updater)...
dotnet publish LumaPhoto\LumaPhoto.csproj -c Release -r win-x64 --self-contained true ^
    -p:PublishSingleFile=true -p:StoreBuild=true -o publish-store --nologo -v minimal
if %ERRORLEVEL% NEQ 0 ( echo Publish failed. & pause & exit /b 1 )

echo [3/5] Staging package payload into Package\...
if not exist "Package\Assets\Models" mkdir "Package\Assets\Models"
copy /y "publish-store\LumaPhoto.exe" "Package\LumaPhoto.exe" >nul
copy /y "publish-store\Assets\Models\u2netp.onnx" "Package\Assets\Models\u2netp.onnx" >nul
if %ERRORLEVEL% NEQ 0 ( echo Staging failed. & pause & exit /b 1 )

echo [4/5] Packing LumaPhoto.msix...
if exist "LumaPhoto.msix" del "LumaPhoto.msix"
"%MAKEAPPX%" pack /d "Package" /p "LumaPhoto.msix" /o
if %ERRORLEVEL% NEQ 0 ( echo MSIX packaging failed. & pause & exit /b 1 )

echo.
echo [5/5] Done. Package at: LumaPhoto.msix
echo.
echo Before this is submittable:
echo   - Package\AppxManifest.xml still has PLACEHOLDER Identity Name/Publisher.
echo     Reserve the app name in Partner Center first, then paste in the real
echo     values it gives you (App management -^> App identity) and re-run this.
echo   - No signing needed for Store submission — the Store re-signs on ingestion.
echo   - To sideload-test locally before submitting, see DEVELOPMENT.md -^>
echo     "MSIX packaging" for the (separate, your-own-machine) steps to trust
echo     a test certificate or enable Developer Mode.
echo.
pause
