@echo off
setlocal

REM Builds a portable, FiveK-free ZIP for direct download.
REM This does not use Inno Setup or MSIX.

echo.
echo [1/3] Publishing FiveK-free portable build...
dotnet publish LumaPhoto\LumaPhoto.csproj -c Release -r win-x64 --self-contained true -p:PublishSingleFile=true -p:IncludeNativeLibrariesForSelfExtract=true -p:StoreBuild=true -o publish-portable --nologo -v minimal
if errorlevel 1 exit /b 1

echo.
echo [2/3] Verifying portable payload...
if exist "publish-portable\fivek_expert_a.onnx" goto :fivek_found
if exist "publish-portable\fivek_expert_c.onnx" goto :fivek_found
if exist "publish-portable\fivek_expert_e.onnx" goto :fivek_found
if exist "publish-portable\enhancer_params.onnx" goto :fivek_found
if not exist "publish-portable\LumaPhoto.exe" goto :missing_app
if not exist "publish-portable\Assets\Models\u2netp.onnx" goto :missing_model
goto :verified

:fivek_found
echo ERROR: A research-trained model appeared in the portable payload. ZIP was not built.
exit /b 1

:missing_app
echo ERROR: LumaPhoto.exe was not produced.
exit /b 1

:missing_model
echo ERROR: The bundled lite background-removal model was not produced.
exit /b 1

:verified
echo Portable payload verified: no research-trained models found.

echo.
echo [3/3] Creating ZIP...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Stop'; $null = New-Item -ItemType Directory -Force -Path 'portable_output'; Compress-Archive -LiteralPath @('publish-portable\LumaPhoto.exe', 'publish-portable\Assets', 'THIRD_PARTY_NOTICES.txt', 'PORTABLE_README.txt') -DestinationPath 'portable_output\LumaPhoto-Portable.zip' -Force"
if errorlevel 1 exit /b 1

echo.
echo Done. The portable ZIP is in portable_output.
exit /b 0
