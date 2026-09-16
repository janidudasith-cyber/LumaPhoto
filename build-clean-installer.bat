@echo off
setlocal

REM Builds the licence-clean LumaPhoto installer.
REM Uses StoreBuild=true, which excludes the FiveK/PPR10K-trained enhancement
REM models (their training data is research-use only) and compiles out the
REM GitHub self-updater. Auto Enhance still works via its rule-based path.

set "ISCC=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"

if not exist "%ISCC%" (
    echo ERROR: Inno Setup 6 was not found at:
    echo %ISCC%
    echo Install it from https://jrsoftware.org/isinfo.php and run this again.
    exit /b 1
)

echo.
echo [1/3] Publishing FiveK-free build...
dotnet publish LumaPhoto\LumaPhoto.csproj -c Release -r win-x64 --self-contained true -p:PublishSingleFile=true -p:IncludeNativeLibrariesForSelfExtract=true -p:StoreBuild=true -o publish-clean --nologo -v minimal
if errorlevel 1 exit /b 1

echo.
echo [2/3] Verifying payload...
if exist "publish-clean\fivek_expert_a.onnx" goto :fivek_found
if exist "publish-clean\fivek_expert_c.onnx" goto :fivek_found
if exist "publish-clean\fivek_expert_e.onnx" goto :fivek_found
if exist "publish-clean\enhancer_params.onnx" goto :fivek_found
if not exist "publish-clean\LumaPhoto.exe" goto :missing_app
if not exist "publish-clean\Assets\Models\u2netp.onnx" goto :missing_model
goto :verified

:fivek_found
echo ERROR: A research-trained model appeared in the payload. Installer was not built.
exit /b 1

:missing_app
echo ERROR: LumaPhoto.exe was not produced.
exit /b 1

:missing_model
echo ERROR: The bundled lite background-removal model was not produced.
exit /b 1

:verified
echo Payload verified: no research-trained models found.

echo.
echo [3/3] Building installer...
"%ISCC%" /DBuildOutputDir=publish-clean installer.iss
if errorlevel 1 exit /b 1

echo.
echo Done. The installer is in installer_output.
exit /b 0
