; LumaPhoto Inno Setup installer script
; Download Inno Setup free from: https://jrsoftware.org/isinfo.php
; Then open this file in Inno Setup Compiler and click Build > Compile

#ifndef MyAppName
  #define MyAppName "LumaPhoto"
#endif
#ifndef MyAppVersion
  #define MyAppVersion "1.5"
#endif
#ifndef MyAppPublisher
  #define MyAppPublisher "LumaPhoto"
#endif
#ifndef MyAppURL
  #define MyAppURL ""
#endif
#ifndef MyAppExe
  #define MyAppExe "LumaPhoto.exe"
#endif
#ifndef BuildOutputDir
  #define BuildOutputDir "publish"
#endif

[Setup]
AppId={{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
LicenseFile=
OutputDir=installer_output
OutputBaseFilename=LumaPhoto-Setup-v{#MyAppVersion}
SetupIconFile=LumaPhoto\LumaPhoto.ico
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
RestartApplications=yes
MinVersion=10.0
ArchitecturesInstallIn64BitMode=x64
ArchitecturesAllowed=x64
UninstallDisplayIcon={app}\{#MyAppExe}
UninstallDisplayName={#MyAppName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#BuildOutputDir}\{#MyAppExe}";    DestDir: "{app}"; Flags: ignoreversion
; Background-removal model. Only the ~5 MB lite model ships; the heavier
; group-photo / hair models are an in-app on-demand download (ModelDownloader)
; into the current user's LocalAppData folder, where EnsureRemover() auto-selects
; the best one present.
; Kept a loose file (not bundled into the single-file exe) so
; AppContext.BaseDirectory resolution finds it at Assets\Models.
#if FileExists(BuildOutputDir + "\Assets\Models\u2netp.onnx")
Source: "{#BuildOutputDir}\Assets\Models\u2netp.onnx"; DestDir: "{app}\Assets\Models"; Flags: ignoreversion
#endif
; FiveK/PPR10K-trained enhancement weights are intentionally never included in
; an installer. Their training-data rights cover research use only.
; The licence-clean build also compiles out their loader paths.
Source: "THIRD_PARTY_NOTICES.txt"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}";       Filename: "{app}\{#MyAppExe}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExe}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
