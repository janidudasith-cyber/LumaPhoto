; LumaPhoto Inno Setup installer script
; Download Inno Setup free from: https://jrsoftware.org/isinfo.php
; Then open this file in Inno Setup Compiler and click Build > Compile

#define MyAppName    "LumaPhoto"
#define MyAppVersion "1.3"
#define MyAppPublisher "LumaPhoto"
#define MyAppURL     "https://github.com/janidudasith-cyber/LumaPhoto"
#define MyAppExe     "LumaPhoto.exe"
#define SourceDir    "publish"

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
Source: "{#SourceDir}\{#MyAppExe}";         DestDir: "{app}"; Flags: ignoreversion
#if FileExists(SourceDir + "\enhancer_params.onnx") && FileExists(SourceDir + "\enhancer_params.json")
Source: "{#SourceDir}\enhancer_params.onnx"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\enhancer_params.json"; DestDir: "{app}"; Flags: ignoreversion
#endif
#if FileExists(SourceDir + "\fivek_expert_c.onnx") && FileExists(SourceDir + "\fivek_expert_c.json")
Source: "{#SourceDir}\fivek_expert_c.onnx"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\fivek_expert_c.json"; DestDir: "{app}"; Flags: ignoreversion
#endif
#if FileExists(SourceDir + "\fivek_expert_a.onnx") && FileExists(SourceDir + "\fivek_expert_a.json")
Source: "{#SourceDir}\fivek_expert_a.onnx"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\fivek_expert_a.json"; DestDir: "{app}"; Flags: ignoreversion
#endif
#if FileExists(SourceDir + "\fivek_expert_e.onnx") && FileExists(SourceDir + "\fivek_expert_e.json")
Source: "{#SourceDir}\fivek_expert_e.onnx"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\fivek_expert_e.json"; DestDir: "{app}"; Flags: ignoreversion
#endif

[Icons]
Name: "{group}\{#MyAppName}";       Filename: "{app}\{#MyAppExe}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExe}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
