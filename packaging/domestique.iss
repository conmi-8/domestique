; Windows installer for fork channel builds (docs/fork-updates.md), compiled
; by .github/workflows/fork-channel.yml from the PyInstaller folder
; dist\Domestique. Per-user install: no administrator prompt, and the app can
; replace itself (channel_update). Rider data lives in %USERPROFILE%\.domestique,
; outside {app}: neither an update nor the uninstaller touches it.
;
;   ISCC.exe /DAppVersion=3.13.1 packaging\domestique.iss  ->  dist\Domestique-Setup.exe
;
; A silent install (/VERYSILENT, as the in-app update runs it) also starts the
; app again: the [Run] entry has no skipifsilent.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
; Fixed for good: the AppId is what makes a new build an upgrade of the old.
AppId={{11D4223A-52E3-4B0D-AADE-35CF62004BCD}
AppName=Domestique
AppVersion={#AppVersion}
AppPublisher=Domestique (fork build)
DefaultDirName={localappdata}\Programs\Domestique
DisableDirPage=yes
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\dist
OutputBaseFilename=Domestique-Setup
SetupIconFile=..\assets\icon.ico
UninstallDisplayIcon={app}\Domestique.exe
Compression=lzma2
SolidCompression=yes
; Close a running Domestique without asking, and do not restart it through the
; Restart Manager: [Run] starts the new build.
CloseApplications=force
RestartApplications=no
WizardStyle=modern
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64

[Tasks]
Name: desktopicon; Description: "Create a desktop shortcut"; Flags: unchecked

[InstallDelete]
; The previous build's libraries: a module dropped from the new build must not
; linger next to it.
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "..\dist\Domestique\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\Domestique"; Filename: "{app}\Domestique.exe"
Name: "{autodesktop}\Domestique"; Filename: "{app}\Domestique.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\Domestique.exe"; Description: "Launch Domestique"; Flags: nowait postinstall
