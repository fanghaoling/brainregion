param(
    [Parameter(Mandatory = $true)]
    [string]$UnityEditor,
    [string]$OutputDirectory
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $UnityEditor -PathType Leaf)) {
    throw "Unity Editor executable was not found: $UnityEditor"
}

$projectRoot = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $projectRoot '..\..\..\target\unity-scene-pipe-smoke'
}
$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

$playerPath = Join-Path $OutputDirectory 'BrainRegionScenePipeSmoke.exe'
$logPath = Join-Path $OutputDirectory 'unity-build.log'
$env:BRAINREGION_UNITY_SMOKE_OUTPUT = $playerPath

$arguments = @(
    '-batchmode',
    '-nographics',
    '-buildTarget', 'Win64',
    '-projectPath', "`"$projectRoot`"",
    '-executeMethod', 'BrainRegion.ScenePipeSmoke.Editor.ScenePipeSmokeBuild.BuildWindowsIl2Cpp',
    '-quit',
    '-logFile', "`"$logPath`""
)
$startParameters = @{
    FilePath = $UnityEditor
    ArgumentList = $arguments
    Wait = $true
    PassThru = $true
    WindowStyle = 'Hidden'
}
$process = Start-Process @startParameters
if ($process.ExitCode -ne 0) {
    throw "Unity IL2CPP build failed with exit code $($process.ExitCode). See $logPath"
}
if (-not (Test-Path -LiteralPath $playerPath -PathType Leaf)) {
    throw "Unity reported success but did not create $playerPath"
}

Write-Output $playerPath
