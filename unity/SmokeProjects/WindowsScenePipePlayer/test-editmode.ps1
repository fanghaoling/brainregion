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

$resultsPath = Join-Path $OutputDirectory 'editmode-results.xml'
$logPath = Join-Path $OutputDirectory 'unity-editmode-tests.log'
$arguments = @(
    '-batchmode',
    '-nographics',
    '-buildTarget', 'Win64',
    '-projectPath', "`"$projectRoot`"",
    '-runTests',
    '-testPlatform', 'EditMode',
    '-testResults', "`"$resultsPath`"",
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
    throw "Unity EditMode tests failed with exit code $($process.ExitCode). See $logPath"
}
if (-not (Test-Path -LiteralPath $resultsPath -PathType Leaf)) {
    throw "Unity reported success but did not create $resultsPath"
}

[xml]$results = Get-Content -LiteralPath $resultsPath -Raw
$failed = [int]$results.'test-run'.failed
if ($failed -ne 0) {
    throw "Unity EditMode tests reported $failed failures. See $resultsPath"
}

Write-Output $resultsPath
