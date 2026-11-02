param(
    [string]$Executable = "$PSScriptRoot\..\target\debug\bad_econ.exe",
    [string]$Output = "$PSScriptRoot\..\target\windows-viewer-profile.json",
    [ValidateRange(1, 10)][int]$Samples = 3,
    [ValidateRange(5, 60)][int]$SampleSeconds = 20,
    [ValidateRange(1, 1000000)][int]$Households = 100000
)

$ErrorActionPreference = 'Stop'
$binary = (Resolve-Path -LiteralPath $Executable).Path
$outputPath = [System.IO.Path]::GetFullPath($Output)
$null = New-Item -ItemType Directory -Path ([System.IO.Path]::GetDirectoryName($outputPath)) -Force
$viewerProcess = Start-Process -FilePath $binary -ArgumentList @('view', '--households', $Households) -WindowStyle Hidden -RedirectStandardError "$outputPath.log" -PassThru

try {
    Start-Sleep -Seconds 5
    $measurements = @(1..$Samples | ForEach-Object {
        $viewerProcess.Refresh()
        $cpuStart = $viewerProcess.TotalProcessorTime.TotalSeconds
        $elapsed = [System.Diagnostics.Stopwatch]::StartNew()
        Start-Sleep -Seconds $SampleSeconds
        $viewerProcess.Refresh()
        $wallSeconds = $elapsed.Elapsed.TotalSeconds
        $cpuSeconds = $viewerProcess.TotalProcessorTime.TotalSeconds - $cpuStart
        $measurement = [pscustomobject]@{
            sample = $_
            wall_seconds = $wallSeconds
            cpu_seconds = $cpuSeconds
            single_core_percent = 100 * $cpuSeconds / $wallSeconds
            working_set_bytes = $viewerProcess.WorkingSet64
            private_bytes = $viewerProcess.PrivateMemorySize64
        }
        Write-Host ("Sample {0}: {1:N3} CPU-seconds / {2:N3} seconds" -f $_, $cpuSeconds, $wallSeconds)
        $measurement
    })
    [pscustomobject]@{
        measured_at_utc = [DateTime]::UtcNow.ToString('o')
        executable_sha256 = (Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash.ToLowerInvariant()
        households = $Households
        state = 'paused at day 0; default chart; no keyboard or mouse input during sampling'
        method = 'Process.TotalProcessorTime deltas; five-second warmup; newly opened viewer window'
        samples = $measurements
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $outputPath
}
finally {
    $viewerProcess.Refresh()
    switch ($viewerProcess.HasExited) {
        $false { Stop-Process -Id $viewerProcess.Id }
    }
}
