$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $taskRoot
$taskResultPath = Join-Path $taskRoot 'results\steady_state\Tabular_MLP_New\training_result.json'
$taskProtocolPath = Join-Path $taskRoot 'results\steady_state\Tabular_MLP_New\training_protocol.json'
$taskBatchPath = Join-Path $taskRoot 'results\pgin\retrain_lhs10_23\seed42\_batch_run.json'
$taskSharedPath = Join-Path $taskRoot 'results\pgin\retrain_lhs10_23\_shared_source_reference.json'
$taskCollisionPath = Join-Path $taskRoot 'saved_models\lhs10_23_no_gain_seed42.pth'
$taskManifestPath = Join-Path $taskRoot 'configs\pgin\retrain_lhs10_23\config_manifest.json'
if (Microsoft.PowerShell.Management\Test-Path -LiteralPath $taskBatchPath) { throw 'Synthetic check requires no actual seed42 batch.' }
if (Microsoft.PowerShell.Management\Test-Path -LiteralPath (Join-Path $taskRoot 'results\pgin\retrain_lhs10_23\_shared_source_reference.json')) { throw 'Synthetic check requires no existing shared reference.' }
$taskLhs = @(Get-ChildItem -LiteralPath (Join-Path $taskRoot 'data\Claus_steady_state') -Filter 'lhs_generated_dynamic_ss_data*.xlsx' -File | Sort-Object Name)
$taskSources = [ordered]@{}
foreach ($f in $taskLhs) { $taskSources[$f.FullName] = (Microsoft.PowerShell.Utility\Get-FileHash -LiteralPath $f.FullName -Algorithm SHA256).Hash }
$taskProtocolJson = [ordered]@{ input_files = @($taskLhs | ForEach-Object { $_.FullName }); source_sha256 = $taskSources } | ConvertTo-Json -Depth 6
$taskProtocolHash = 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'
$taskResultHash = 'BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB'
$taskWeights = Join-Path $taskRoot 'saved_models\Tabular_MLP_5in_2out_QV.pth'
$taskMean = Join-Path $taskRoot 'results\steady_state\Tabular_MLP_New\zscore_mean.csv'
$taskStd = Join-Path $taskRoot 'results\steady_state\Tabular_MLP_New\zscore_std.csv'
$taskArtifactHashes = [ordered]@{
    'teacher.pth' = (Microsoft.PowerShell.Utility\Get-FileHash -LiteralPath $taskWeights).Hash
    'zscore_mean.csv' = (Microsoft.PowerShell.Utility\Get-FileHash -LiteralPath $taskMean).Hash
    'zscore_std.csv' = (Microsoft.PowerShell.Utility\Get-FileHash -LiteralPath $taskStd).Hash
    'training_protocol.json' = $taskProtocolHash
}
$taskResultJson = [ordered]@{ completed = $true; output_sha256 = $taskArtifactHashes } | ConvertTo-Json -Depth 5
$taskManifestHash = (Microsoft.PowerShell.Utility\Get-FileHash -LiteralPath $taskManifestPath).Hash
$taskPriorSources = [ordered]@{}
$taskPriorSources[$taskWeights] = $taskArtifactHashes['teacher.pth']
$taskPriorJson = [ordered]@{ status = 'failed'; seed = 42; teacher_and_source_sha256 = $taskPriorSources; config_manifest_sha256 = $taskManifestHash } | ConvertTo-Json -Depth 5
$taskVirtualBatch = $false
$taskVirtualCollision = $false

function global:Test-Path {
    param([string]$LiteralPath, [string]$PathType)
    if ($LiteralPath -in @($taskResultPath, $taskProtocolPath)) { return $true }
    if ($LiteralPath -eq $taskBatchPath -and $taskVirtualBatch) { return $true }
    if ($LiteralPath -eq $taskCollisionPath -and $taskVirtualCollision) { return $true }
    Microsoft.PowerShell.Management\Test-Path @PSBoundParameters
}
function global:Get-Content {
    param([string]$LiteralPath, [switch]$Raw, [string]$Encoding)
    if ($LiteralPath -eq $taskResultPath) { return $taskResultJson }
    if ($LiteralPath -eq $taskProtocolPath) { return $taskProtocolJson }
    if ($LiteralPath -eq $taskBatchPath -and $taskVirtualBatch) { return $taskPriorJson }
    Microsoft.PowerShell.Management\Get-Content @PSBoundParameters
}
function global:Get-FileHash {
    param([string]$LiteralPath, [string]$Algorithm)
    if ($LiteralPath -eq $taskProtocolPath) { return [pscustomobject]@{ Hash = $taskProtocolHash } }
    if ($LiteralPath -eq $taskResultPath) { return [pscustomobject]@{ Hash = $taskResultHash } }
    Microsoft.PowerShell.Utility\Get-FileHash @PSBoundParameters
}

& .\scripts\training\run_retrain_lhs10_23.ps1 -Seed 42 -CheckOnly | Out-Null
Write-Output 'Synthetic completed ANN / matching sources: full-batch preflight passed'
$taskVirtualBatch = $true
& .\scripts\training\run_retrain_lhs10_23.ps1 -Seed 42 -OnlyAlias ss2p5_only -CheckOnly | Out-Null
Write-Output 'Synthetic failed batch / pending SS-only: single-model preflight passed'
$taskVirtualBatch = $false
$taskResultJson = [ordered]@{ completed = $false; output_sha256 = $taskArtifactHashes } | ConvertTo-Json -Depth 5
try {
    & .\scripts\training\run_retrain_lhs10_23.ps1 -Seed 42 -CheckOnly | Out-Null
    throw 'Unexpected preflight acceptance'
} catch {
    if ($_.Exception.Message -eq 'Unexpected preflight acceptance' -or $_.Exception.Message -notlike '*not complete*') { throw }
    Write-Output 'Incomplete ANN rejected'
}
$taskArtifactHashes['teacher.pth'] = '0000000000000000000000000000000000000000000000000000000000000000'
$taskResultJson = [ordered]@{ completed = $true; output_sha256 = $taskArtifactHashes } | ConvertTo-Json -Depth 5
try {
    & .\scripts\training\run_retrain_lhs10_23.ps1 -Seed 42 -CheckOnly | Out-Null
    throw 'Unexpected preflight acceptance'
} catch {
    if ($_.Exception.Message -eq 'Unexpected preflight acceptance' -or $_.Exception.Message -notlike '*does not match*') { throw }
    Write-Output 'Mismatched installed teacher rejected'
}
$taskArtifactHashes['teacher.pth'] = (Microsoft.PowerShell.Utility\Get-FileHash -LiteralPath $taskWeights).Hash
$taskResultJson = [ordered]@{ completed = $true; output_sha256 = $taskArtifactHashes } | ConvertTo-Json -Depth 5
$taskSources[$taskLhs[0].FullName] = '0000000000000000000000000000000000000000000000000000000000000000'
$taskProtocolJson = [ordered]@{ input_files = @($taskLhs | ForEach-Object { $_.FullName }); source_sha256 = $taskSources } | ConvertTo-Json -Depth 6
try {
    & .\scripts\training\run_retrain_lhs10_23.ps1 -Seed 42 -CheckOnly | Out-Null
    throw 'Unexpected preflight acceptance'
} catch {
    if ($_.Exception.Message -eq 'Unexpected preflight acceptance' -or $_.Exception.Message -notlike '*LHS source differs*') { throw }
    Write-Output 'Mismatched LHS source rejected'
}
$taskSources[$taskLhs[0].FullName] = (Microsoft.PowerShell.Utility\Get-FileHash -LiteralPath $taskLhs[0].FullName).Hash
$taskProtocolJson = [ordered]@{ input_files = @($taskLhs | ForEach-Object { $_.FullName }); source_sha256 = $taskSources } | ConvertTo-Json -Depth 6
$taskVirtualCollision = $true
try {
    & .\scripts\training\run_retrain_lhs10_23.ps1 -Seed 42 -CheckOnly | Out-Null
    throw 'Unexpected preflight acceptance'
} catch {
    if ($_.Exception.Message -eq 'Unexpected preflight acceptance' -or $_.Exception.Message -notlike '*Refusing to overwrite*') { throw }
    Write-Output 'Existing checkpoint rejected'
}
