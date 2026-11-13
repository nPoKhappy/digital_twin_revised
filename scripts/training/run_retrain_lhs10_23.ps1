param(
    [ValidateSet(42, 43, 44)]
    [int]$Seed = 42,
    [string]$OnlyAlias,
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
Set-Location -LiteralPath $projectRoot
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
$configRoot = Join-Path $projectRoot 'configs\pgin\retrain_lhs10_23'
$configManifestPath = Join-Path $configRoot 'config_manifest.json'
$sharedReferencePath = Join-Path $projectRoot 'results\pgin\retrain_lhs10_23\_shared_source_reference.json'
$batchRoot = Join-Path $projectRoot "results\pgin\retrain_lhs10_23\seed$Seed"
$originalBatchManifestPath = Join-Path $batchRoot '_batch_run.json'
$batchManifestPath = $originalBatchManifestPath

if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
    throw "Missing project Python: $pythonExe"
}
if (-not (Test-Path -LiteralPath $configManifestPath -PathType Leaf)) {
    throw "Missing config manifest: $configManifestPath"
}
if ($OnlyAlias) {
    if ($OnlyAlias -notmatch '^[A-Za-z0-9_]+$') { throw 'OnlyAlias contains invalid characters.' }
    if (-not (Test-Path -LiteralPath $originalBatchManifestPath -PathType Leaf)) {
        throw 'A single-model continuation requires the original failed batch record.'
    }
    $previousBatch = Get-Content -LiteralPath $originalBatchManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($previousBatch.status -ne 'failed' -or $previousBatch.seed -ne $Seed) {
        throw 'Single-model continuation requires a failed batch record for the same seed; inspect any running/interrupted job first.'
    }
    foreach ($entry in $previousBatch.teacher_and_source_sha256.PSObject.Properties) {
        if (-not (Test-Path -LiteralPath $entry.Name -PathType Leaf) -or (Get-FileHash -LiteralPath $entry.Name -Algorithm SHA256).Hash -ne $entry.Value) {
            throw "Original batch source changed; refusing to mix runs: $($entry.Name)"
        }
    }
    if ((Get-FileHash -LiteralPath $configManifestPath -Algorithm SHA256).Hash -ne $previousBatch.config_manifest_sha256) {
        throw 'Config manifest changed since the original batch.'
    }
    $batchManifestPath = Join-Path $batchRoot "_manual_${OnlyAlias}.json"
}
if (Test-Path -LiteralPath $batchManifestPath) {
    throw "Run record already exists; preserve and inspect it before rerunning: $batchManifestPath"
}

$allJobs = Get-Content -LiteralPath $configManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
$jobs = @($allJobs | Where-Object { $_.seed -eq $Seed })
if ($jobs.Count -ne 21 -or @($jobs | Select-Object -ExpandProperty exp_name -Unique).Count -ne 21) {
    throw "Expected 21 unique configs for seed $Seed."
}
if ($OnlyAlias) {
    $jobs = @($jobs | Where-Object { $_.alias -eq $OnlyAlias })
    if ($jobs.Count -ne 1) { throw "Unknown or nonunique model alias: $OnlyAlias" }
}
$allowedModules = @('scripts.training.train_pgin_stepwise_gain', 'scripts.training.train_pgin_stepwise_gain_ss_loss', 'scripts.training.train_pgin_stepwise_gain_difficult_group_gradnorm')
foreach ($job in $jobs) {
    $configPath = Join-Path $projectRoot $job.config
    $checkpointPath = Join-Path $projectRoot $job.checkpoint
    $resultPath = Join-Path $projectRoot $job.results
    if ($job.module -notin $allowedModules -or -not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
        throw "Unexpected trainer or missing config: $($job.config)"
    }
    if ((Get-FileHash -LiteralPath $configPath -Algorithm SHA256).Hash -ne $job.config_sha256) {
        throw "Config differs from the prepared manifest: $configPath"
    }
    if ((Test-Path -LiteralPath $checkpointPath) -or (Test-Path -LiteralPath $resultPath)) {
        throw "Refusing to overwrite experiment artifacts: $checkpointPath or $resultPath"
    }
}

$teacherPaths = @(
    'saved_models\Tabular_MLP_5in_2out_QV.pth',
    'results\steady_state\Tabular_MLP_New\zscore_mean.csv',
    'results\steady_state\Tabular_MLP_New\zscore_std.csv'
)
$dynamicPaths = @(
    'data\Claus_dynamic\Test_dataform_change_air2_R=5_train_split.csv',
    'data\Claus_dynamic\Test_dataform_change_air2_R=5-1_train_split.csv',
    'data\Claus_dynamic\Test_dataform_change_air2_R=5-2_train_split.csv',
    'data\Claus_dynamic\Test_dataform_change_air2_R=5-6_train_split.csv'
)
$expectedLhs = @(10,11,15,16,17,18,19,20,21,22,23) | ForEach-Object { "lhs_generated_dynamic_ss_data_$_.xlsx" }
$lhsPath = Join-Path $projectRoot 'data\Claus_steady_state'
$actualLhs = @(Get-ChildItem -LiteralPath $lhsPath -File -Filter 'lhs_generated_dynamic_ss_data*.xlsx' | Sort-Object Name)
if ((($actualLhs.Name | Sort-Object) -join '|') -ne (($expectedLhs | Sort-Object) -join '|')) {
    throw 'The top-level steady-state training file set differs from the 11-file plan; inspect before training.'
}
$annResultPath = Join-Path $projectRoot 'results\steady_state\Tabular_MLP_New\training_result.json'
$annProtocolPath = Join-Path $projectRoot 'results\steady_state\Tabular_MLP_New\training_protocol.json'
if (-not (Test-Path -LiteralPath $annResultPath -PathType Leaf) -or -not (Test-Path -LiteralPath $annProtocolPath -PathType Leaf)) {
    throw 'Completed retrained ANN result/protocol is missing at the installed teacher path; do not start dynamic training.'
}
$annResult = Get-Content -LiteralPath $annResultPath -Raw -Encoding UTF8 | ConvertFrom-Json
$annProtocol = Get-Content -LiteralPath $annProtocolPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($annResult.completed -ne $true -or -not $annResult.output_sha256 -or -not $annProtocol.source_sha256) {
    throw 'Installed ANN report is not complete or lacks required source/artifact hashes.'
}
$annArtifacts = [ordered]@{
    'teacher.pth' = (Join-Path $projectRoot 'saved_models\Tabular_MLP_5in_2out_QV.pth')
    'zscore_mean.csv' = (Join-Path $projectRoot 'results\steady_state\Tabular_MLP_New\zscore_mean.csv')
    'zscore_std.csv' = (Join-Path $projectRoot 'results\steady_state\Tabular_MLP_New\zscore_std.csv')
    'training_protocol.json' = $annProtocolPath
}
foreach ($name in $annArtifacts.Keys) {
    $path = $annArtifacts[$name]
    $property = $annResult.output_sha256.PSObject.Properties[$name]
    if (-not $property -or -not (Test-Path -LiteralPath $path -PathType Leaf) -or (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $property.Value) {
        throw "Installed ANN artifact does not match completed training report: $name"
    }
}
$annInputNames = @($annProtocol.input_files | ForEach-Object { [System.IO.Path]::GetFileName($_) } | Sort-Object)
if (($annInputNames -join '|') -ne (($expectedLhs | Sort-Object) -join '|')) {
    throw 'ANN training input files differ from the planned 11 LHS files.'
}
foreach ($file in $actualLhs) {
    $property = $annProtocol.source_sha256.PSObject.Properties[$file.FullName]
    if (-not $property -or (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash -ne $property.Value) {
        throw "LHS source differs from completed ANN training protocol: $($file.FullName)"
    }
}
$pythonSourcePaths = @(Get-ChildItem -LiteralPath (Join-Path $projectRoot 'src') -Recurse -File -Filter '*.py' | Sort-Object FullName | ForEach-Object { $_.FullName })
if ($pythonSourcePaths.Count -eq 0) { throw 'No src Python files found.' }
$protectedPaths = @($teacherPaths + $dynamicPaths + @($actualLhs | ForEach-Object { $_.FullName }) + $pythonSourcePaths + @(
    $annResultPath,
    $annProtocolPath,
    $PSCommandPath,
    'pyproject.toml',
    'uv.lock',
    'scripts\training\train_pgin_stepwise_gain.py',
    'scripts\training\train_pgin_stepwise_gain_ss_loss.py',
    'scripts\training\train_pgin_stepwise_gain_difficult_group_gradnorm.py'
))
$protectedHashes = [ordered]@{}
foreach ($relativeOrFull in $protectedPaths) {
    $path = if ([System.IO.Path]::IsPathRooted($relativeOrFull)) { $relativeOrFull } else { Join-Path $projectRoot $relativeOrFull }
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Required source is missing: $path" }
    $protectedHashes[$path] = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
}
if (Test-Path -LiteralPath $sharedReferencePath -PathType Leaf) {
    $reference = Get-Content -LiteralPath $sharedReferencePath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ((Get-FileHash -LiteralPath $configManifestPath -Algorithm SHA256).Hash -ne $reference.config_manifest_sha256 -or
        @($reference.protected_sha256.PSObject.Properties).Count -ne $protectedHashes.Count) {
        throw 'Shared source reference does not match the current configuration or source list.'
    }
    foreach ($path in $protectedHashes.Keys) {
        $entry = $reference.protected_sha256.PSObject.Properties[$path]
        if (-not $entry -or $entry.Value -ne $protectedHashes[$path]) {
            throw "Shared teacher or source differs from the earlier seed: $path"
        }
    }
}

& $pythonExe -c "import sys,torch; print('PyTorch:',torch.__version__,'CUDA:',torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 2)"
if ($LASTEXITCODE -ne 0) { throw 'CUDA is unavailable; refusing to start a long CPU training batch.' }
if ($CheckOnly) {
    Write-Host "Preflight passed for seed ${Seed}: $($jobs.Count) experiments; no training or output was started."
    $jobs | Select-Object family, alias, module, exp_name | Format-Table -AutoSize
    return
}

if (-not (Test-Path -LiteralPath $sharedReferencePath)) {
    $referenceDir = Split-Path -Parent $sharedReferencePath
    New-Item -ItemType Directory -Path $referenceDir -Force | Out-Null
    $reference = [ordered]@{
        created_utc = [DateTime]::UtcNow.ToString('o')
        config_manifest_sha256 = (Get-FileHash -LiteralPath $configManifestPath -Algorithm SHA256).Hash
        protected_sha256 = $protectedHashes
    }
    # CreateNew fails rather than replacing a reference made by another run.
    $writer = [System.IO.StreamWriter]::new([System.IO.File]::Open($sharedReferencePath, [System.IO.FileMode]::CreateNew), [System.Text.UTF8Encoding]::new($false))
    try { $writer.Write(($reference | ConvertTo-Json -Depth 10)) } finally { $writer.Dispose() }
}

$batch = [ordered]@{
    status = 'running'
    seed = $Seed
    started_utc = [DateTime]::UtcNow.ToString('o')
    teacher_and_source_sha256 = $protectedHashes
    config_manifest_sha256 = (Get-FileHash -LiteralPath $configManifestPath -Algorithm SHA256).Hash
    completed = @()
    active = $null
    description = 'Teacher is shared across dynamic seeds; ANN holdout is not an unseen holdout for the complete dynamic system.'
}
New-Item -ItemType Directory -Path $batchRoot -Force | Out-Null
$batch | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $batchManifestPath -Encoding UTF8

try {
    foreach ($job in $jobs) {
        foreach ($path in $protectedHashes.Keys) {
            if (-not (Test-Path -LiteralPath $path -PathType Leaf) -or (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $protectedHashes[$path]) {
                throw "Protected teacher or source changed during the batch: $path"
            }
        }
        $currentLhs = @(Get-ChildItem -LiteralPath $lhsPath -File -Filter 'lhs_generated_dynamic_ss_data*.xlsx' | Select-Object -ExpandProperty Name | Sort-Object)
        if (($currentLhs -join '|') -ne (($expectedLhs | Sort-Object) -join '|')) {
            throw 'Steady-state source set changed during the batch.'
        }
        $currentPythonSources = @(Get-ChildItem -LiteralPath (Join-Path $projectRoot 'src') -Recurse -File -Filter '*.py' | Sort-Object FullName | ForEach-Object { $_.FullName })
        if (($currentPythonSources -join '|') -ne ($pythonSourcePaths -join '|')) {
            throw 'The src Python source file set changed during the batch.'
        }
        $configPath = Join-Path $projectRoot $job.config
        if ((Get-FileHash -LiteralPath $configPath -Algorithm SHA256).Hash -ne $job.config_sha256) {
            throw "Config changed during the batch: $configPath"
        }
        $batch.active = $job.exp_name
        $batch | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $batchManifestPath -Encoding UTF8
        Write-Host "`n[$($batch.completed.Count + 1)/$($jobs.Count)] $($job.family) $($job.alias), seed=$Seed"
        $extraArgs = @()
        if ($job.cli_args) { $extraArgs = @($job.cli_args) }
        & $pythonExe -u -m $job.module --config $configPath @extraArgs
        if ($LASTEXITCODE -ne 0) { throw "Training failed: $($job.exp_name), exit=$LASTEXITCODE" }
        $checkpoint = Join-Path $projectRoot $job.checkpoint
        $resultPath = Join-Path $projectRoot $job.results
        foreach ($file in @($checkpoint, (Join-Path $resultPath 'zscore_mean.csv'), (Join-Path $resultPath 'zscore_std.csv'))) {
            if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Expected training artifact missing: $file" }
        }
        foreach ($path in $protectedHashes.Keys) {
            if (-not (Test-Path -LiteralPath $path -PathType Leaf) -or (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $protectedHashes[$path]) {
                throw "Protected teacher or source changed during training: $path"
            }
        }
        $currentLhs = @(Get-ChildItem -LiteralPath $lhsPath -File -Filter 'lhs_generated_dynamic_ss_data*.xlsx' | Select-Object -ExpandProperty Name | Sort-Object)
        $currentPythonSources = @(Get-ChildItem -LiteralPath (Join-Path $projectRoot 'src') -Recurse -File -Filter '*.py' | Sort-Object FullName | ForEach-Object { $_.FullName })
        if (($currentLhs -join '|') -ne (($expectedLhs | Sort-Object) -join '|') -or ($currentPythonSources -join '|') -ne ($pythonSourcePaths -join '|')) {
            throw 'The steady-state or src Python source file set changed during training.'
        }
        if ((Get-FileHash -LiteralPath $configPath -Algorithm SHA256).Hash -ne $job.config_sha256) {
            throw "Config changed during training: $configPath"
        }
        $batch.completed += [ordered]@{
            exp_name = $job.exp_name
            checkpoint = $job.checkpoint
            checkpoint_sha256 = (Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash
            zscore_mean_sha256 = (Get-FileHash -LiteralPath (Join-Path $resultPath 'zscore_mean.csv') -Algorithm SHA256).Hash
            zscore_std_sha256 = (Get-FileHash -LiteralPath (Join-Path $resultPath 'zscore_std.csv') -Algorithm SHA256).Hash
        }
        $batch.active = $null
        $batch | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $batchManifestPath -Encoding UTF8
    }
    $batch.status = 'complete'
    $batch.finished_utc = [DateTime]::UtcNow.ToString('o')
    $batch | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $batchManifestPath -Encoding UTF8
    Write-Host "`nCompleted $($jobs.Count) model(s) for seed $Seed."
} catch {
    $batch.status = 'failed'
    $batch.error = $_.Exception.Message
    $batch.finished_utc = [DateTime]::UtcNow.ToString('o')
    $batch | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $batchManifestPath -Encoding UTF8
    throw
}
