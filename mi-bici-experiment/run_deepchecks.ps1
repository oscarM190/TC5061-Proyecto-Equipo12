# Create the isolated Python 3.10 environment when needed, then test and validate.
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$projectRoot = Split-Path -Parent $PSCommandPath
$repositoryRoot = Split-Path -Parent $projectRoot
$environmentPath = Join-Path $projectRoot '.venv-deepchecks-windows'
$pythonPath = Join-Path $environmentPath 'Scripts\python.exe'
$requirementsPath = Join-Path $projectRoot 'requirements-deepchecks.txt'
$localUvPath = Join-Path $repositoryRoot '.venv\Scripts\uv.exe'

if (Test-Path -LiteralPath $localUvPath -PathType Leaf) {
    $uvPath = $localUvPath
} else {
    $uvCommand = Get-Command uv -ErrorAction SilentlyContinue
    if ($null -eq $uvCommand) {
        throw 'uv was not found in the main .venv or on PATH. Install uv first.'
    }
    $uvPath = $uvCommand.Source
}

if (Test-Path -LiteralPath $pythonPath -PathType Leaf) {
    Write-Host "Using existing Python 3.10 environment: $environmentPath"
} elseif (Test-Path -LiteralPath $environmentPath) {
    throw "An incomplete environment exists at $environmentPath. Remove or repair it before retrying."
} else {
    Write-Host "Creating Python 3.10 environment: $environmentPath"
    New-Item -ItemType Directory -Path (Split-Path -Parent $environmentPath) -Force | Out-Null
    & $uvPath venv --no-project --managed-python --python 3.10 $environmentPath
    if ($LASTEXITCODE -ne 0) { throw 'uv could not create the Python 3.10 environment.' }
}

& $pythonPath -c 'import sys; assert sys.version_info[:2] == (3, 10), sys.version'
if ($LASTEXITCODE -ne 0) { throw "The environment at $environmentPath is not Python 3.10." }

Write-Host 'Installing Deepchecks requirements...'
& $uvPath pip install --python $pythonPath -r $requirementsPath
if ($LASTEXITCODE -ne 0) { throw 'Installing Deepchecks requirements failed.' }

Push-Location $projectRoot
try {
    Write-Host 'Running data validation tests...'
    & $pythonPath -m pytest -q tests/test_data_validation.py
    if ($LASTEXITCODE -ne 0) { throw 'Data validation tests failed.' }

    Write-Host 'Validating the trips Parquet with Deepchecks...'
    & $pythonPath -m mi_bici_experiment.data_validation
    if ($LASTEXITCODE -ne 0) { throw 'Deepchecks validation failed. Review reports/deepchecks.' }
} finally {
    Pop-Location
}
