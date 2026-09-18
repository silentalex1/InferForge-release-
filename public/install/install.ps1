[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.SecurityProtocolType]::Tls12
$ErrorActionPreference = "Stop"
$Index = "https://hyperneural.cfd/pypi/simple/"
$Wheel = "https://inferforge.org/pypi/packages/inferforge-0.2.1-py3-none-any.whl"

function Find-Python {
    foreach ($cmd in @("py -3", "python", "python3")) {
        try {
            $out = Invoke-Expression "$cmd --version 2>&1" | Out-String
            if ($out -match "Python\s+3\.\d+" -and $out -notmatch "was not found") { return $cmd }
        } catch { continue }
    }
    return $null
}

Write-Host ""
Write-Host "  InferForge Installer" -ForegroundColor Cyan
Write-Host "  ====================" -ForegroundColor Cyan
Write-Host ""

$python = Find-Python
if (-not $python) {
    Write-Host "Python 3.10+ is required but was not found." -ForegroundColor Red
    Write-Host "Install it from https://www.python.org/downloads/ and tick 'Add python to PATH', then run this again." -ForegroundColor Yellow
    exit 1
}

$version = (Invoke-Expression "$python -c ""import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')""" 2>&1).ToString().Trim()
$major, $minor = $version.Split('.') | ForEach-Object { [int]$_ }
if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 10)) {
    Write-Host "Python 3.10+ required. Found $version" -ForegroundColor Red
    exit 1
}
Write-Host "Python $version detected ($python)" -ForegroundColor Green

$pipOk = $false
try { & $python -m pip --version 2>&1 | Out-Null; if ($LASTEXITCODE -eq 0) { $pipOk = $true } } catch {}
if (-not $pipOk) {
    Write-Host "Installing pip..." -ForegroundColor Yellow
    & $python -m ensurepip --upgrade
}
& $python -m pip install --upgrade pip --quiet

Write-Host "Installing InferForge..." -ForegroundColor Cyan
& $python -m pip install --upgrade inferforge --index-url $Index --extra-index-url https://pypi.org/simple
if ($LASTEXITCODE -ne 0) {
    Write-Host "Index install failed, fetching the wheel directly..." -ForegroundColor Yellow
    & $python -m pip install --upgrade $Wheel
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installation failed." -ForegroundColor Red
        exit 1
    }
}

$forgeVersion = (& $python -m inferforge --version 2>&1 | Out-String).Trim()
Write-Host ""
Write-Host "  InferForge installed: $forgeVersion" -ForegroundColor Green
Write-Host "  forge --help      all commands" -ForegroundColor White
Write-Host "  forge connect     link your inferforge.org account" -ForegroundColor White
Write-Host "  forge pull <model>  download a model" -ForegroundColor White
Write-Host ""
