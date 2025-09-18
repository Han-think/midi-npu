$global:PY = "C:\\Users\\Lenovo\\miniforge3\\envs\\ov\\python.exe"
if (!(Test-Path $PY)) { throw "Python not found at $PY" }
$env:OV_CACHE_DIR = "$PWD\\.ov_cache"; if (!(Test-Path $env:OV_CACHE_DIR)) { New-Item -ItemType Directory -Force $env:OV_CACHE_DIR | Out-Null }
$env:OV_DEVICE = "NPU"
if (-not $env:SF2_PATH) { $env:SF2_PATH = "$PWD\\assets\\FluidR3_GM.sf2" }
