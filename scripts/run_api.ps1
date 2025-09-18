param([string]$Py = "C:\\Users\\Lenovo\\miniforge3\\envs\\ov\\python.exe")
$env:OV_CACHE_DIR = "$PWD\\.ov_cache"
if (-not (Test-Path $env:OV_CACHE_DIR)) { New-Item -ItemType Directory -Force $env:OV_CACHE_DIR | Out-Null }
$env:OV_DEVICE   = "NPU"      # 폴백은 코드에서 AUTO→CPU
$env:SF2_PATH    = "$PWD\\assets\\FluidR3_GM.sf2"  # 없으면 서버에서 에러 안내
& $Py -m uvicorn src.api.server:app --host 127.0.0.1 --port 9009
