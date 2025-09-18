param([string]$Py = "C:\\Users\\Lenovo\\miniforge3\\envs\\ov\\python.exe")

Write-Host ">>> Installing Python packages..." -ForegroundColor Cyan
& $Py -m pip install -U -r requirements.txt

Write-Host ">>> (Optional) HuggingFace login if needed for gated repos."
Write-Host "    Run:  $Py -m huggingface_hub login" -ForegroundColor Yellow

# MusicGen 데모용 IR (옵션)
$mgDir = "models\\musicgen_static_ov"
if (-not (Test-Path $mgDir)) {
  Write-Host ">>> Downloading Intel/musicgen-static-openvino (optional demo)..." -ForegroundColor Cyan
  & $Py -c "from huggingface_hub import snapshot_download; snapshot_download('Intel/musicgen-static-openvino', local_dir=r'models\\musicgen_static_ov', local_dir_use_symlinks=False)"
} else {
  Write-Host ">>> models/musicgen_static_ov already exists, skip."
}

Write-Host ">>> Done. Next:"
Write-Host "  scripts\\prepare.ps1   # 데이터→JSONL"
Write-Host "  scripts\\train.ps1     # 학습"
Write-Host "  scripts\\export_ov.ps1 # OV IR"
Write-Host "  scripts\\run_api.ps1   # 서버 실행(9009)"
