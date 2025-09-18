# REQUIRE: Python 3.11+, assets\FluidR3_GM.sf2(필수)
$ErrorActionPreference = "Stop"

# 0) Python / venv
if (-not (Get-Command python -ErrorAction SilentlyContinue)) { throw "Python이 PATH에 없음" }
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt

# 1) 사운드폰트 확인(없으면 중단 = 무음 금지)
$sf2="assets\FluidR3_GM.sf2"
if (-not (Test-Path $sf2)) { throw "사운드폰트 $sf2 가 필요합니다." }
$env:SF2_PATH=$sf2
$env:SKIP_AUDIO=""        # 무음 금지
$env:ALLOW_FAKE_GEN=""    # FAKE 금지

# 2) 데이터 준비(사용자 MIDI)
# 구조: data/raw/<song>/track.mid (+ sections.json 선택)
if (-not (Test-Path "data\raw")) { throw "data\raw 폴더에 MIDI를 넣어주세요." }
python -m src.training.prepare_dataset

# 3) 경량 실전 학습(빠른 검증용)
if (-not (Test-Path "src\training\configs\real-mini.yaml")) {
  @"
train: {epochs: 2, batch_size: 2, lr: 2.0e-4, seq_len: 512}
model: {n_layer: 4, n_head: 4, n_embd: 256, vocab_path: data/processed/vocab.json}
data:  {train_jsonl: data/processed/jsonl/train.jsonl}
"@ | Set-Content -Path src\training\configs\real-mini.yaml -Encoding UTF8
}
python -m src.training.train_lm --config src/training/configs/real-mini.yaml

# 4) OpenVINO IR Export
$ckpt = (Get-ChildItem checkpoints -Directory | Sort-Object Name -Descending | Select-Object -First 1).FullName
if (-not $ckpt) { throw "체크포인트가 없음(checkpoints/epochN)" }
python -m src.export.export_ov --ckpt $ckpt --out exports\gpt_ov

# 5) OV XML / 디바이스
$xml = (Get-ChildItem exports -Recurse -Filter *.xml | Select-Object -First 1).FullName
if (-not $xml) { throw "OV XML 생성 실패" }
$env:OV_XML_PATH=$xml
try {
  $devs = python - <<'PY'
import openvino as ov; print(ov.Core().available_devices)
PY
  if ($devs -match "NPU") { $env:OV_DEVICE="NPU" } else { $env:OV_DEVICE="AUTO" }
} catch { $env:OV_DEVICE="AUTO" }
$env:OV_CACHE_DIR=".ov_cache"

# 6) 서버 실행
python -m uvicorn src.api.server:app --host 127.0.0.1 --port 9009
