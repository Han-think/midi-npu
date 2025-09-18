param([ValidateSet('setup','prepare','train','export','serve','demo')][string]$Task='setup')
. $PSScriptRoot\_env.ps1

switch ($Task) {
  'setup' {
    & $PY -m pip install -U -r requirements.txt
    $mg='models\musicgen_static_ov'
    if (!(Test-Path $mg)) { & $PY -c "from huggingface_hub import snapshot_download; snapshot_download('Intel/musicgen-static-openvino', local_dir=r'models\\musicgen_static_ov', local_dir_use_symlinks=False)" }
    Write-Host 'Setup OK'; break
  }
  'prepare' {
    & $PY src\training\prepare_dataset.py; break
  }
  'train' {
    & $PY src\training\train_lm.py --config src\training\configs\default.yaml; break
  }
  'export' {
    & $PY src\export\export_ov.py --ckpt checkpoints\epoch2 --out exports\gpt_ov; break
  }
  'serve' {
    & $PY -m uvicorn src.api.server:app --host 127.0.0.1 --port 9009; break
  }
  'demo' {
    $b=@{prompt='lofi hiphop, warm, 90bpm';duration=8}|ConvertTo-Json
    Invoke-RestMethod -Uri 'http://127.0.0.1:9009/v1/audio/musicgen' -Method Post -ContentType 'application/json' -Body $b
    $c=@{base_style='rock';bpm=120;key='C';
        sections=@(@{name='intro';duration=4},@{name='verse1';duration=8},@{name='chorus';duration=10},@{name='solo';duration=6},@{name='outro';duration=4});
        seed=42;with_vocal=$false;max_tokens=512} | ConvertTo-Json -Depth 5
    Invoke-RestMethod -Uri 'http://127.0.0.1:9009/v1/midi/compose_full' -Method Post -ContentType 'application/json' -Body $c |
      % { [IO.File]::WriteAllBytes('fullsong.wav',[Convert]::FromBase64String($_.b64)) }
    break
  }
}
