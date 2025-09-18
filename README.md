# midi-npu
[![Quality](https://github.com/Han-think/midi-npu/actions/workflows/quality.yml/badge.svg)](../../actions)
npu_midi_compose

## Real Audio (NPU) Quickstart

> REQUIRE: `assets/FluidR3_GM.sf2` (GM SoundFont, not included)

```powershell
# repo 루트
powershell -ExecutionPolicy Bypass -File scripts\real_run.ps1
# 새 콘솔에서 호출 예:
$body = @{
  base_style="rock"; bpm=120; key="C";
  sections=@(@{name="intro";duration=4}, @{name="verse";duration=8});
  seed=1; with_vocal=$false; max_tokens=128
}
Invoke-RestMethod -Uri "http://127.0.0.1:9009/v1/midi/compose_full" \
  -Method Post -ContentType "application/json" -Body ($body | ConvertTo-Json -Depth 5) \
  | Out-File out.json
python scripts\save_wav.py out.json   # -> out.wav
```
