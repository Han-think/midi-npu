param(
  [string]$Py = "C:\\Users\\Lenovo\\miniforge3\\envs\\ov\\python.exe",
  [string]$Ckpt = "checkpoints\\epoch2",
  [string]$Out = "exports\\gpt_ov"
)
& $Py src\\export\\export_ov.py --ckpt $Ckpt --out $Out
