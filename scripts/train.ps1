param(
  [string]$Py = "C:\\Users\\Lenovo\\miniforge3\\envs\\ov\\python.exe",
  [string]$Config = "src\\training\\configs\\default.yaml"
)
& $Py src\\training\\train_lm.py --config $Config
