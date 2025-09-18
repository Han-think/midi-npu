import json, base64, io, soundfile as sf, sys
p = sys.argv[1] if len(sys.argv)>1 else "out.json"
o = json.load(open(p,"r",encoding="utf-8"))
audio = base64.b64decode(o["b64"])
data, sr = sf.read(io.BytesIO(audio))
sf.write("out.wav", data, sr)
print("saved out.wav sr=", sr, "len=", len(data))
