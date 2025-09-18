import base64
import io
import json
import sys

import soundfile as sf


def main(p: str = "out.json", out_wav: str = "out.wav") -> None:
    with open(p, "r", encoding="utf-8") as f:
        o = json.load(f)
    audio_b = base64.b64decode(o["b64"])
    data, sr = sf.read(io.BytesIO(audio_b))
    sf.write(out_wav, data, sr)
    print(f"saved {out_wav} sr={sr} len={len(data)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "out.json")
