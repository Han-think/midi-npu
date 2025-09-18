import os, json, math, argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, AdamW, get_cosine_schedule_with_warmup

class JsonlDS(Dataset):
    def __init__(self, path, seq_len=2048):
        self.items=[]; self.seq_len=seq_len
        with open(path,"r",encoding="utf-8") as f:
            for line in f: self.items.append(json.loads(line)["tokens"])
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        x=self.items[i][:self.seq_len]; x=x if len(x)>2 else x+[0,0]
        return torch.tensor(x[:-1]), torch.tensor(x[1:])

def collate(b):
    import torch
    i=[t[0] for t in b]; o=[t[1] for t in b]
    i=torch.nn.utils.rnn.pad_sequence(i,batch_first=True,padding_value=0)
    o=torch.nn.utils.rnn.pad_sequence(o,batch_first=True,padding_value=0)
    return i,o

def main(cfg_path):
    import yaml
    cfg=yaml.safe_load(open(cfg_path,"r"))
    vocab=json.load(open(cfg["model"]["vocab_path"],"r")); vs=max(vocab.values())+1
    gpt_cfg = GPT2Config(vocab_size=vs, n_layer=cfg["model"]["n_layer"], n_head=cfg["model"]["n_head"],
                         n_embd=cfg["model"]["n_embd"], n_positions=cfg["train"]["seq_len"])
    model=GPT2LMHeadModel(gpt_cfg)
    ds=JsonlDS(cfg["data"]["train_jsonl"], cfg["train"]["seq_len"])
    dl=DataLoader(ds,batch_size=cfg["train"]["batch_size"],shuffle=True,collate_fn=collate)
    dev="cuda" if torch.cuda.is_available() else "cpu"; model.to(dev)
    opt=AdamW(model.parameters(), lr=cfg["train"]["lr"])
    total_steps=len(dl)*cfg["train"]["epochs"]; sch=get_cosine_schedule_with_warmup(opt, 0, total_steps)
    os.makedirs("checkpoints",exist_ok=True)
    model.train()
    for ep in range(cfg["train"]["epochs"]):
        s=0.0;n=0
        for inp,tgt in dl:
            inp,tgt=inp.to(dev),tgt.to(dev)
            out=model(input_ids=inp, labels=tgt)
            loss=out.loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step(); sch.step()
            s+=loss.item(); n+=1
        print(f"epoch {ep+1} loss={s/max(n,1):.4f}")
        model.save_pretrained(f"checkpoints/epoch{ep+1}")
    print("done")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--config", default="src/training/configs/default.yaml")
    a=ap.parse_args(); main(a.config)
