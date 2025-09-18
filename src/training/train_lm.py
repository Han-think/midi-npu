import os,json,torch,math,argparse,yaml
from torch.utils.data import Dataset,DataLoader
from transformers import GPT2Config,GPT2LMHeadModel,AdamW,get_cosine_schedule_with_warmup

class DS(Dataset):
    def __init__(s,p,seq=2048): s.items=[json.loads(l)['tokens'] for l in open(p,'r',encoding='utf-8')]; s.seq=seq
    def __len__(s): return len(s.items)
    def __getitem__(s,i):
        x=s.items[i][:s.seq]; x=x if len(x)>2 else x+[0,0]
        import torch; return torch.tensor(x[:-1]),torch.tensor(x[1:])

def coll(b):
    import torch; i=[t[0] for t in b]; o=[t[1] for t in b]
    return (torch.nn.utils.rnn.pad_sequence(i,True,0),
            torch.nn.utils.rnn.pad_sequence(o,True,0))

def main(cfgp):
    cfg=yaml.safe_load(open(cfgp)); vocab=json.load(open(cfg['model']['vocab_path']))
    vs=max(vocab.values())+1
    m=GPT2LMHeadModel(GPT2Config(vocab_size=vs,n_layer=cfg['model']['n_layer'],n_head=cfg['model']['n_head'],n_embd=cfg['model']['n_embd'],n_positions=cfg['train']['seq_len']))
    ds=DS(cfg['data']['train_jsonl'],cfg['train']['seq_len']); dl=DataLoader(ds,batch_size=cfg['train']['batch_size'],shuffle=True,collate_fn=coll)
    dev='cuda' if torch.cuda.is_available() else 'cpu'; m.to(dev); opt=AdamW(m.parameters(),lr=cfg['train']['lr'])
    sch=get_cosine_schedule_with_warmup(opt,0,len(dl)*cfg['train']['epochs']); os.makedirs('checkpoints',exist_ok=True); m.train()
    for e in range(cfg['train']['epochs']):
        s=0;n=0
        for x,y in dl:
            x,y=x.to(dev),y.to(dev); out=m(input_ids=x,labels=y); loss=out.loss
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); sch.step()
            s+=loss.item(); n+=1
        print(f'epoch {e+1} loss={s/max(n,1):.4f}'); m.save_pretrained(f'checkpoints/epoch{e+1}')

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='src/training/configs/default.yaml'); a=ap.parse_args(); main(a.config)
