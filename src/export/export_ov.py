import argparse,os,sys,subprocess

def main(ckpt,out):
    os.makedirs(out,exist_ok=True)
    cmd=[sys.executable,'-m','optimum.exporters.openvino',f'--model={ckpt}','--task=text-generation','--weight-format=fp16','--ov_config=PERFORMANCE_HINT=LATENCY',out]
    print('Running:',' '.join(cmd)); subprocess.run(cmd,check=True); print('Exported:',out)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--ckpt',required=True); ap.add_argument('--out',required=True)
    a=ap.parse_args(); main(a.ckpt,a.out)
