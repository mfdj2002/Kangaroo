import os, torch, shutil, glob, time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_home", type=str, default="/data/shared_workspace/fujingkai/dataset/sharegpt_0_67999_mufp16")
parser.add_argument("--start_epoch", type=int, default=0)
parser.add_argument("--end_epoch", type=int, default=20)
parser.add_argument("--bs", type=int, default=4)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--exit_layer", type=int, default=2)
parser.add_argument("--debug", action='store_true')
parser.add_argument("--lora", action='store_true')
args = parser.parse_args()
print(args)

work_dir = os.path.abspath(os.path.dirname(__file__))
os.chdir(work_dir)

def data_moxing(src, dst):
    shutil.copytree(src, dst, dirs_exist_ok=True)

LOCAL_TRAIN_DATA = args.data_home
if args.lora:
    LOCAL_CKPT_DATA = "/data/shared_workspace/fujingkai/models/EarlyExitLora"
    train_script = "train_lora.py"
else:
    LOCAL_CKPT_DATA = "/data/shared_workspace/fujingkai/models/EarlyExitAdapter"
    train_script = "train.py"

MASTER_ADDR = "localhost"
MASTER_PORT = 6000
rank = "0"
nnodes = 1
processes = int(nnodes) * torch.cuda.device_count()

for epoch in range(args.start_epoch, args.end_epoch):
    print("start epoch: ", epoch)
    dst_dir = LOCAL_TRAIN_DATA

    command = f"MKL_SERVICE_FORCE_INTEL=1 MKL_THREADING_LAYER=GNU accelerate launch --multi_gpu \
        --num_machines {nnodes} --num_processes {processes} --main_process_ip {MASTER_ADDR} \
            --main_process_port {MASTER_PORT+epoch} --machine_rank {rank} --mixed_precision=fp16 \
                {train_script} \
                --start {epoch} --tmpdir {dst_dir} --cpdir {LOCAL_CKPT_DATA} \
                    --basepath /data/shared_workspace/fujingkai/models/models--kevinpro--Vicuna-7B-CoT \
                        --configpath {LOCAL_CKPT_DATA}/config.json \
                            --bs {args.bs} --lr {args.lr} --exit_layer {args.exit_layer}"

    print(command)
    os.system(command)