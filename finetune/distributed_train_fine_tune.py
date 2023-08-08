import os
import sys
import time
import subprocess
import argparse
import warnings
warnings.filterwarnings("ignore")

import torch

# change this to your utils directory
sys.path.insert(0, 'C:\\PhD\\projects\\IEEE SP\\github repo\\EEGWaveRegressor\\utils')
from distributed_util import *

if __name__ == '__main__':
    # Parameters
    # Length of the decision window
    window_length = 8
    hop_length = 0.117

    args_list = ['fine_tune_subject.py']
    #args_list += args_str.split(' ') if len(args_str) > 0 else []
    #args_list.append('--config={}'.format(config))
    num_gpus = torch.cuda.device_count()
    print("num_gpus: ", num_gpus)
    args_list.append('--train_config={}'.format("wavenet_best_4_20_32_finetune.json"))
    args_list.append('--window_length={}'.format(window_length))
    args_list.append('--hop_length={}'.format(hop_length))
    args_list.append('--num_gpus={}'.format(num_gpus))
    args_list.append("--group_name=group_{}".format(time.strftime("%Y_%m_%d-%H%M%S")))

    if not os.path.isdir("exp/"):
        os.makedirs("exp/")
        os.chmod("exp/", 0o775)

    workers = []

    for i in range(num_gpus):
        args_list[-2] = '--rank={}'.format(i)
        stdout = None if i == 0 else open(
            os.path.join("exp/", "GPU_{}.log".format(i)), "w")
        print(args_list)
        p = subprocess.Popen([str(sys.executable)]+args_list, stdout=stdout)
        workers.append(p)

    for p in workers:
        p.wait()


