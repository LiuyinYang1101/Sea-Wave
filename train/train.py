import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.insert(0, 'C:\\PhD\\projects\\IEEE SP\\github repo\\EEGWaveRegressor\\utils')
sys.path.insert(0, 'C:\\PhD\\projects\\IEEE SP\\github repo\\EEGWaveRegressor\\models')

from util import rescale, find_max_epoch, print_size, pearson_loss
from pathlib import Path
from distributed_util import init_distributed, apply_gradient_allreduce, reduce_tensor
import time
from EEGWaveRegressor import WaveNet_regressor as EEGWaveRegressor
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from CustomDatasetPytorch import CustomAllLoadDataset
from torch.utils.data.distributed import DistributedSampler
from cyclic_scheduler import CyclicLRWithRestarts

def train(train_conig_file, window_length, hop_length, num_gpus, rank, group_name):
    train_from_scratch = True
    fs = 64

    with open(train_conig_file) as fp:
        config = json.load(fp)
        train_config = config["train_config"]  # training parameters
        global dist_config
        dist_config = config["dist_config"]  # to initialize distributed training
        global wavenet_config
        wavenet_config = config["wavenet_config"]  # to define wavenet
        global trainset_config
        trainset_config = config["trainset_config"]  # to load trainset
        global diffusion_hyperparams

    output_directory = train_config["output_directory"]
    tensorboard_directory = train_config["tensorboard_directory"]
    ckpt_iter = train_config["ckpt_iter"]
    iters_per_ckpt = train_config["iters_per_ckpt"]
    iters_per_logging = train_config["iters_per_logging"]
    n_iters = train_config["n_iters"]
    learning_rate = train_config["learning_rate"]
    train_from_scratch = train_config["train_from_scratch"]
    batch_size = train_config["batch_size_per_gpu"]

    pretrained_model_path = Path(trainset_config["pretrained_model_path"])
    fs = trainset_config["fs"]
    data_folder = Path(trainset_config["data_path"])

    stimulus_features = ["envelope"]
    features = ["eeg"] + stimulus_features
    
    train_files = [path for path in Path(data_folder).resolve().glob("train_-_*") if
                   path.stem.split("_-_")[-1].split(".")[0] in features]

    valid_files = [path for path in Path(data_folder).resolve().glob("val_-_*") if
                   path.stem.split("_-_")[-1].split(".")[0] in features]
    
    test_files = [path for path in Path(data_folder).resolve().glob("test_-_*") if
                   path.stem.split("_-_")[-1].split(".")[0] in features]
    
    train_files = train_files + valid_files

    # generate experiment (local) path
    local_path = "EEGWaveRegressor_population_ch{}_layer{}".format(wavenet_config["res_channels"],wavenet_config["num_res_layers"])
    # Create tensorboard logger.
    if rank == 0:
        tb = SummaryWriter(os.path.join('exp', local_path, tensorboard_directory))

    # distributed running initialization
    if num_gpus > 1:
        init_distributed(rank, num_gpus, group_name, **dist_config)

    # Get shared output_directory ready
    output_directory = os.path.join('exp', local_path, output_directory)
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        print("output directory", output_directory, flush=True)


    net = EEGWaveRegressor(**wavenet_config).cuda()
    print_size(net)

    # apply gradient all reduce
    if num_gpus > 1:
        net = apply_gradient_allreduce(net)
    if train_from_scratch == False:
        if ckpt_iter == 'max':
            ckpt_iter = find_max_epoch(pretrained_model_path)
            if ckpt_iter >= 0:
                try:
                    # load checkpoint file
                    model_path = os.path.join(pretrained_model_path, '{}.pkl'.format(ckpt_iter))
                    checkpoint = torch.load(model_path, map_location='cpu')
                    # feed model dict and optimizer state
                    net.load_state_dict(checkpoint['model_state_dict'])
                    print('Successfully loaded model at iteration {}'.format(ckpt_iter))
                except Exception as inst:
                    ckpt_iter = -1
                    print(inst)
                    print('try load failed, start training from initialization.')
            else:
                ckpt_iter = -1
                print('No valid checkpoint model found, start training from initialization.')
    else:
        print('Train from scratch.')

   
    # load training data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = CustomAllLoadDataset(train_files, int(window_length*fs), int(hop_length*fs),aug=False)
    train_dataset.convertToTensorType()
    train_dataset.send_to_device()
    
    if rank == 0: # only evaluate on the first GPU
        test_dataset = CustomAllLoadDataset(test_files, int(3*fs), int(0.5*fs),False,False)
        test_dataset.convertToTensorType()
        test_dataset.send_to_device()
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=0, shuffle = False)
    
    #print(num_gpus, " to distribute dataset")
    # distributed sampler
    if num_gpus > 1:
        train_sampler = DistributedSampler(train_dataset) 
    else: 
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                              batch_size=batch_size,
                                              sampler=train_sampler,
                                              num_workers=0,
                                              pin_memory=False)
    
    # define optimizer
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = CyclicLRWithRestarts(optimizer, batch_size, train_dataset.__len__(), restart_period=10, t_mult=1.2, policy="cosine")
    
    mse_loss = nn.MSELoss()
    # Instantiate the scheduler with a warm restart schedule
    # training
    if train_from_scratch:
        n_iter = 0
    else:
        n_iter = ckpt_iter + 1
    best_score = 0
    while n_iter < n_iters + 1:
        if num_gpus >1:
            train_sampler.set_epoch(n_iter)
        scheduler.step()
        start_time = time.time()
        batch_loss = 0
        batch_pear_loss = 0
        no_epoch = 0
        #print("new iteration")
        net.train()
        for i, data in enumerate(train_loader, 0):
            #print("new batch")
            no_epoch = i
            # get the inputs; data is a list of [inputs, labels]
            eeg, audio = data[0], data[1]
            #print("to cuda")
            # back-propagation
            optimizer.zero_grad()
            
            output = net(eeg)
            per_loss = pearson_loss(output, audio, axis=-1)
     
            loss = mse_loss(output, audio)
            
            if(torch.isnan(loss).any()==True):
                print("nan detected")
            
            if num_gpus > 1:
                reduced_pear_loss = reduce_tensor(per_loss.data, num_gpus).item()
                reduced_loss = reduce_tensor(loss.data, num_gpus).item()
                batch_loss += reduced_loss
                batch_pear_loss += reduced_pear_loss
            else:
                batch_loss += loss.item()
                batch_pear_loss += per_loss.item()
            
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
            optimizer.step()
            scheduler.batch_step()
            # output to log
            # note, only do this on the first gpu
        print("iteration done")
        
        if n_iter % iters_per_logging == 0:
            # validation
            if rank == 0:
                with torch.no_grad():
                    net.eval()  # Optional when not using Model Specific layer
                    valLoss = 0
                    valPearson = 0
                    no_valid_epoch = 0
                    for i, data in enumerate(test_loader, 0):
                        # get the inputs
                        no_valid_epoch = i
                        eeg, audio = data[0], data[1]
                        # calc loss
                        output = net(eeg)
                        val_loss_mse = mse_loss(output, audio)
                        val_loss_per = pearson_loss(output, audio)
                        
                        valLoss += val_loss_mse.item()
                        valPearson += val_loss_per.item()
                        #print(valPearson,audio.shape)
                    print("validation done")
                
                train_pearson = -batch_pear_loss / (no_epoch+1)
                train_loss = batch_loss / (no_epoch+1)
                val_loss = valLoss / (no_valid_epoch+1)
                val_pear_loss = -valPearson / (no_valid_epoch+1)
                if best_score <= val_pear_loss:
                    best_score = val_pear_loss
                    if rank == 0:
                        checkpoint_name = 'best_{}.pkl'.format(n_iter)
                        torch.save({'model_state_dict': net.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict()},
                                   os.path.join(output_directory, checkpoint_name))
                        print('new best model at iteration %s is saved' % n_iter)
                else:
                    pass
                    #scheduler.step(-val_pear_loss)
                    
                if rank == 0:
                    print("iteration: {} \ training loss: {} \ training pearson: {} \ validation loss: {} \ validation pearson: {}".format(n_iter, train_loss, train_pearson, val_loss, val_pear_loss))
                    tb.add_scalar("Log-Train-Loss", train_loss, n_iter)
                    tb.add_scalar("Log-Validation-Loss", val_loss, n_iter)
                    tb.add_scalar("Validation-Pearson", val_pear_loss, n_iter)


        # save checkpoint
        if n_iter > 0 and n_iter % iters_per_ckpt == 0 and rank == 0:
            checkpoint_name = '{}.pkl'.format(n_iter)
            torch.save({'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       os.path.join(output_directory, checkpoint_name))
            print('model at iteration %s is saved' % n_iter)

        n_iter += 1
        print("--- %s seconds ---" % (time.time() - start_time))
    # Close TensorBoard.
    if rank == 0:
        tb.close()

if __name__ == "__main__":

    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--train_config_file', type=str, default='train-shallow.json', 
                        help='JSON file for configuration')
    parser.add_argument('-w', '--window_length', type=int, default=3,
                        help='window length of the frame')
    parser.add_argument('-hop', '--hop_length', type=float, default=0.5,
                        help='hop length between windows')
    parser.add_argument('-r', '--rank', type=int, default=0,
                        help='rank of process for distributed')
    parser.add_argument('-g', '--group_name', type=str, default='',
                        help='name of group for distributed')
    args = parser.parse_args()

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        if args.group_name == '':
            print("WARNING: Multiple GPUs detected but no distributed group set")
            print("Only running 1 GPU. Use distributed.py for multiple GPUs")
            num_gpus = 1

    if num_gpus == 1 and args.rank != 0:
        raise Exception("Doing single GPU training on rank > 0")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    train(args.train_config_file,args.window_length,args.hop_length,num_gpus, args.rank, args.group_name)