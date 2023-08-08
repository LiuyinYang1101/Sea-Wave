import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

import sys
sys.path.insert(0, 'C:\\PhD\\projects\\IEEE SP\\github repo\\EEGWaveRegressor\\utils')
sys.path.insert(0, 'C:\\PhD\\projects\\IEEE SP\\github repo\\EEGWaveRegressor\\models')

from util import rescale, find_max_epoch, print_size, pearson_loss
from CustomDatasetPytorch import CustomAllLoadDataset
from pathlib import Path
from distributed_util import init_distributed, apply_gradient_allreduce, reduce_tensor
import time
from EEGWaveRegressor import WaveNet_regressor as EEGWaveRegressor
from torch.utils.data import DataLoader
from cyclic_scheduler import CyclicLRWithRestarts

# return all sub's name
def find_all_subjects(data_folder):
    subject_set = set()
    for path in Path(data_folder).resolve().glob("train_-_*"):
        subName = path.stem.split("_-_")[1]
        subject_set.add(subName)
    return subject_set

# return all subs that have finished fine-tuning
def find_fine_tuned_subjects(data_folder):
    subject_set = set()
    for p in Path(data_folder).glob("*"):
        subName = p.stem.split("_")[-1]
        sub_sub_name = subName.split("-")[0]
        if sub_sub_name == 'sub':
            subject_set.add(subName)
        else:
            pass
    return subject_set

def find_subject_files(subject, data_folder,ifTrain):
    stimulus_features = ["envelope"]
    features = ["eeg"] + stimulus_features
    #print("try to search for ", subject)
    if ifTrain: # find train files:
        target_train_files = []
        for path in Path(data_folder).resolve().glob("train_-_*"):
            if path.stem.split("_-_")[1] == subject and path.stem.split("_-_")[-1].split(".")[0] in features:
                target_train_files.append(path)
        for path in Path(data_folder).resolve().glob("val_-_*"):
            if path.stem.split("_-_")[1] == subject and path.stem.split("_-_")[-1].split(".")[0] in features:
                target_train_files.append(path)
        return target_train_files
    else: # find test files:
        target_test_files = []
        for path in Path(data_folder).resolve().glob("test_-_*"):
            if path.stem.split("_-_")[1] == subject and path.stem.split("_-_")[-1].split(".")[0] in features:
                target_test_files.append(path)
        return target_test_files
    
def train(train_conig_file, window_length, hop_length, num_gpus, rank, group_name):
    
    f_write = open("fineTuneLogs.txt", "w")
    with open(train_conig_file) as fp:
        config = json.load(fp)
        train_config = config["train_config"]  # training parameters
        global dist_config
        dist_config = config["dist_config"]  # to initialize distributed training
        global wavenet_config
        wavenet_config = config["wavenet_config"]  # to define wavenet
        global trainset_config
        trainset_config = config["trainset_config"]  # to load trainset


    output_directory = train_config["output_directory"]
    tensorboard_directory = train_config["tensorboard_directory"]
    ckpt_iter = train_config["ckpt_iter"]
    iters_per_ckpt = train_config["iters_per_ckpt"]
    iters_per_logging = train_config["iters_per_logging"]
    n_iters = train_config["n_iters"]
    learning_rate = train_config["learning_rate"]
    batch_size = train_config["batch_size_per_gpu"]

    pretrained_model_path = trainset_config["pretrained_model_path"]
    data_folder = Path(config["trainset_config"]["data_path"])
    stimulus_features = ["envelope"]
    features = ["eeg"] + stimulus_features
    # get all subjects:
    all_subs = find_all_subjects(data_folder)
    
    fine_tune_folder = config["gen_config"]["finetune_folder"]
    fine_Subs = find_fine_tuned_subjects(fine_tune_folder)
    remained_subs = sorted(all_subs.difference(fine_Subs))
    
    print("in total: ", len(remained_subs), " subjects")
    # distributed running initialization
    if num_gpus > 1:
        init_distributed(rank, num_gpus, group_name, **dist_config)
    
    # predefine model
    net = EEGWaveRegressor(**wavenet_config).cuda()
    print_size(net)
    # apply gradient all reduce
    if num_gpus > 1:
        net = apply_gradient_allreduce(net)
        
    train_loader = None
    test_loader = None
    bestPearson = []
    bestIter = []
    #all_subs = ["sub-051"]#["sub-033","sub-020","sub-043","sub-008","sub-061","sub-051","sub-036","sub-019","sub-006","sub-070","sub-052"]
    print("start fine tuing for ", len(remained_subs), " subjects")
    for sub in remained_subs:
        # generate experiment (local) path
        local_path = "Regressor_mse_pearson_fine_tune_ch{}_".format(wavenet_config["res_channels"])
        local_path = local_path+sub
        output_directory = train_config["output_directory"]
        # Create tensorboard logger.
        if rank == 0:
            tb = SummaryWriter(os.path.join('exp', local_path, tensorboard_directory))
        sub_train = find_subject_files(sub,data_folder, True)
        sub_test = find_subject_files(sub,data_folder, False)
        print(len(sub_train), len(sub_test))
        
        # Get shared output_directory ready
        output_directory = os.path.join('exp', local_path, output_directory)
        if rank == 0:
            if not os.path.isdir(output_directory):
                os.makedirs(output_directory)
                os.chmod(output_directory, 0o775)
            print("output directory for: ", sub, output_directory, flush=True)
        # define optimizer
        optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)
        # load checkpoint
        try:
            ckpt_iter = 62
            # load checkpoint file
            model_path = os.path.join(pretrained_model_path, '{}.pkl'.format(ckpt_iter))
            checkpoint = torch.load(model_path, map_location='cpu')
            net.load_state_dict(checkpoint['model_state_dict'])
            print("gelu skip model loaded at iter: ", ckpt_iter)
        except Exception as inst:
            print(inst)
            print("gelu skip model loading failed ")
   
        # load training data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        del train_loader
        train_dataset = CustomAllLoadDataset(sub_train, int(window_length*64), int(hop_length*64),aug=False)
        train_dataset.convertToTensorType()
        train_dataset.send_to_device()
        scheduler = CyclicLRWithRestarts(optimizer, batch_size, train_dataset.__len__(), restart_period=10, t_mult=1.2, policy="cosine")
        if num_gpus > 1:
            train_sampler = DistributedSampler(train_dataset)  
            random_sampler = RandomSampler(train_sampler)
        else: 
            random_sampler = None
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=random_sampler, num_workers=0, pin_memory=False)

        if rank == 0: # only evaluate on the first GPU
            del test_loader
            test_dataset = CustomAllLoadDataset(sub_test, int(3*64), int(0.5*64),False,False)
            test_dataset.convertToTensorType()
            test_dataset.send_to_device()
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle = False)
        mse_loss = nn.MSELoss()


        # training
        n_iter = 0
        this_sub_best_pearson = 0
        this_sub_best_iter = 0
        tobreak = False
        while n_iter <= 20:
            start_time = time.time()
            scheduler.step()
            batch_mse_loss = 0
            batch_pear_loss = 0
            no_epoch = 0
            #print("new iteration")
            net = net.train()
            for i, data in enumerate(train_loader, 0):
                #print("new batch")
                no_epoch = i
                # get the inputs; data is a list of [inputs, labels]
                eeg, audio = data[0], data[1]
                #print("to cuda")
                # back-propagation
                optimizer.zero_grad()
                output = net(eeg)
                # loss calculation
                loss_mse = mse_loss(output, audio)
                loss_per = pearson_loss(output, audio, axis=-1)
                #print(loss_per.item())
                loss = 0.6*loss_mse+0.4*loss_per
                if(torch.isnan(loss).any()==True):
                    print("nan detected")
                if num_gpus > 1:
                    reduced_pear_loss = reduce_tensor(loss_per.data, num_gpus).item()
                    reduced_mse_loss = reduce_tensor(loss_mse.data, num_gpus).item()
                    batch_mse_loss += reduced_mse_loss
                    batch_pear_loss += reduced_pear_loss
                else:
                    reduced_mse_loss = loss_mse.item()
                    reduced_pear_loss = loss_per.item()
                    batch_mse_loss += reduced_mse_loss
                    batch_pear_loss += reduced_pear_loss

                loss.backward()
                #torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
                optimizer.step()
                scheduler.batch_step()
   
            print("iteration done")
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
                    train_mse = batch_mse_loss / (no_epoch+1)
                    val_mse = valLoss / (no_valid_epoch+1)
                    val_pear_loss = -valPearson / (no_valid_epoch+1)
                    if val_pear_loss > this_sub_best_pearson:
                        this_sub_best_iter = n_iter
                        this_sub_best_pearson = val_pear_loss
                    else:
                        tobreak = True
                    output_msg = "iteration: {} \ training mse: {} \ training pearson: {} \ validation mse: {} \ validation pearson: {}\n".format(n_iter, train_mse, train_pearson, val_mse, val_pear_loss)
                    print(output_msg)
                    f_write.write(output_msg)
                    tb.add_scalar("Train-MSE", train_mse, n_iter)
                    tb.add_scalar("Train-Pearson", train_pearson, n_iter)
                    tb.add_scalar("Validation-MSE", val_mse, n_iter)
                    tb.add_scalar("Validation-Pearson", val_pear_loss, n_iter)

            # save checkpoint
            if n_iter >= 0 and n_iter % iters_per_ckpt == 0 and rank == 0:
                checkpoint_name = '{}.pkl'.format(n_iter)
                torch.save({'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(output_directory, checkpoint_name))
                print('model at iteration %s is saved' % n_iter)
            if tobreak == True:
                break
            n_iter += 1
            print("--- %s seconds ---" % (time.time() - start_time))
        # Close TensorBoard.
        if rank == 0:
            tb.close()
            bestPearson.append(this_sub_best_pearson)
            bestIter.append(this_sub_best_iter)
    # finished all subjects:
    if rank == 0:
        i = 0
        sum_all_pearson = 0
        for sub in all_subs:
            sum_all_pearson += bestPearson[i]
            print(sub," the highest pearson: ",bestPearson[i], " at iteration: ",bestIter[i])
            i = i+1
        print("On average: pearson = ", sum_all_pearson/i)
        f_write.close()
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