from torch.utils.data import DataLoader
from CustomDatasetPytorch import CustomAllLoadDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
import torch
def load_train_loader(train_files_path, window_length, hop_length, device, aug, num_gpus=2):
    train_dataset = CustomAllLoadDataset(train_files_path, int(window_length*64), int(hop_length*64),aug=aug)
    train_dataset.convertToTensorType()
    train_dataset.send_to_device()
    #print(num_gpus, " to distribute dataset")
    # distributed sampler
    if num_gpus > 1:
        train_sampler = DistributedSampler(train_dataset)  
        random_sampler = RandomSampler(train_sampler)
    else: 
        random_sampler = None
    trainloader = torch.utils.data.DataLoader(train_dataset, 
                                              batch_size=1,  
                                              sampler=random_sampler,
                                              num_workers=0,
                                              pin_memory=False)
    return trainloader