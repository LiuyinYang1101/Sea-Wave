import torch
from pathlib import Path
import itertools
import numpy as np
import time
import random
from random import randrange
import math
import torchaudio.transforms as T
import pickle
###Test Streaming DataLoader with PyTorch####
class MyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, filePaths, frameLength, hopSize, loadAll=False):
        super(MyIterableDataset).__init__()
        self.loadAll = loadAll
        self.filePaths = self.group_recordings(filePaths)
        self.frameLength = frameLength
        self.hopSize = hopSize
        self.filePage = len(self.filePaths)
        self.filePool = list(range(self.filePage))
        print(self.filePage)
        random.shuffle(self.filePool)

        self.currentFileIndx = 0
        self.CurrentEEG = []
        self.CurrentAudio = []
        self.samplePosistions = []

        self.currentSampleIndex = 0
        self.loadDataToBuffer(self.currentFileIndx, loadAll)

        self.samplePosMap = []
        self.generateSamplePostion()

    def group_recordings(self, files):
        # Group recordings and corresponding stimuli.
        new_files = []
        grouped = itertools.groupby(sorted(files), lambda x: "_-_".join(x.stem.split("_-_")[:3]))
        for recording_name, feature_paths in grouped:
            new_files += [sorted(feature_paths, key=lambda x: "0" if x == "eeg" else x)]
        return new_files

    import random

    def loadDataToBuffer(self, fileIndex, loadAll):
        if loadAll == True:
            for i in range(self.filePage):
                self.CurrentEEG.append(
                    np.load(self.filePaths[self.filePool[self.currentFileIndx]][0]).astype(np.float32))
                self.CurrentAudio.append(
                    np.load(self.filePaths[self.filePool[self.currentFileIndx]][1]).astype(np.float32))
        else:
            self.CurrentEEG = np.load(self.filePaths[self.filePool[self.currentFileIndx]][0]).astype(np.float32)
            self.CurrentAudio = np.load(self.filePaths[self.filePool[self.currentFileIndx]][1]).astype(np.float32)

    def generateSamplePostion(self):
        count = 0
        if self.loadAll == True:
            for i in range(self.filePage):
                totalLength, _ = self.CurrentAudio[i].shape
                startPos = [*range(self.frameLength, totalLength + 1, self.hopSize)]
                self.samplePosMap.append(startPos)
                noData = (totalLength - self.frameLength) // self.hopSize + 1
                assert len(startPos) == noData
                count += noData
            return count
        else:
            for i in range(self.filePage):
                tempAudio = np.load(self.filePaths[i][1]).astype(np.float32)
                totalLength, _ = tempAudio.shape
                startPos = [*range(self.frameLength, totalLength + 1, self.hopSize)]
                self.samplePosMap.append(startPos)
                noData = (totalLength - self.frameLength) // self.hopSize + 1
                assert len(startPos) == noData
                count += noData
            return count

    def sample_random_data_number_in_one_batch(self, n, total):
        # Return a randomly chosen list of n nonnegative integers summing to total.
        # n: the number of total files    total: batch size
        return [x - 1 for x in self.constrained_sum_sample_pos(n, total + n)]

    def constrained_sum_sample_pos(self, n, total):
        # Return a randomly chosen list of n positive integers summing to total.Each such list is equally likely to occur."""
        dividers = sorted(random.sample(range(1, total), n - 1))
        return [a - b for a, b in zip(dividers + [total], [0] + dividers)]

    def __iter__(self):

        return self

    def __next__(self):
        if self.loadAll == True:
            if self.currentSampleIndex < len(
                    self.samplePosMap[self.filePool[self.currentFileIndx]]):  # still in the same file
                thisEnd = self.samplePosMap[self.filePool[self.currentFileIndx]][self.currentSampleIndex]
                self.currentSampleIndex += 1
                return self.CurrentEEG[self.filePool[self.currentFileIndx]][thisEnd - self.frameLength:thisEnd, :], \
                       self.CurrentAudio[self.filePool[self.currentFileIndx]][thisEnd - self.frameLength:thisEnd, :]
            else:  # move to the next file
                # print("next file")
                #### need to shuffle samples from the last file
                random.shuffle(self.samplePosMap[self.filePool[self.currentFileIndx]])
                self.currentFileIndx += 1
                self.currentSampleIndex = 0
                if self.currentFileIndx < self.filePage:  # still in the same iteration
                    # self.loadDataToBuffer(self.currentFileIndx)
                    thisEnd = self.samplePosMap[self.filePool[self.currentFileIndx]][self.currentSampleIndex]
                    self.currentSampleIndex += 1
                    return self.CurrentEEG[self.filePool[self.currentFileIndx]][thisEnd - self.frameLength:thisEnd, :], \
                           self.CurrentAudio[self.filePool[self.currentFileIndx]][thisEnd - self.frameLength:thisEnd, :]
                else:
                    # print("here 2")
                    random.shuffle(self.filePool)
                    self.currentFileIndx = 0
                    # self.loadDataToBuffer(self.currentFileIndx)
                    raise StopIteration
                    print("iteration done, restart")
        else:
            if self.currentSampleIndex < len(
                    self.samplePosMap[self.filePool[self.currentFileIndx]]):  # still in the same file
                thisEnd = self.samplePosMap[self.filePool[self.currentFileIndx]][self.currentSampleIndex]
                self.currentSampleIndex += 1
                return self.CurrentEEG[thisEnd - self.frameLength:thisEnd, :], self.CurrentAudio[
                                                                               thisEnd - self.frameLength:thisEnd, :]
            else:  # move to the next file
                # print("next file")
                #### need to shuffle samples from the last file
                random.shuffle(self.samplePosMap[self.filePool[self.currentFileIndx]])
                self.currentFileIndx += 1
                self.currentSampleIndex = 0
                if self.currentFileIndx < self.filePage:  # still in the same iteration
                    self.loadDataToBuffer(self.currentFileIndx)
                    thisEnd = self.samplePosMap[self.filePool[self.currentFileIndx]][self.currentSampleIndex]
                    self.currentSampleIndex += 1
                    return self.CurrentEEG[thisEnd - self.frameLength:thisEnd, :], self.CurrentAudio[
                                                                                   thisEnd - self.frameLength:thisEnd,
                                                                                   :]
                else:
                    # print("here 2")
                    random.shuffle(self.filePool)
                    self.currentFileIndx = 0
                    self.loadDataToBuffer(self.currentFileIndx)
                    raise StopIteration
                    print("iteration done, restart")


class CustomProcessInOrderDataset(torch.utils.data.Dataset):
    def __init__(self, filePaths, frameLength, hopSize, device):
        self.filePaths = self.group_recordings(filePaths)
        self.frameLength = frameLength
        self.hopSize = hopSize
        self.device = device
        self.filePage = len(self.filePaths)
        self.filesIndex = list(range(self.filePage))
        self.halfFile = int(self.filePage*0.5)
        # load all data to memory as continuous data
        self.EEGData = []
        self.AudioData = []
        self.EEG_on_GPU = []
        self.Aud_on_GPU = []
        self.loadDataToBuffer()
        self.convertToTensorType()
        self.noData = self.getNoData()
        # random file order
        self.random_file_order()
        # send the first half to GPU
        self.send_to_device()
        self.firstHalf = True
        self.firstHalfData = 0
        # generate sample position
        self.sampleIndxMap = []
        self.noDataOnGPU = self.generateSamplePostionOnGPU()
        self.firstHalfData = self.noDataOnGPU
        print(self.filePage," files loaded. ", self.noData, " training examples loaded ", self.halfFile, " files sent to GPU, with total data number: ", self.noDataOnGPU)
        # shuffle the training sample index to get random batch
        self.shuffleSamplesOnGPU()

    def random_file_order(self):
        random.shuffle(self.filesIndex)

    def send_to_device(self, first_half=True):
        del self.EEG_on_GPU
        del self.Aud_on_GPU
        self.EEG_on_GPU = []
        self.Aud_on_GPU = []
        if first_half==True: # send the first half to GPU
            self.firstHalf = True
            for i in range(self.halfFile):
                tempEEG = self.EEGData[self.filesIndex[i]]
                tempAud = self.AudioData[self.filesIndex[i]]

                self.EEG_on_GPU.append(tempEEG.to(self.device))
                self.Aud_on_GPU.append(tempAud.to(self.device))

        else: # send the last half to GPU
            self.firstHalf = False
            for i in range(self.halfFile,self.filePage):
                tempEEG = self.EEGData[self.filesIndex[i]]
                tempAud = self.AudioData[self.filesIndex[i]]

                self.EEG_on_GPU.append(tempEEG.to(self.device))
                self.Aud_on_GPU.append(tempAud.to(self.device))

    def generateSamplePostionOnGPU(self):
        count = 0
        for i in range(len(self.EEG_on_GPU)):
            _,totalLength = self.EEG_on_GPU[i].shape
            startPos = [*range(self.frameLength, totalLength + 1, self.hopSize)]
            for pos in startPos:
                self.sampleIndxMap.append((i, pos))
            noData = (totalLength - self.frameLength) // self.hopSize + 1
            assert len(startPos) == noData
            count += noData
        return count

    def shuffleSamplesOnGPU(self):
        random.shuffle(self.sampleIndxMap)

    def convertToTensorType(self):
        self.EEGData = [torch.from_numpy(eegtrial).permute(1,0) for eegtrial in self.EEGData]
        self.AudioData = [torch.from_numpy(audiotrial).permute(1,0) for audiotrial in self.AudioData]

    def group_recordings(self, files):
        # Group recordings and corresponding stimuli.
        new_files = []
        grouped = itertools.groupby(sorted(files), lambda x: "_-_".join(x.stem.split("_-_")[:3]))
        for recording_name, feature_paths in grouped:
            new_files += [sorted(feature_paths, key=lambda x: "0" if x == "eeg" else x)]
        return new_files

    def loadDataToBuffer(self):
        for i in range(self.filePage):
            self.EEGData.append(np.load(self.filePaths[i][0]).astype(np.float32))
            self.AudioData.append(np.load(self.filePaths[i][1]).astype(np.float32))

    def getNoData(self):
        count = 0
        for i in range(self.filePage):
            _,totalLength = self.AudioData[i].shape
            noData = (totalLength - self.frameLength) // self.hopSize + 1
            count += noData
        return count

    def __len__(self):
        return self.noData

    def __getitem__(self, idx):
        if  self.firstHalf==True:
            if idx <= self.noDataOnGPU-1:
                fileIndex = self.sampleIndxMap[idx][0]
                endIndex = self.sampleIndxMap[idx][1]
                startIndex = self.sampleIndxMap[idx][1] - self.frameLength
                return self.EEG_on_GPU[fileIndex][:,startIndex:endIndex], self.Aud_on_GPU[fileIndex][:,startIndex:endIndex]
            else: # all data on GPU has just been iterated
                # send the second half to GPU
                self.send_to_device(first_half=False)
                # generate sample position
                self.sampleIndxMap = []
                self.noDataOnGPU = self.generateSamplePostionOnGPU()
                #print("first half data finished ", len(self.EEG_on_GPU)," files sent to GPU, with total data number: ", self.noDataOnGPU)
                # shuffle the training sample index to get random batch
                self.shuffleSamplesOnGPU()
                idx_new = idx - self.firstHalfData
                assert(self.noDataOnGPU == self.noData-self.firstHalfData)
                fileIndex = self.sampleIndxMap[idx_new][0]
                endIndex = self.sampleIndxMap[idx_new][1]
                startIndex = self.sampleIndxMap[idx_new][1] - self.frameLength
                return self.EEG_on_GPU[fileIndex][:,startIndex:endIndex], self.Aud_on_GPU[fileIndex][:,startIndex:endIndex]
        else:
            idx_new = idx - self.firstHalfData
            fileIndex = self.sampleIndxMap[idx_new][0]
            endIndex = self.sampleIndxMap[idx_new][1]
            startIndex = self.sampleIndxMap[idx_new][1] - self.frameLength
            if idx < self.noData-1:
                return self.EEG_on_GPU[fileIndex][:, startIndex:endIndex], self.Aud_on_GPU[fileIndex][:,startIndex:endIndex]
            else: # the last sample from one complete iteration
                tempEEG = self.EEG_on_GPU[fileIndex][:, startIndex:endIndex]
                tempAud = self.Aud_on_GPU[fileIndex][:,startIndex:endIndex]
                # reshuffle for a new iteration
                self.random_file_order()
                self.send_to_device(first_half=True)
                # generate sample position
                self.sampleIndxMap = []
                self.noDataOnGPU = self.generateSamplePostionOnGPU()
                self.firstHalfData = self.noDataOnGPU
                #print("finish one iteration ", len(self.EEG_on_GPU), " files sent to GPU, with total data number: ", self.noDataOnGPU)
                # shuffle the training sample index to get random batch
                self.shuffleSamplesOnGPU()
                return tempEEG, tempAud

class CustomAllLoadDataset(torch.utils.data.Dataset):
    def __init__(self, filePaths, frameLength, hopSize, aug = False, toTrain = True):
        self.filePaths = self.group_recordings(filePaths)
        self.frameLength = frameLength
        self.hopSize = hopSize
        self.filePage = len(self.filePaths)
        self.aug = aug
        self.toTrain = toTrain
        # load all data to memory as continuous data
        self.EEGData = []
        self.AudioData = []
        self.loadDataToBuffer()
        if toTrain:
            self.sampleIndxMap = []
            self.noData = self.generateSamplePostion()
        else:
            self.noData = self.filePage
        #print(self.filePage," files loaded. ", self.noData, " training examples loaded, use data augementation: ", self.aug)

    def send_to_device(self, no=1):
        if no==1:
            self.EEGData = [eegtrial.cuda()  for eegtrial in self.EEGData]
            self.AudioData = [audiotrial.cuda() for audiotrial in self.AudioData]
        else:
            noFilesToSend = math.floor(self.filePage*no)
            newEEGData = []
            newAudData = []
            for i in range(self.filePage):
                tempEEG = self.EEGData.pop(0)
                tempAud = self.AudioData.pop(0)
                if i<noFilesToSend:
                    newEEGData.append(tempEEG.to(device))
                    newAudData.append(tempAud.to(device))
                else:
                    newEEGData.append(tempEEG)
                    newAudData.append(tempAud)
            assert (len(newAudData)==self.filePage)
            self.EEGData = newEEGData
            self.AudioData = newAudData




    def convertToTensorType(self):
        self.EEGData = [torch.from_numpy(eegtrial).permute(1,0) for eegtrial in self.EEGData]
        self.AudioData = [torch.from_numpy(audiotrial).permute(1,0) for audiotrial in self.AudioData]

    def group_recordings(self, files):
        # Group recordings and corresponding stimuli.
        new_files = []
        grouped = itertools.groupby(sorted(files), lambda x: "_-_".join(x.stem.split("_-_")[:3]))
        for recording_name, feature_paths in grouped:
            new_files += [sorted(feature_paths, key=lambda x: "0" if x == "eeg" else x)]
        return new_files

    def loadDataToBuffer(self):
        for i in range(self.filePage):
            self.EEGData.append(np.load(self.filePaths[i][0]).astype(np.float32))
            self.AudioData.append(np.load(self.filePaths[i][1]).astype(np.float32))

    def generateSamplePostion(self):
        count = 0
        for i in range(self.filePage):
            totalLength, _ = self.AudioData[i].shape
            if totalLength > self.frameLength:
                startPos = [*range(self.frameLength, totalLength + 1, self.hopSize)]
                for pos in startPos:
                    self.sampleIndxMap.append((i, pos))
                noData = (totalLength - self.frameLength) // self.hopSize + 1
                #print("file: ", i, " total length: ", totalLength, " frameLength: ", self.frameLength, " ", noData, len(startPos))
                assert len(startPos) == noData
                count += noData
            else:
                pass
        return count

    def __len__(self):
        return self.noData

    def __getitem__(self, idx):
        if self.toTrain:
            fileIndex = self.sampleIndxMap[idx][0]
            endIndex = self.sampleIndxMap[idx][1]
            startIndex = endIndex - self.frameLength
            
            return self.EEGData[fileIndex][:,startIndex:endIndex], self.AudioData[fileIndex][:,startIndex:endIndex]
        else:
            return self.EEGData[idx], self.AudioData[idx]
            


class CustomAllLoadDataset_distributed(torch.utils.data.Dataset):
    def __init__(self, filePaths, frameLength, hopSize):
        self.filePaths = self.group_recordings(filePaths)
        self.frameLength = frameLength
        self.hopSize = hopSize
        self.filePage = len(self.filePaths)
        # load all data to memory as continuous data
        self.EEGData = []
        self.AudioData = []
        self.loadDataToBuffer()

        self.sampleIndxMap = []
        self.noData = self.generateSamplePostion()
        print(self.filePage," files loaded. ", self.noData, " training examples loaded")

    def send_to_device(self, device, no=1):
        if no==1:
            self.EEGData = [eegtrial.to(device)  for eegtrial in self.EEGData]
            self.AudioData = [audiotrial.to(device) for audiotrial in self.AudioData]
        else:
            noFilesToSend = math.floor(self.filePage*no)
            newEEGData = []
            newAudData = []
            for i in range(self.filePage):
                tempEEG = self.EEGData.pop(0)
                tempAud = self.AudioData.pop(0)
                if i<noFilesToSend:
                    newEEGData.append(tempEEG.to(device))
                    newAudData.append(tempAud.to(device))
                else:
                    newEEGData.append(tempEEG)
                    newAudData.append(tempAud)
            assert (len(newAudData)==self.filePage)
            self.EEGData = newEEGData
            self.AudioData = newAudData

    def convertToTensorType(self):
        self.EEGData = [torch.from_numpy(eegtrial).permute(1,0) for eegtrial in self.EEGData]
        self.AudioData = [torch.from_numpy(audiotrial).permute(1,0) for audiotrial in self.AudioData]

    def group_recordings(self, files):
        # Group recordings and corresponding stimuli.
        new_files = []
        grouped = itertools.groupby(sorted(files), lambda x: "_-_".join(x.stem.split("_-_")[:3]))
        for recording_name, feature_paths in grouped:
            new_files += [sorted(feature_paths, key=lambda x: "0" if x == "eeg" else x)]
        return new_files

    def loadDataToBuffer(self):
        for i in range(self.filePage):
            self.EEGData.append(np.load(self.filePaths[i][0]).astype(np.float32))
            self.AudioData.append(np.load(self.filePaths[i][1]).astype(np.float32))

    def generateSamplePostion(self):
        count = 0
        for i in range(self.filePage):
            totalLength, _ = self.AudioData[i].shape
            startPos = [*range(self.frameLength, totalLength + 1, self.hopSize)]
            for pos in startPos:
                self.sampleIndxMap.append((i, pos))
            noData = (totalLength - self.frameLength) // self.hopSize + 1
            assert len(startPos) == noData
            count += noData
        return count

    def __len__(self):
        return self.noData

    def __getitem__(self, idx):
        fileIndex = self.sampleIndxMap[idx][0]
        endIndex = self.sampleIndxMap[idx][1]
        startIndex = self.sampleIndxMap[idx][1] - self.frameLength
        return self.EEGData[fileIndex][:,startIndex:endIndex], self.AudioData[fileIndex][:,startIndex:endIndex]
        
class CustomAllLoadRawWaveDataset(torch.utils.data.Dataset):
    def __init__(self, eeg_filePaths, audio_filePaths, frameLength, hopSize, resample_rate_eeg, resample_rate_audio):
        self.eeg_filePaths = eeg_filePaths
        self.audio_filePaths = audio_filePaths
        self.frameLength = frameLength
        self.hopSize = hopSize
        self.resample_rate_eeg = resample_rate_eeg
        self.resample_rate_audio = resample_rate_audio
        
        self.eeg_window_length = int(self.frameLength*self.resample_rate_eeg)
        self.audio_window_length = int(self.frameLength*self.resample_rate_audio)
        self.eeg_hop_length = int(self.hopSize*self.resample_rate_eeg)
        self.audio_hop_length = int(self.hopSize*self.resample_rate_audio)
        
        self.filePage = len(self.eeg_filePaths)
        # load all data to memory as continuous data
        self.EEGData, self.AudioData = self.loadAllData()
        

        self.sampleIndxMap = []
        self.noData = self.generateSamplePostion()
        print("data load finished: ", self.filePage, " files in total ", self.noData, " samples in total")
    def group_recordings(self, files):
        # Group recordings and corresponding stimuli.
        new_files = []
        grouped = itertools.groupby(sorted(files), lambda x: "_-_".join(x.stem.split("_-_")[:3]))
        for recording_name, feature_paths in grouped:
            new_files += [sorted(feature_paths, key=lambda x: "0" if x == "eeg" else x)]
        return new_files

    def findStimulusFile(self,fileName):
        for path in self.audio_filePaths:
            if path.stem == Path(fileName).stem:
                return path
        
    def resampleData(self,original_data,fs,fs_new):
        resampler = T.Resample(fs, fs_new, dtype=original_data.dtype)
        resampled_waveform = resampler(original_data)
        return resampled_waveform

    def loadAllData(self):
        eegDataAll = []
        audioDataAll = []
        for eeg_file in self.eeg_filePaths:
            with open(eeg_file, 'rb') as f:
                data = pickle.load(f)
                eegData = torch.permute(torch.from_numpy(data['eeg']),(1,0))
                #print(eegData.shape)
                stimulus_name = data['stimulus']
                stimulus_file = self.findStimulusFile(stimulus_name)
                eeg_resampled = self.resampleData(eegData,data['fs'],500)
                #print(eeg_resampled.shape)
                # audio
                b = np.load(stimulus_file)
                audio_data = torch.unsqueeze(torch.from_numpy(b['audio']),0)
                sample_rate = b['fs']
                #print(audio_data.shape)
                audio_resampled = self.resampleData(audio_data,sample_rate,4000)
                #print(audio_resampled.shape)
                eegDataAll.append(eeg_resampled)
                audioDataAll.append(audio_resampled)
        return eegDataAll, audioDataAll
    
    def generateSynchronizedPos(self,audio,eeg):
        startPos_eeg = [*range(self.eeg_window_length, eeg.shape[1] + 1, self.eeg_hop_length)]
        startPos_audio = [*range(self.audio_window_length, audio.shape[1] + 1, self.audio_hop_length)]
        while len(startPos_eeg) != len(startPos_audio):
            if len(startPos_eeg) < len(startPos_audio):
                startPos_audio.pop()
            else:
                startPos_eeg.pop()
        return startPos_eeg, startPos_audio

    def generateSamplePostion(self):
        count = 0
        for i in range(self.filePage):
            totalLength_audio = self.AudioData[i].shape[0]
            startPos_eeg, startPos_audio = self.generateSynchronizedPos(self.AudioData[i],self.EEGData[i])
            for index, pos in enumerate(startPos_eeg):
                self.sampleIndxMap.append((i, pos,startPos_audio[index]))
            noData = len(startPos_eeg)
            print("file ", i, "load: ", noData)
            count += noData
        return count
    
    def send_to_device(self, device):
        self.EEGData = [eegtrial.to(device)  for eegtrial in self.EEGData]
        self.AudioData = [audiotrial.to(device) for audiotrial in self.AudioData]
            
    def __len__(self):
        return self.noData

    def __getitem__(self, idx):
        jit = random.randint(-12, 12)
        fileIndex = self.sampleIndxMap[idx][0]
        eeg_endIndex = self.sampleIndxMap[idx][1]
        if eeg_endIndex - self.eeg_window_length > 16 and eeg_endIndex+16 < eeg_endIndexself.EEGData[fileIndex].shape[1]:
            eeg_endIndex = eeg_endIndex + jit
        audio_endIndex = self.sampleIndxMap[idx][2]
        eeg_startIndex = eeg_endIndex - self.eeg_window_length
        audio_startIndex = audio_endIndex - self.audio_window_length
        
        return self.EEGData[fileIndex][:,eeg_startIndex:eeg_endIndex], self.AudioData[fileIndex][:,audio_startIndex:audio_endIndex]