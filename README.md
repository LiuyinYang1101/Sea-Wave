# Sea-Wave

Sea-Wave is an open-source repository for a novel approach to auditory EEG decoding. This project presents the adapted Wavenet model, a state-of-the-art architecture for auditory EEG data decoding. Check our paper:

## Introduction

This study presents Sea-Wave, a WaveNet-based architecture for reconstructing speech envelopes from auditory EEG. The model is an extension of our submission for the Auditory EEG Challenge of the ICASSP Signal Processing Grand Challenge 2023. We improve upon our prior work by evaluating model components and hyperparameters through an ablation study and hyperparameter search, respectively. Our best subject-independent model achieves a Pearson correlation of 22.58% on seen and 11.58% on unseen subjects. After subject-specific finetuning, we find an average relative improvement of 30% for the seen subjects and a Pearson correlation of 56.57% for the best subject. Finally, we explore a number of model visualizations to obtain a better understanding of the model, the differences across subjects and the EEG features that relate to auditory perception.

## Paper Reference

If you use Sea-Wave in your research or work, please consider citing our paper:

```
@INPROCEEDINGS{a-Wavenet,
  author={Van Dyck, Bob and Yang, Liuyin and Van Hulle, Marc M.},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Decoding Auditory EEG Responses Using an Adapted Wavenet}, 
  year={2023},
  volume={},
  number={},
  pages={1-2},
  doi={10.1109/ICASSP49357.2023.10095420}}
```

## Acknowledgments

We would like to acknowledge the following papers and implementations, which have significantly influenced our research and contributed to the development of Sea-Wave:

- FloWaveNet: https://github.com/ksw0306/FloWaveNet/blob/master/modules.py
- DiffWave: https://github.com/philsyn/DiffWave-Vocoder

We are grateful to the authors of these works for their valuable contributions to the field.

## Environment


## Getting Started

To get started with Sea-Wave, follow the instructions below:

1. Clone this repository to your local machine.
2. Install the required dependencies (list the dependencies and their versions, if applicable).
3. Download the EEG data and preprocessed datasets as described in the competition documentation: split_data more specifically.
4. Run the model training script, specifying the desired hyperparameters and dataset paths in the corresponding .json file.
5. Evaluate the trained model's performance on held-out subjects and held-out stories.

## Usage

to train (train the subject-independent model): go to the train folder, and execute the following script:
```
python distributed_train.py
```

to finetune (train the subject-dependent model): go to the finetune folder, and execute the following script:
```
python distributed_train_fine_tune.py
```

## Model Interpretability

One of the key strengths of Sea-Wave is its interpretability. We also provide scripts to analyze the model. These scripts are in the analysis folder (which will be updated soon) 

## License

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


## Contact

For any inquiries or questions, please contact liuyin.yang@kuleuven.be
