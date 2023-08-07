# EEGWaveRegressor

EEGWaveRegressor is an open-source repository for a novel approach to auditory EEG decoding. This project presents the EEGWaveRegressor model, a state-of-the-art architecture for auditory EEG data decoding. Check our paper:

## Introduction

This study introduces the EEGWaveRegressor, a novel model for auditory EEG decoding as an extension of our adapted Wavenet model, the model that ranked second in the Auditory EEG Challenge regression subtask of the ICASSP Signal Processing Grand Challenge 2023. We introduce EEGWaveRegressor which resulted in superior performance compared to state-of-the-art models as well as good interpretability. The best model achieved a Pearson correlation of 0.2258 on held-out stories and 0.1158 on held-out subjects. After fine-tuning, the best subject could reach a Pearson score above 0.5.

## Paper Reference

If you use EEGWaveRegressor in your research or work, please consider citing our paper:

\[@INPROCEEDINGS{a-Wavenet,
  author={Van Dyck, Bob and Yang, Liuyin and Van Hulle, Marc M.},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Decoding Auditory EEG Responses Using an Adapted Wavenet}, 
  year={2023},
  volume={},
  number={},
  pages={1-2},
  doi={10.1109/ICASSP49357.2023.10095420}}]

## Acknowledgments

We would like to acknowledge the following papers and implementations, which have significantly influenced our research and contributed to the development of EEGWaveRegressor:

- \[Other Paper 1\]
- \[Other Paper 2\]
- \[Other Paper 3\]

We are grateful to the authors of these works for their valuable contributions to the field.

## Environment


## Getting Started

To get started with EEGWaveRegressor, follow the instructions below:

1. Clone this repository to your local machine.
2. Install the required dependencies (list the dependencies and their versions, if applicable).
3. Download the EEG data and preprocessed datasets as described in the accompanying documentation.
4. Run the model training script, specifying the desired hyperparameters and dataset paths.
5. Evaluate the trained model's performance on held-out subjects and held-out stories.

## Usage

\[Provide usage instructions and examples here\]

## Model Interpretability

One of the key strengths of EEGWaveRegressor is its interpretability. We also provide scripts to analyze the model. These scripts are in the analysis folder 

## License

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


## Contact

For any inquiries or questions, please contact liuyin.yang@kuleuven.be
