# Loop VQ-VAE Offical

This repository is implementation of "Symbolic Music Loop Generation with Neural Discrete Representations" (accepted at ISMIR 2022). The outcomes are generated loops of 8 bars consisting of bass and drum. We take the following steps; 1) *loop extraction* from MIDI datasets using our loop detector trained by the structure of loop raw audio, 2) *loop generation* from an autoregressive model trained by discrete latent codes of the extracted loops.

## Getting Started

### Environments

* Python 3.8.8
* Ubuntu 20.04.2 LTS
* Read requirements.txt for other Python libraries

### Data Download

* Looperman Dataset (https://www.looperman.com)
* Lakh MIDI Dataset (LMD-full) (https://colinraffel.com/projects/lmd/)

### Data Preprocess at Section 3.2.1

* **preprocess_wav.ipynb** is to transform *Xwav* to *Cwav*
* **preprocess_midi.ipynb** is to transform *Xmidi* to *Cmidi*

### Loop Extraction at Section 3.2.2

* **loop_detection.ipynb** is to train the loop detector and extract loop samples from it

### Loop Generation at Section 3.3

* **VQVAE_main.ipynb** is to train VQ-VAE and obtain quantized embeddings
* **VQVAE_LSTM.ipynb** is to train autoregressive model with discrete representations

## Samples
You can listen our generated samples on Google Drive (https://drive.google.com/drive/folders/1cZmeQUJRiI0964cSynKkoSp5Gg2YJPZj?usp=sharing)

## Appreciation
Special thanks for DaeHan Ahn (University of Ulsan). He contributed to improve the quality of the paper by re-organizing the content structure.

## References
Sangjun Han, Hyeongrae Ihm, Moontae Lee, Woohyung Lim (LG AI Research), "Symbolic Music Loop Generation with Neural Discrete Representations", Proc. of the 23rd International Society for Music Information Retrieval Conference, 2022
