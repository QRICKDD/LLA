# Local Low-frequency Attack
Code for Decision-Based Attack to Speaker Recognition System via Local Low-Frequency Perturbation by Jiacheng Deng, Li Done.
![avatar](https://s3.bmp.ovh/imgs/2021/11/5612c130ea4f8dc9.jpg)

Comparison of different attacking methods. Top row: from the left to right are the source audio $x$, target audio $x^t$ and their residuals $x^t-x$.
Second row to the last row: the results for ours, HSJA, and SIGN-OPT; from the left to the right are results for 5K, 15K and 25K queries. For comparison, the adversarial perturbation (orange) is plotted, superimposing on the residual $x^t − x$ (blue). Better view in color version.

# About Prefiles
All pre-train models and files and be accessed from [Cloud Disk](https://drive.google.com/drive/folders/1RMaPPxeuwSoyGXAMV4E3vnLu1Q-p5M2x?usp=sharing).

After downloading, please put them in the prefile folder.

# Dependencies
The code for our paper runs with Python 3.8 and requires Pytorch of version 1.8.1 or higher. Please pip install the following packages:
* numpy
* soudfile
* torchaudio
* pytorch

# Running in Docker, MacOS or Ubuntu
We provide as an example the source code to run LLA Attack on a SincNet trained on TIMIT and Librispeech. Run the following commands in shell:

```shell
###############################################
# Omit if already git cloned.
git clone https://github.com/QRICKDD/LLA.git
cd LLA
############################################### 
# Carry out LLA attack based targeted attack on provided samples.
python run PROPOSE_RUN.py
# Carry out HSJA attack based targeted attack on provided samples.
python run HSJA_ATTACK_RUN.py
# Carry out SIGN-OPT attack based targeted attack on provided samples.
python run SIGN_OPT_RUN.py

# Results are stored in myresult/lib or hsjaresult/timit or signresult/lib.
# For each perturbed audio, save in myresult/libaudio or myresult/timitaudio.
```

See `PROPOSE_RUN.py`, `HSJA_RUN.py` and `SIGN_OPT_RUN.py` for details. 
