# Local Low-frequency Attack
Code for Decision-Based Attack to Speaker Recognition System via Local Low-Frequency Perturbation by Jiacheng Deng, Li Dong.

![avatar](https://raw.githubusercontent.com/QRICKDD/LLA/master/picture/temp_picture.png)

Comparison of different attacking methods. Top row: from the left to right are the source audio, target audio and their residuals.
Second row to the last row: the results for ours, HSJA, QEBA-F, and SIGN-OPT; from the left to the right are results for 5K, 15K and 25K queries. For comparison, the adversarial perturbation (orange) is plotted, superimposing on the residual(blue). Better view in color version.

# About Prefiles
Prefiles can be accessed from [Cloud Disk](https://drive.google.com/drive/folders/1RMaPPxeuwSoyGXAMV4E3vnLu1Q-p5M2x?usp=sharing).

After downloading, please put them in the prefile folder.

# Dependencies
The code for our paper runs with Python 3.8 and requires Pytorch of version 1.8.1 or higher. Please pip install the following packages:
* numpy
* soudfile
* torchaudio
* pytorch-cuda

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
# Carry out QEBA-F attack based targeted attack on provided samples.
python run QEBA_F_RUN.py

# Results are stored in myresult/lib or hsjaresult/timit or signresult/lib.
# For each perturbed audio, save in myresult/libaudio or myresult/timitaudio.
```

See `PROPOSE_RUN.py`, `HSJA_RUN.py`, `QEBA_F_RUN.py` and `SIGN_OPT_RUN.py` for details. 

# Additional experiments on whether gender affects attack efficiency

The experiment tested male speakers MBCG0 and MABC0 and female speakers ELMA0 and FLMA0 in the TIMIT dataset, where - - Indicates that the required query budget exceeds 25K.

The experimental results show that the inner-gender attacks are more successful than the targeted attacks between different genders.

![avatar](https://s3.bmp.ovh/imgs/2021/12/037b49f152e5a206.png)

![avatar](https://s3.bmp.ovh/imgs/2021/12/ff5eaecc68fa8cb8.png)

Relevant audio files are saved in the FMAttackDataset folder.

# About my other work and paper
https://gitee.com/djc_QRICK

