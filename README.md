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

# Running in Windows or Ubuntu
​```python
git clone https://github.com/QRICKDD/LLA.git

cd LLA

python run PROPOSE_RUN.py

python run SIGN_OPT_RUN.py

python run HSJA_ATTACK_RUN.py

​```
