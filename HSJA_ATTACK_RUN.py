# -- coding: utf-8 --
from HJSA_ATTACK import *
from utils import Sincnet
import torch
import numpy as np
import soundfile as sf
import librosa.display
import matplotlib.pyplot as plt
import pickle
import utils
import os

# 初始化模型 模型在gpu上

MODE = "TIMIT"
abs_path=os.getcwd()
is_test=False

def mkd(name):
    if os.path.exist(name)==False:
        os.mkdir(os.path.join(abs_path,name))
model = Sincnet.get_speaker_model(MODE)
speaker_label, label_speaker = Sincnet.get_speaker_label(MODE)

if MODE=="Librispeech":
    save_dir = "hsjaresult\lib"
    save_adv_dir="hsjaresult\libaudio"
    mkd(save_dir)
    mkd(save_adv_dir)
    attackdir = r"AttackDataset\lib-attack-audio"
    targetdir = r"AttackDataset\lib-target-audio"
else:
    save_dir = r"hsjaresult\timit"
    save_adv_dir = r"hsjaresult\timitaudio"
    mkd(save_dir)
    mkd(save_adv_dir)
    attackdir = r"AttackDataset\timit-attack-audio"
    targetdir = r"AttackDataset\timit-target-audio"
o_audio_files = os.listdir(attackdir)
o_audio_files.sort(key=lambda item: int(item.split('-')[0]))
t_audio_files = os.listdir(targetdir)
t_audio_files.sort(key=lambda item: int(item.split('-')[0]))

if is_test:
    save_dir="expH"
    save_adv_dir = "expH"


start=[1,2,3,4]
flag=False

for a, t in zip(o_audio_files, t_audio_files):
    if int(a.split('-')[0]) not in start:
        continue
    print("now attack audio:", a)
    real_data, fs = sf.read(os.path.join(attackdir, a))
    real_data = real_data / np.linalg.norm(real_data, ord=np.inf)
    real_data *= 0.95
    target_data, fs = sf.read(os.path.join(targetdir, t))
    target_data = target_data / np.linalg.norm(target_data, ord=np.inf)
    target_data *= 0.95
    if MODE=="Librispeech":
        real_name=a.split('-')[2]
        target_name = t.split('-')[2]
    elif MODE=="TIMIT":
        real_name = a.split('-')[-1].split('.')[0].lower()
        target_name = t.split('-')[-1].split('.')[0].lower()
    real_index=speaker_label[real_name]
    target_index = speaker_label[target_name]
    #test is ok
    pid, _ = Sincnet.sentence_test(model,torch.from_numpy(target_data).float().cuda())
    if pid != target_index:
        print("error")
        continue

    dl=a.split('-')[0]
    fname="test-{}".format(dl)

    attack=HSJA(MODE,SAVE_DIR_PATH=os.path.join(save_dir,fname))

    adv = attack(real_data[None,...][None,...],target_index,target_data[None,...][None,...])

    aname="adv-{}.wav".format(dl)
    sf.write(os.path.join(abs_path,save_adv_dir,aname),adv,16000)
    del (attack)

    if is_test==True:
        break


