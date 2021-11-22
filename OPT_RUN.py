# -- coding: utf-8 --
from OPT_ATTACK import *
from utils import Sincnet
import torch
import numpy as np
import soundfile as sf
import librosa.display
import matplotlib.pyplot as plt
import pickle
import utils
import os


MODE = "TIMIT"
abs_path=os.getcwd()

model = Sincnet.get_speaker_model(MODE)
speaker_label, label_speaker = Sincnet.get_speaker_label(MODE)
def mkd(name):
    if os.path.exist(name)==False:
        os.mkdir(os.path.join(abs_path,name))
        
if MODE=="Librispeech":
    save_dir = "optresult\lib"
    save_adv_dir="optresult\libaudio"
    mkd(save_dir)
    mkd(save_adv_dir)
    attackdir = r"AttackDataset\lib-attack-audio"
    targetdir = r"AttackDataset\lib-target-audio"
else:
    save_dir = r"optresult\timit"
    save_adv_dir = r"optresult\timitaudio"
    mkd(save_dir)
    mkd(save_adv_dir)
    attackdir = r"AttackDataset\timit-attack-audio"
    targetdir = r"AttackDataset\timit-target-audio"
o_audio_files = os.listdir(attackdir)
o_audio_files.sort(key=lambda item: int(item.split('-')[0]))
t_audio_files = os.listdir(targetdir)
t_audio_files.sort(key=lambda item: int(item.split('-')[0]))

#Timit no id 16 25 34

start=1
flag=False

for a, t in zip(o_audio_files, t_audio_files):
    if flag==False:
        if a.split('-')[0]==str(start):
            flag=True
        else:
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

    attack=OPT_attack_lf(MODE,os.path.join(save_dir,fname))

    adv = attack(real_data[None,...][None,...],target_index,target_data[None,...][None,...])

    al=len(os.listdir(save_adv_dir))
    aname="adv-{}.wav".format(al+1)
    sf.write(os.path.join(abs_path,save_adv_dir,aname),adv.squeeze(),16000)
    del (attack)
