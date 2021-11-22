# -- coding: utf-8 --
from utils import Sincnet
import torch
import numpy as np
import soundfile as sf
import librosa.display
import matplotlib.pyplot as plt
import pickle
import utils
import os
import LOCAL_ATT_HSJA_ATTACK

MODE = "TIMIT"
abs_path=os.getcwd()

model = Sincnet.get_speaker_model(MODE)
speaker_label, label_speaker = Sincnet.get_speaker_label(MODE)
def mkd(name):
    if os.path.exist(name)==False:
        os.mkdir(os.path.join(abs_path,name))
if MODE=="Librispeech":
    save_dir = "lahresult\lib"
    save_adv_dir="lahresult\libaudio"
    mkd(save_dir)
    mkd(save_adv_dir)
    attackdir = r"AttackDataset\lib-attack-audio"
    targetdir = r"AttackDataset\lib-target-audio"
else:
    save_dir = r"lahresult\timit"
    save_adv_dir = r"lahresult\timitaudio"
    mkd(save_dir)
    mkd(save_adv_dir)
    attackdir = r"AttackDataset\timit-attack-audio"
    targetdir = r"AttackDataset\timit-target-audio"
o_audio_files = os.listdir(attackdir)
o_audio_files.sort(key=lambda item: int(item.split('-')[0]))
t_audio_files = os.listdir(targetdir)
t_audio_files.sort(key=lambda item: int(item.split('-')[0]))


start=10
flag=False

for a, t in zip(o_audio_files, t_audio_files):
    if flag==False:
        if a.split('-')[0]==str(start):
            flag=True
        else:
            continue
    if MODE=="Librispeech":
        real_name=a.split('-')[2]
        target_name = t.split('-')[2]
    elif MODE=="TIMIT":
        real_name = a.split('-')[-1].split('.')[0].lower()
        target_name = t.split('-')[-1].split('.')[0].lower()
    # read audio
    real_data, fs = sf.read(os.path.join(attackdir, a))
    target_data, fs = sf.read(os.path.join(targetdir, t))
    # test is ok
    pid, _ = Sincnet.sentence_test(model,torch.from_numpy(target_data).float().cuda())
    target_index = speaker_label[target_name]
    if pid != target_index:
        print("error")
        continue
    # set file name
    print("now attack audio:", a)
    dl = a.split('-')[0]

    # local and attenutation attack
    save_pfn=os.path.join(abs_path,save_dir,"test-{}.txt".format(dl))
    save_info_n=os.path.join(abs_path,save_dir,"info-{}.txt".format(dl))
    laatk=LOCAL_ATT_HSJA_ATTACK.LAATTACK(os.path.join(attackdir, a), os.path.join(targetdir, t),
                                           save_p_fname=save_pfn, save_info_fname=save_info_n, MODE=MODE, dct_field=0.65)
    o2_audio, preturb,query_num,interval=laatk.targeted_attack()
    # save audio
    aprename = "adv-pre-{}.wav".format(dl)
    sf.write(os.path.join(abs_path, save_adv_dir, aprename), o2_audio+preturb, 16000)

    # pre deal audio
    real_data = real_data / np.linalg.norm(real_data, ord=np.inf)
    real_data *= 0.95
    target_data = target_data / np.linalg.norm(target_data, ord=np.inf)
    target_data *= 0.95

    fname="test-{}".format(dl)

    attack=LOCAL_ATT_HSJA_ATTACKpy.HSJA(MODE=MODE, SAVE_DIR_PATH=save_pfn, query_num=query_num, interval=interval)
    adv = attack(real_data[None,...][None,...],target_index,(preturb+real_data)[None,...][None,...])

    aname="adv-{}.wav".format(dl)
    sf.write(os.path.join(abs_path,save_adv_dir,aname),adv,16000)
    del (laatk)
    del (attack)


