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
import PROPOSE_ATTACK
import LOCAL_ATT_HSJA_ATTACK

# 初始化模型 模型在gpu上

MODE = "Librispeech"
abs_path=r"F:\SR-ATK"
is_test=True

model = Sincnet.get_speaker_model(MODE)
speaker_label, label_speaker = Sincnet.get_speaker_label(MODE)

if MODE=="Librispeech":
    save_dir = "2022218/QEBA/lib"
    save_adv_dir="2022218/QEBA/libaudio"
    attackdir = r"用于画图sampleaudio\lib-attack-audio"
    targetdir = r"用于画图sampleaudio\lib-target-audio"
else:
    save_dir = r"2022218/QEBA/timit"
    save_adv_dir = r"2022218/QEBA/timitaudio"
    attackdir = r"用于画图sampleaudio\timit-attack-audio"
    targetdir = r"用于画图sampleaudio\timit-target-audio"
o_audio_files = os.listdir(attackdir)
o_audio_files.sort(key=lambda item: int(item.split('-')[0]))
t_audio_files = os.listdir(targetdir)
t_audio_files.sort(key=lambda item: int(item.split('-')[0]))

if is_test:
    save_dir="show/temp"
    save_adv_dir = "show/tempadv"


#start=list(range(1,50))
start=[35]
flag=False

for a, t in zip(o_audio_files, t_audio_files):
    if int(a.split('-')[0]) not in start:
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
    # laatk=LOCAL_ATT_HSJA_ATTACK.LAATTACK(os.path.join(attackdir,a),os.path.join(targetdir,t),
    #                                save_p_fname=save_pfn,save_info_fname=save_info_n,MODE=MODE,dct_field=0.65)
    # o2_audio, preturb,query_num,interval=laatk.targeted_attack()
    # # save audio
    # aprename = "adv-pre-{}.wav".format(dl)
    # sf.write(os.path.join(abs_path, save_adv_dir, aprename), o2_audio+preturb, 16000)

    # pre deal audio
    real_data = real_data / np.linalg.norm(real_data, ord=np.inf)
    real_data *= 0.95
    target_data = target_data / np.linalg.norm(target_data, ord=np.inf)
    target_data *= 0.95

    fname="test-{}".format(dl)

    attack=PROPOSE_ATTACK.HSJA(MODE=MODE,SAVE_DIR_PATH=save_pfn,query_num=0,interval=[0,16000])
    #adv = attack(real_data[None,...][None,...],target_index,(preturb+real_data)[None,...][None,...],dct_field=0.65)
    adv = attack(real_data[None, ...][None, ...], target_index, target_data[None, ...][None, ...],dct_field=0.65)

    aname="adv-{}.wav".format(dl)
    sf.write(os.path.join(abs_path,save_adv_dir,aname),adv,16000)
    del (attack)

    if is_test==True:
        break
