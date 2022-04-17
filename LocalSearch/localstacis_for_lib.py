# -*- coding: UTF-8 -*-
from utils import Sincnet
#from utils.sincet_for_lib import *
import torch
import numpy as np
import soundfile as sf
import os
import time
from LocalSearch.select_region_attack_timit import select_region

MODE = "Librispeech"
#model = get_speaker_model2(MODE)
model = Sincnet.get_speaker_model(MODE)
speaker_label, label_speaker = Sincnet.get_speaker_label(MODE)

def show_max(wav):
    qq,_ = Sincnet.sentence_test_lib(model,wav.float().cuda())
    print(qq)

adir=r"F:\SR-ATK\用于画图sampleaudio\lib-attack-audio"
tdir=r"F:\SR-ATK\用于画图sampleaudio\lib-target-audio"
o_audio_files = os.listdir(adir)
o_audio_files.sort(key=lambda item: int(item.split('-')[0]))
t_audio_files = os.listdir(tdir)
t_audio_files.sort(key=lambda item: int(item.split('-')[0]))

save_txt_path=r"F:\SR-ATK\占比记录\newlib.txt"
record=[]#[非目标分类区占比 ， 攻击区域占比，原初始扰动大小，更新的初始扰动大小]
for a,t in zip(o_audio_files,t_audio_files):
    a=os.path.join(adir,a)
    t=os.path.join(tdir,t)
    real_name = os.path.basename(a).split("-")[2]
    target_name = os.path.basename(t).split("-")[2]
    real_index = speaker_label[real_name]
    print("攻击者id",real_index)
    target_index = speaker_label[target_name]
    print("目标id",target_index)

    real_data, fs = sf.read(a)
    real_data = real_data / np.linalg.norm(real_data, ord=np.inf)

    target_data, fs = sf.read(t)
    target_data = target_data / np.linalg.norm(target_data, ord=np.inf)

    new_target,mask,old_pro,new_pro,old_ps,new_ps=select_region(model,real_data,target_data,
                                                                t_label=target_index,
                                                                length=1000,all_len=4000)
    record.append([old_pro,new_pro,old_ps,new_ps])
    with open(save_txt_path, "a") as f:
        f.write(str(old_pro) + " " + str(new_pro) + " " + str(old_ps) + " " + str(new_ps) + "\n")
import numpy as np
print(np.mean(record,axis=0))




