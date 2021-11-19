# -- coding: utf-8 --
# -- coding: utf-8 --
import soundfile as sf
import numpy as np
from pesq import pesq
import os
def SNR2(x1, x2):
    from numpy.linalg import norm
    return 20 * np.log10(norm(x1) / norm(x2))

def get_all_adv(dir):
    flist=os.listdir(dir)
    advlist=[]
    for item in flist:
        if item.startswith("adv") and not item.startswith("adv-pre"):
            advlist.append(os.path.join(dir,item))
    return advlist

source_timit_dir=r"F:\SR-ATK\sampleaudio\timit-attack-audio"
target_timit_dir=r"F:\SR-ATK\sampleaudio\timit-target-audio"
source_lib_dir=r"F:\SR-ATK\sampleaudio\lib-attack-audio"
target_lib_dir=r"F:\SR-ATK\sampleaudio\lib-target-audio"

#下面写绝对路径
HSJA_dir_TIMIT=r"F:\SR-ATK\hsjaresult\timitaudio"
HSJA_dir_Lib=r"F:\SR-ATK\hsjaresult\libaudio"

SIGN_dir_TIMIT=r"F:\SR-ATK\signresult\timitaudio"
SIGN_dir_Lib=r"F:\SR-ATK\signresult\libaudio"

MY_dir_TIMIT=r"F:\SR-ATK\myresult2\timitaudio"
MY_dir_Lib=r"F:\SR-ATK\myresult2\libaudio"


def get_audio(id, MODE):
    if MODE=="TIMIT":
        o_dir=source_timit_dir
    else:
        o_dir=source_lib_dir
    o_all_audios =os.listdir(o_dir)
    oa=None
    for item in o_all_audios:
        if item.startswith("{}-".format(id)):
            oa = item
            break
    x, sr = sf.read(os.path.join(o_dir,oa))
    x /= np.linalg.norm(x, ord=np.inf)
    return x * 0.95

HSJA_TIMIT_ALL_ADV=get_all_adv(HSJA_dir_TIMIT)
HSJA_LIB_ALL_ADV=get_all_adv(HSJA_dir_Lib)
SIGN_TIMIT_ALL_ADV=get_all_adv(SIGN_dir_TIMIT)
SIGN_LIB_ALL_ADV=get_all_adv(SIGN_dir_Lib)
MY_TIMIT_ALL_ADV=get_all_adv(MY_dir_TIMIT)
MY_LIB_ALL_ADV=get_all_adv(MY_dir_Lib)
#
threshold=999
def mean_SNR_PESQ(adv_lists,MODE):
    SNR=0
    num=len(adv_lists)
    score=0
    for adv_path in adv_lists:
        ox=get_audio(adv_path.split('\\')[-1].split('-')[-1].split(".")[0],MODE)
        advx,sr=sf.read(adv_path)
        if np.linalg.norm((ox - advx))<threshold:
            SNR+=SNR2(ox, (ox - advx))
            score += pesq(sr, ox, advx, 'wb')
        else:
            num-=1
    return SNR / num,score/num
    # return SNR/num,score/num

H_T_SNR,H_T_PESQ=mean_SNR_PESQ(HSJA_TIMIT_ALL_ADV,"TIMIT")
H_L_SNR,H_L_PESQ=mean_SNR_PESQ(HSJA_LIB_ALL_ADV,"LIB")
S_T_SNR,S_T_PESQ=mean_SNR_PESQ(SIGN_TIMIT_ALL_ADV,"TIMIT")
S_L_SNR,S_L_PESQ=mean_SNR_PESQ(SIGN_LIB_ALL_ADV,"LIB")
M_T_SNR,M_T_PESQ=mean_SNR_PESQ(MY_TIMIT_ALL_ADV,"TIMIT")
M_L_SNR,M_L_PESQ=mean_SNR_PESQ(MY_LIB_ALL_ADV,"LIB")

print("HSJA_TIMIT_SNR:",H_T_SNR)
print("SIGN_TIMIT_SNR:",S_T_SNR)
print("MY_TIMIT_SNR:",M_T_SNR)

print("HSJA_LIB_SNR:",H_L_SNR)
print("SIGN_LIB_SNR:",S_L_SNR)
print("MY_LIB_SNR:",M_L_SNR)

print("HSJA_TIMIT_PESQ:",H_T_PESQ)
print("SIGN_TIMIT_PESQ:",S_T_PESQ)
print("MY_TIMIT_PESQ:",M_T_PESQ)

print("HSJA_LIB_PESQ:",H_L_PESQ)
print("SIGN_LIB_PESQ:",S_L_PESQ)
print("MY_LIB_PESQ:",M_L_PESQ)
