# -- coding: utf-8 --
import soundfile as sf
import glob
import os
import numpy as np
import random
s1=r"D:\python\MY_file\SOUNDS\SR-ATK\sampleaudio/one-16000/*"
s2=r"D:\python\MY_file\SOUNDS\SR-ATK\sampleaudio/two-32000/*"
s3=r"D:\python\MY_file\SOUNDS\SR-ATK\sampleaudio/four-64000/*"

def get_file_num(f):
    f_number=glob.glob(f)
    f_len=len(f_number)
    if f_len>=20:
        return -1
    return f_len
def save_f(x,f_num,save_f,name):
    sf_dir=save_f[:-1]
    if (f_num+1)%2==1:#说明是攻击
        f_name = str((f_num+1) // 2+1) + "-attack-" + name + ".wav"
    else:
        f_name = str((f_num+1) // 2) + "-target-" + name + ".wav"
    sf.write(os.path.join(sf_dir,f_name),x,16000)
def get_near_dir(f):
    x,sr=sf.read(f)
    if len(x)<32000 and len(x)>=16000:
        return x[:16000],s1
    elif len(x)>=32000 and len(x)<64000:
        return x[:32000],s2
    elif len(x)>=64000:
        return x[:64000],s3
    else:
        print("X:len:",len(x))
        return None,None

def get_random_wav(f):
    file_list=[]
    for a,b,c in os.walk(f):
        for item in c:
            if os.path.splitext(item)[-1]=='.WAV':
                file_list.append([os.path.join(a,item),os.path.split(a)[-1]])
    return file_list

success=0
file_list=get_random_wav(r'D:\python\MY_file\SOUNDS\SR-ATK\TRAIN')
while success<60:
    [wav_file,target_name]=random.choice(file_list)
    x,f_dir=get_near_dir(wav_file)
    if f_dir==None:
        continue
    num=get_file_num(f_dir)
    if num==-1:
        continue
    success+=1
    save_f(x,num,f_dir,target_name)





