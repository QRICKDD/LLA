# -*- coding: UTF-8 -*-
from utils import Sincnet
import torch
def predict_one_label(model,audio):
    qq, _ = Sincnet.sentence_test(model, audio.float().cuda())
    return qq

def project(original_image, perturbed_images, alphas):
    return (1 - alphas) * original_image + alphas * perturbed_images

#原始音频，目标音频
def binary_search_batch(model,original_image, perturbed_image, label_or_target):
    query=0
    high = 1
    thresholds = 0.001
    low = 0
    while (high - low) / thresholds > 1:
        mid = (high + low) / 2.0
        mid_images = project(original_image, perturbed_image, mid)
        query+=1
        pre_label = predict_one_label(model,mid_images)
        if pre_label==label_or_target:
            low=mid
        else:
            high=mid
    out_image = project(original_image, perturbed_image, high)

    return out_image,torch.norm(out_image-original_image,dim=0).item()

def select_region(model,wav_attack,wav_target,a_label,t_label,length=1000,all_len=4000):
    count=16000/length
    t_true=0
    a_true=0
    wav_attack=torch.from_numpy(wav_attack)
    wav_target=torch.from_numpy(wav_target)
    all_clip=list(range(0,16000,length))
    nb=all_len // length
    socres=[]#[indx]
    for item in all_clip:
        wa=wav_attack[item:item+length]
        wt=wav_target[item:item+length]
        padta=torch.cat([wa]*nb,dim=0)
        pretub=torch.cat([wt]*nb,dim=0)
        pre_t_label=predict_one_label(model,pretub)
        pre_a_label = predict_one_label(model, padta)
        #print("循环预测[{}:{}]:{}".format(item,item+length,pr))
        if pre_t_label!=t_label and pre_a_label!=a_label:
            socres.append(item)
        if pre_t_label==t_label:
            t_true+=1
        if pre_a_label==a_label:
            a_true+=1
    new_target=wav_target.clone()
    mask=torch.ones(16000)
    for item in socres:
        mask[item:int(item+length)]=0
        new_target[item:item+length]=wav_attack[item:item+length]

    print("a正确分类区域占比:",a_true*length/16000)
    print("t正确分类区域占比:", t_true * length / 16000)
    print("同时非a非t替代占比区域:",len(socres)*length/16000)
    with open("F:\SR-ATK\LocalSearch\local.txt","a") as f:
        f.write(str(a_true*length/16000)+" "+str(t_true * length / 16000)+" "+str(len(socres)*length/16000)+"\n")
    return  new_target,mask
