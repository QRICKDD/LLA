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

def select_region(model,wav_attack,wav_target,t_label,length=2000,all_len=4000):
    count=16000/length
    old_propor=0
    wav_attack=torch.from_numpy(wav_attack)
    wav_target=torch.from_numpy(wav_target)
    all_clip=list(range(0,16000,length))
    nb=all_len // length
    socres=[]#[indx, (is target label), perturb_scale,pertub_wav]
    for item in all_clip:
        wa=wav_attack[item:item+length]
        wt=wav_target[item:item+length]
        padta=torch.cat([wa]*nb,dim=0)
        pretub=torch.cat([wt]*nb,dim=0)
        pre_label=predict_one_label(model,pretub)
        print("循环预测[{}:{}]:{}".format(item,item+length,pre_label))
        if pre_label!=t_label:
            old_propor+=1
            socres.append([item,False,torch.norm(wt-wa,dim=0),padta[:length]])
        else:
            #out_wav,ps=binary_search_batch(model,padta,pretub,t_label)
            socres.append([item, True, torch.norm(wt-wa,dim=0), pretub[:length]])
    #最终的target样本
    new_target=wav_target.clone()
    #选择的攻击区域
    unused_select_regions=[]
    socres.sort(key=lambda x:x[2])

    for idx,item in enumerate(socres):
        pre_label=predict_one_label(model,new_target)
        if item[1]==False and pre_label==t_label:
            temptarget=new_target.clone()
            temptarget[item[0]:item[0] + length] = wav_attack[item[0]:item[0] + length]
            if predict_one_label(model,temptarget)!=t_label:
                continue
            else:
                unused_select_regions.append([item[0],item[0]+length])
                new_target[item[0]:item[0]+length]=wav_attack[item[0]:item[0]+length]

    mask=torch.ones(16000)
    for item in unused_select_regions:
        mask[item[0]:item[1]]=0
    proportion=1 - (len(unused_select_regions) * length / 16000)
    print("非正确分类区域占比:",old_propor/count)
    print("替代占比区域:",proportion)
    new_ps=torch.norm(new_target-wav_attack,dim=0)
    print("新扰动量:",new_ps)
    old_ps=torch.norm(wav_attack-wav_target,dim=0)
    print("旧扰动量:", old_ps)
    return  new_target,mask,old_propor/count,proportion,old_ps,new_ps
