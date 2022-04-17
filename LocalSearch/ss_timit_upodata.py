# -*- coding: UTF-8 -*-
import copy

from utils import Sincnet
import torch
def predict_one_label(model,audio):
    qq, _ = Sincnet.sentence_test(model, audio.float().cuda())
    return qq

def project(original_image, perturbed_images, alphas):
    return (1 - alphas) * original_image + alphas * perturbed_images

def kuozeng(x:torch.Tensor):
    while x.shape[0]<4000:
        x=torch.cat([x,x],dim=0)
    return x

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

    return torch.norm(out_image-original_image,dim=0).item()

def select_region(model,wav_attack,wav_target,a_label,t_label,length=1000,all_len=4000):
    count=16000/length
    t_true=0
    a_true=0
    wav_attack=torch.from_numpy(wav_attack)
    wav_target=torch.from_numpy(wav_target)
    all_clip=list(range(0,16000,length))
    nb=all_len // length
    socres=[]#[indx]
    a_t_s=[]#分类正确的attack区间
    t_t_s=[]#分类正确的target 区间
    # 选择区间
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
            t_t_s.append(item)
        if pre_a_label==a_label:
            a_true+=1
            a_t_s.append(item)

    #选择是否替代:0不替代， 1 替代
    is_subtitue=[]
    #测试区间  包括拼接真实目标样本  扩增替代区间  测试扰动大小 选择替代与否
    #评价attack  target 区间
    pt_attack = torch.cat([wav_target[idx:idx + length] for idx in t_t_s], dim=0)
    pt_attack=kuozeng(pt_attack)
    kuozengbei=pt_attack.shape[0]//(length*len(t_t_s))
    #遍历非正确区间
    for idx in socres:
        a_mis_pad=torch.cat([wav_attack[idx:idx+length]]*(len(t_t_s)*kuozengbei),dim=0)
        dis_a_t=binary_search_batch(model,a_mis_pad,pt_attack,t_label)
        t_mis_pad = torch.cat([wav_target[idx:idx + length]] *(len(t_t_s)*kuozengbei), dim=0)
        dis_t_t = binary_search_batch(model, t_mis_pad, pt_attack, t_label)
        if dis_a_t<dis_t_t:
            is_subtitue.append(1)
        else:
            is_subtitue.append(0)
    new_target=wav_target.clone()
    mask=torch.ones(16000)
    for item,is_s in zip(socres,is_subtitue):
        if is_s==1:
            mask[item:int(item+length)]=0
            new_target[item:item+length]=wav_attack[item:item+length]

    print("a正确分类区域占比:",a_true*length/16000)
    print("t正确分类区域占比:", t_true * length / 16000)
    print("同时非a非t替代占比区域:",len(socres)*length/16000)
    print("替代区间:,",sum(is_subtitue)*length/16000)
    with open("F:\SR-ATK\LocalSearch\local.txt","a") as f:
        f.write(str(a_true*length/16000)+" "+str(t_true * length / 16000)+" "+str(len(socres)*length/16000)+"\n")
    return  new_target,mask
