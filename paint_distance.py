# -- coding: utf-8 --
import os
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import pandas as pd

def print_sr(q_list,v_list,num):
    q=np.array(q_list)
    index=np.argmin(np.abs(q-num))
    print("SR:",v_list[index])
    print("q:",q_list[index])


def getmaxnum_list(all_num_list):
    i=0
    min_num=all_num_list[0][-1]
    for index,num_list in enumerate(all_num_list):
        if num_list[-1]>min_num:
            min_num=num_list[-1]
            i=index
    return all_num_list[i]

def getminnum_list(all_num_list):
    i=0
    min_num=all_num_list[0][-1]
    for index,num_list in enumerate(all_num_list):
        if num_list[-1]<min_num:
            min_num=num_list[-1]
            i=index
    return all_num_list[i]

def get_success_rate(all_data,thresold):
    q_list=[1]
    sr_list=[0]
    num=0
    all_num=len(all_data)-1
    queries=list(range(0,25000,10))
    for q in queries:
        tnum=0
        for sidata in all_data:
            if sidata[q+1]<thresold:
                tnum+=1
        if tnum>num:
            num=tnum
            q_list.append(q+1)
            sr_list.append(num/all_num)
    fin_rate=0
    for item in all_data:
        if item[25000-1]<thresold:
            fin_rate+=1
    q_list=q_list+[25000]
    sr_list=sr_list+[fin_rate/all_num]

    return q_list,sr_list

def getmean_list(all_num_list):
    return np.mean(all_num_list, axis=0).tolist()

def buquan(dir_name,start_value):
    data = pd.read_csv(dir_name, header=None, sep='\t')
    list_num, nparray_list=data[0].tolist(),data[1].tolist()
    res=[]
    if abs(nparray_list[0]-start_value)>0.1 and list_num[0]!=1:
        list_num=[0]+list_num
        nparray_list=[start_value]+nparray_list
    if list_num[-1]<25000:
        list_num=list_num+[25000]
        nparray_list=nparray_list+[nparray_list[-1]]
    for index,item in enumerate(list_num[:-1]):
        # print("next:",list_num[index+1])
        # print("now:", item)
        if list_num[index+1]-item<0:
            print("list_num:",list_num)
            print("dir_name:",dir_name)
            print(nparray_list)
            print("index:",index)
            print("len(list_num):", len(list_num))
            print("list_num[index]:", list_num[index])
            print("list_num[index+1]:",list_num[index+1])
            print(item)
        tl=np.linspace(nparray_list[index],nparray_list[index+1],num=list_num[index+1]-item,endpoint=False).tolist()
        res.extend(tl)
    num_list = list(range(list_num[0], list_num[-1]))
    return num_list,res

def get_all_test(path):
    res=[]
    fname=os.listdir(path)
    for item in fname:
        if item.startswith("test"):
            res.append(os.path.join(path,item))
    return res

ABS_ROOT="F:\SR-ATK"
source_timit_dir=r"sampleaudio\timit-attack-audio"
target_timit_dir=r"sampleaudio\timit-target-audio"
source_lib_dir=r"sampleaudio\lib-attack-audio"
target_lib_dir=r"sampleaudio\lib-target-audio"

#下面写绝对路径
HSJA_dir_TIMIT=r"F:\SR-ATK\hsjaresult\timit"
HSJA_dir_Lib=r"F:\SR-ATK\hsjaresult\lib"

SIGN_dir_TIMIT=r"F:\SR-ATK\signresult\timit"
SIGN_dir_Lib=r"F:\SR-ATK\signresult\lib"

MY_dir_TIMIT=r"F:\SR-ATK\myresult2\timit"
MY_dir_Lib=r"F:\SR-ATK\myresult2\lib"



o_all_timit_audios=os.listdir(os.path.join(ABS_ROOT,source_timit_dir))
#排序代码填充
o_all_timit_audios.sort(key=lambda item: int(item.split('-')[0]))

t_all_timit_audios=os.listdir(os.path.join(ABS_ROOT,target_timit_dir))
#排序代码填充
t_all_timit_audios.sort(key=lambda item: int(item.split('-')[0]))

o_all_lib_audios=os.listdir(os.path.join(ABS_ROOT,source_lib_dir))
#排序代码填充
o_all_lib_audios.sort(key=lambda item: int(item.split('-')[0]))

t_all_lib_audios=os.listdir(os.path.join(ABS_ROOT,target_lib_dir))
#排序代码填充
t_all_lib_audios.sort(key=lambda item: int(item.split('-')[0]))


#根据id MODE获取对应的init
def get_init_distance(id,MODE):
    if MODE=="timit":
        o_all_audios=o_all_timit_audios
        t_all_audios = t_all_timit_audios
        temp_abs_source_dir=os.path.join(ABS_ROOT,source_timit_dir)
        temp_abs_target_dir = os.path.join(ABS_ROOT, target_timit_dir)
    else:
        o_all_audios = o_all_lib_audios
        t_all_audios = t_all_lib_audios
        temp_abs_source_dir = os.path.join(ABS_ROOT, source_lib_dir)
        temp_abs_target_dir = os.path.join(ABS_ROOT, target_lib_dir)
    for item in o_all_audios:
        if item.startswith("{}-".format(id)):
            oa=os.path.join(temp_abs_source_dir,item)
            break
    for item in t_all_audios:
        if item.startswith("{}-".format(id)):
            ta=os.path.join(temp_abs_target_dir,item)
            break
    o,sr=sf.read(oa)
    t, sr = sf.read(ta)
    o/=np.linalg.norm(o,np.inf)
    t /= np.linalg.norm(t, np.inf)
    o*=0.95
    t*=0.95
    return np.linalg.norm(o-t)



#返回timit_value列表  以及 lib_value列表
def get_all_buquan(TIMIT_path,Lib_path):
    timit_all_test=get_all_test(TIMIT_path)
    lib_all_test=get_all_test(Lib_path)
    ta_n_list=[]
    ta_value=[]
    la_n_list = []
    la_value = []
    #便利TIMIT test
    for tim_test_f in timit_all_test:
        init_p=get_init_distance(tim_test_f.split('\\')[-1].split('-')[-1].split(".")[0],MODE="timit")
        tim_num_list,tim_value=buquan(tim_test_f,init_p)
        ta_n_list.append(tim_num_list[:25000])
        ta_value.append(tim_value[:25000])
    #遍历 Lib test
    for lib_test_f in lib_all_test:
        init_p=get_init_distance(lib_test_f.split('\\')[-1].split('-')[-1].split(".")[0],MODE="lib")
        lib_num_list,lib_value=buquan(lib_test_f,init_p)
        la_n_list.append(lib_num_list[:25000])
        la_value.append(lib_value[:25000])
    return ta_value,la_value



#获取全部数据
HSJA_TIMIT_VALUES,HSJA_LIB_VALUES=get_all_buquan(HSJA_dir_TIMIT,HSJA_dir_Lib)
SIGN_TIMIT_VALUES,SIGN_LIB_VALUES=get_all_buquan(SIGN_dir_TIMIT,SIGN_dir_Lib)
MY_TIMIT_VALUES,MY_LIB_VALUES=get_all_buquan(MY_dir_TIMIT,MY_dir_Lib)

#获取最大最小线条
HSJA_TIMIT_MIN=getminnum_list(HSJA_TIMIT_VALUES)
HSJA_TIMIT_MAX=getmaxnum_list(HSJA_TIMIT_VALUES)
SIGN_TIMIT_MIN=getminnum_list(SIGN_TIMIT_VALUES)
SIGN_TIMIT_MAX=getmaxnum_list(SIGN_TIMIT_VALUES)
MY_TIMIT_MIN=getminnum_list(MY_TIMIT_VALUES)
MY_TIMIT_MAX=getmaxnum_list(MY_TIMIT_VALUES)
#Lib MIN MAX
HSJA_LIB_MIN=getminnum_list(HSJA_LIB_VALUES)
HSJA_LIB_MAX=getmaxnum_list(HSJA_LIB_VALUES)
SIGN_LIB_MIN=getminnum_list(SIGN_LIB_VALUES)
SIGN_LIB_MAX=getmaxnum_list(SIGN_LIB_VALUES)
MY_LIB_MIN=getminnum_list(MY_LIB_VALUES)
MY_LIB_MAX=getminnum_list(MY_LIB_VALUES)
# MEAN TIMIT
HSJA_TIMIT_MEAN=getmean_list(HSJA_TIMIT_VALUES)
SIGN_TIMIT_MEAN=getmean_list(SIGN_TIMIT_VALUES)
MY_TIMIT_MEAN=getmean_list(MY_TIMIT_VALUES)
# MEAN LIB
HSJA_LIB_MEAN=getmean_list(HSJA_LIB_VALUES)
SIGN_LIB_MEAN=getmean_list(SIGN_LIB_VALUES)
MY_LIB_MEAN=getmean_list(MY_LIB_VALUES)

print("HSJA_TIMIT_MEAN[0],[5k],[15k],[25k]:",HSJA_TIMIT_MEAN[0],"\t",HSJA_TIMIT_MEAN[5000],"\t",HSJA_TIMIT_MEAN[15000],"\t",HSJA_TIMIT_MEAN[25000-1])
print("HSJA_LIB_MEAN[0],[5k],[15k],[25k]:",HSJA_LIB_MEAN[0],"\t",HSJA_LIB_MEAN[5000],"\t",HSJA_LIB_MEAN[15000],"\t",HSJA_LIB_MEAN[25000-1])
print("SIGN_TIMIT_MEAN[0],[5k],[15k],[25k]:",SIGN_TIMIT_MEAN[0],"\t",SIGN_TIMIT_MEAN[5000],"\t",SIGN_TIMIT_MEAN[15000],"\t",SIGN_TIMIT_MEAN[25000-1])
print("SIGN_LIB_MEAN[0],[5k],[15k],[25k]:",SIGN_LIB_MEAN[0],"\t",SIGN_LIB_MEAN[5000],"\t",SIGN_LIB_MEAN[15000],"\t",SIGN_LIB_MEAN[25000-1])
print("MY_TIMIT_MEAN[0],[5k],[15k],[25k]:",MY_TIMIT_MEAN[0],"\t",MY_TIMIT_MEAN[5000],"\t",MY_TIMIT_MEAN[15000],"\t",MY_TIMIT_MEAN[25000-1])
print("MY_LIB_MEAN[0],[5k],[15k],[25k]:",MY_LIB_MEAN[0],"\t",MY_LIB_MEAN[5000],"\t",MY_LIB_MEAN[15000],"\t",MY_LIB_MEAN[25000-1])
# Success rate
THRESOLD=3
HQ_T_2,HSJA_TIMIT_SR_2=get_success_rate(HSJA_TIMIT_VALUES,thresold=THRESOLD)
HQ_T_1,HSJA_TIMIT_SR_1=get_success_rate(HSJA_TIMIT_VALUES,thresold=THRESOLD-1)
SQ_T_2,SIGN_TIMIT_SR_2=get_success_rate(SIGN_TIMIT_VALUES,thresold=THRESOLD)
SQ_T_1,SIGN_TIMIT_SR_1=get_success_rate(SIGN_TIMIT_VALUES,thresold=THRESOLD-1)
MQ_T_2,MY_TIMIT_SR_2=get_success_rate(MY_TIMIT_VALUES,thresold=THRESOLD)
MQ_T_1,MY_TIMIT_SR_1=get_success_rate(MY_TIMIT_VALUES,thresold=THRESOLD-1)
# LIB
THRESOLD=3
HQ_L_2,HSJA_LIB_SR_2=get_success_rate(HSJA_LIB_VALUES,thresold=THRESOLD)
HQ_L_1,HSJA_LIB_SR_1=get_success_rate(HSJA_LIB_VALUES,thresold=THRESOLD-1)
SQ_L_2,SIGN_LIB_SR_2=get_success_rate(SIGN_LIB_VALUES,thresold=THRESOLD)
SQ_L_1,SIGN_LIB_SR_1=get_success_rate(SIGN_LIB_VALUES,thresold=THRESOLD-1)
MQ_L_2,MY_LIB_SR_2=get_success_rate(MY_LIB_VALUES,thresold=THRESOLD)
MQ_L_1,MY_LIB_SR_1=get_success_rate(MY_LIB_VALUES,thresold=THRESOLD-1)



print("HSJA-LIB")
print("Q:")
for q in HQ_L_1:
    print(q,"\t",end="")
print("")
for v in HSJA_LIB_SR_1:
    print(v,"\t",end="")
print("")
print("SIGN-LIB")
print("Q:")
for q in SQ_L_1:
    print(q,"\t",end="")
print("")
for v in SIGN_LIB_SR_1:
    print(v,"\t",end="")
print("")
print("MY-LIB")
print("Q:")
for q in MQ_L_1:
    print(q,"\t",end="")
print("")
for v in MY_LIB_SR_1:
    print(v,"\t",end="")
print("")
print("HSJA-TIMIT")
print("Q:")
for q in HQ_T_1:
    print(q,"\t",end="")
print("")
for v in HSJA_TIMIT_SR_1:
    print(v,"\t",end="")
print("")
print("SIGN-TIMIT")
print("Q:")
for q in SQ_T_1:
    print(q,"\t",end="")
print("")
for v in SIGN_TIMIT_SR_1:
    print(v,"\t",end="")
print("")
print("MY-TIMIT")
print("Q:")
for q in MQ_T_1:
    print(q,"\t",end="")
print("")
for v in MY_TIMIT_SR_1:
    print(v,"\t",end="")

# 1 [TIMIT distance] 2 [Lib distance] 3 [TIMIT ] 4
IS_MID=3
fontsize=18
psdi=r"F:\SR-ATK\picture"
if IS_MID==1:
    #绘制图1
    # # #两条曲线之间的区域
    # #plt.fill_between(x,func(x),fund(x),color='blue',alpha=0.25)
    fig=plt.figure(figsize=(7,6))
    ax=fig.add_subplot(111)
    ax.set_yticks([0,3,6,9,12,15])
    ax.set_xticks([0,5000,10000,15000,20000,25000])
    ax.set_yticklabels([0, 3,6,9,12,15], fontsize=fontsize)
    ax.set_xticklabels([0,"5k","10k","15k","20k","25k"],fontsize=fontsize)
    plt.ylim(0,15)
    plt.xlim(0,25000)
    # plt.fill_between(list(range(1,25001)),HSJA_TIMIT_MIN,HSJA_TIMIT_MAX,color='pink',alpha=0.5)
    # plt.fill_between(list(range(1,25001)),SIGN_TIMIT_MIN,SIGN_TIMIT_MAX,color='green',alpha=0.2)
    # plt.fill_between(list(range(1, 25001)), MY_TIMIT_MIN, MY_TIMIT_MAX, color='orange',alpha=0.5)

    p1,=plt.plot(HSJA_TIMIT_MEAN,c='black',linewidth=2.0,linestyle="-.")
    p2,=plt.plot(SIGN_TIMIT_MEAN,c='black',linewidth=2.0)
    p3,=plt.plot(MY_TIMIT_MEAN,c='red',linewidth=2.0)
    plt.legend([p1, p2,p3], ["HSJA","SIGN-OPT","Proposed"], loc='upper right',fontsize=fontsize)


    # plt.hlines(y=n1m[-1],xmin=need_queries,xmax=25000,linestyles='--',linewidth=2.0)
    # plt.hlines(y=n1m[-1],xmin=0,xmax=25000,linestyles='--',linewidth=2.0)
    # plt.vlines(x=need_queries,ymin=0,ymax=n1m[-1],linestyles="--",linewidth=2.0)
    # plt.vlines(x=need_queries,ymin=0,ymax=15,linestyles="--",linewidth=2.0)

    plt.xlabel("Queries",fontsize=fontsize)
    plt.ylabel("$L_2$ Distance",fontsize=fontsize)
    plt.title(r"TIMIT",fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(os.path.join(psdi,'TIMIT_distance_mean.pdf'), bbox_inches='tight', dpi=800)
    plt.show()
elif IS_MID==2:
    #绘制图2
    fig = plt.figure(figsize=(7,6))
    ax2=fig.add_subplot(111)
    ax2.set_yticks([0,3,6,9,12,15])
    ax2.set_xticks([0,5000,10000,15000,20000,25000])
    ax2.set_yticklabels([0, 3, 6, 9, 12, 15], fontsize=fontsize)
    ax2.set_xticklabels([0, "5k", "10k", "15k", "20k", "25k"], fontsize=fontsize)
    plt.ylim(0,15)
    plt.xlim(0,25000)
    # plt.fill_between(list(range(1,25001)),HSJA_LIB_MIN,HSJA_LIB_MAX,color='pink',alpha=0.5)
    # plt.fill_between(list(range(1,25001)),SIGN_LIB_MIN,SIGN_LIB_MAX,color='green',alpha=0.2)
    # plt.fill_between(list(range(1, 25001)), MY_LIB_MIN, MY_LIB_MAX, color='orange', alpha=0.5)

    p3,=plt.plot(HSJA_LIB_MEAN,c='black',linewidth=2.0,linestyle="-.")
    p4,=plt.plot(SIGN_LIB_MEAN,c='black',linewidth=2.0)
    p5,=plt.plot(MY_LIB_MEAN,c='red',linewidth=2.0)
    plt.legend([p3, p4,p5], ["HSJA","SIGN-OPT","Proposed"], loc='upper right',fontsize=fontsize)

    # plt.hlines(y=n3m[-1],xmin=need_queries,xmax=25000,linestyles='--',linewidth=2.0)
    # plt.vlines(x=need_queries,ymin=0,ymax=n3m[-1],linestyles="--",linewidth=2.0)
    # plt.hlines(y=n3m[-1],xmin=0,xmax=25000,linestyles='--',linewidth=2.0)
    # plt.vlines(x=need_queries,ymin=0,ymax=20,linestyles="--",linewidth=2.0)
    plt.xlabel("Queries",fontsize=fontsize)
    plt.ylabel("$L_2$ Distance",fontsize=fontsize)
    plt.title(r"LibriSpeech",fontsize=fontsize)

    plt.tight_layout()
    plt.savefig(os.path.join(psdi,'LIB_distance_mean.pdf'), bbox_inches='tight', dpi=800)
    plt.show()
elif IS_MID==3:
    #绘制图3
    fig = plt.figure(figsize=(7,6))
    ax2=fig.add_subplot(111)
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax2.set_xticks([0, 5000, 10000, 15000, 20000, 25000])
    ax2.set_yticklabels([0,0.25,0.5,0.75,1],fontsize=fontsize)
    ax2.set_xticklabels([0, "5k", "10k", "15k", "20k", "25k"], fontsize=fontsize)
    plt.ylim(0,1)
    plt.xlim(0,25000)



    # p3,=plt.plot(HQ_T_1,HSJA_TIMIT_SR_1,c='black',linewidth=2.0,linestyle="-.")
    # p4,=plt.plot(SQ_T_1,SIGN_TIMIT_SR_1,c='black',linewidth=2.0)
    # p5,=plt.plot(MQ_T_1,MY_TIMIT_SR_1,c='red',linewidth=2.0)
    p3, = plt.plot(HQ_T_2, HSJA_TIMIT_SR_2, c='black', linewidth=2.0, linestyle="-.")
    p4, = plt.plot(SQ_T_2, SIGN_TIMIT_SR_2, c='black', linewidth=2.0)
    p5, = plt.plot(MQ_T_2, MY_TIMIT_SR_2, c='red', linewidth=2.0)
    plt.legend([p3, p4,p5], ["HSJA","SIGN-OPT","Proposed"], loc='upper left',fontsize=fontsize)


    plt.xlabel("Queries",fontsize=fontsize)
    #plt.ylabel("Success rate $l_2$<2",fontsize=fontsize)
    plt.ylabel("Success rate $L_2$<3", fontsize=fontsize)
    plt.title(r"TIMIT",fontsize=fontsize)


    plt.tight_layout()
    plt.savefig(os.path.join(psdi,'TIMIT_SR-2.pdf'), bbox_inches='tight', dpi=800)
    plt.show()
elif IS_MID==4:
    #绘制图4
    fig = plt.figure(figsize=(7,6))
    ax2=fig.add_subplot(111)
    ax2.set_yticks([0,0.25,0.5,0.75,1])
    ax2.set_xticks([0,5000,10000,15000,20000,25000])
    ax2.set_yticklabels([0, 0.25, 0.5, 0.75, 1], fontsize=fontsize)
    ax2.set_xticklabels([0, "5k", "10k", "15k", "20k", "25k"], fontsize=fontsize)
    plt.ylim(0,1)
    plt.xlim(0,25000)

    # p3,=plt.plot(HQ_L_1,HSJA_LIB_SR_1,c='black',linewidth=2.0,linestyle="-.")
    # p4,=plt.plot(SQ_L_1,SIGN_LIB_SR_1,c='black',linewidth=2.0)
    # p5,=plt.plot(MQ_L_1,MY_LIB_SR_1,c='r',linewidth=2.0)
    p3, = plt.plot(HQ_L_2, HSJA_LIB_SR_2, c='black', linewidth=2.0, linestyle="-.")
    p4, = plt.plot(SQ_L_2, SIGN_LIB_SR_2, c='black', linewidth=2.0)
    p5, = plt.plot(MQ_L_2, MY_LIB_SR_2, c='r', linewidth=2.0)
    plt.legend([p3, p4,p5], ["HSJA","SIGN-OPT","Proposed"], loc='upper left',fontsize=fontsize)


    plt.xlabel("Queries",fontsize=fontsize)
    #plt.ylabel("Success rate $l_2$<2",fontsize=fontsize)
    plt.ylabel("Success rate $L_2$<3", fontsize=fontsize)
    plt.title(r"LibriSpeech",fontsize=fontsize)


    plt.tight_layout()
    plt.savefig(os.path.join(psdi,'LIB_SR-2.pdf'), bbox_inches='tight', dpi=800)
    plt.show()
elif IS_MID==5:
    #绘制图5
    # 展示查询效率
    # #plt.fill_between(x,func(x),fund(x),color='blue',alpha=0.25)
    fig=plt.figure(figsize=(7,6))
    ax=fig.add_subplot(111)
    #ax.set_yticks([0,3,6,9,12,15])
    ax.set_xticks([15, 12, 9, 6, 3, 0])
    ax.set_yticks([0,5000,10000,15000,20000,25000])
    ax.set_xticklabels(["15", "12", "9", "6", "3", "0"], fontsize=fontsize)
    ax.set_yticklabels([0,"5k","10k","15k","20k","25k"],fontsize=fontsize)
    plt.xlim(0,15)
    plt.ylim(0,25000)
    # plt.fill_between(list(range(1,25001)),HSJA_TIMIT_MIN,HSJA_TIMIT_MAX,color='pink',alpha=0.5)
    # plt.fill_between(list(range(1,25001)),SIGN_TIMIT_MIN,SIGN_TIMIT_MAX,color='green',alpha=0.2)
    # plt.fill_between(list(range(1, 25001)), MY_TIMIT_MIN, MY_TIMIT_MAX, color='orange',alpha=0.5)
    temp_y_aaaa=list(range(25000,0,-1))
    def ssort(l):
        l=np.array(l)
        l=sorted(l)
        return l
    p1,=plt.plot(ssort(HSJA_TIMIT_MEAN),temp_y_aaaa,c='black',linewidth=2.0,linestyle="-.")
    p2,=plt.plot(ssort(SIGN_TIMIT_MEAN),temp_y_aaaa,c='black',linewidth=2.0)
    p3,=plt.plot(ssort(MY_TIMIT_MEAN),temp_y_aaaa,c='red',linewidth=2.0)
    plt.legend([p1, p2,p3], ["HSJA","SIGN-OPT","Proposed"], loc='upper left',fontsize=fontsize)
    def get_v_point(point,x):
        x=np.array(x)
        x=np.abs(x-point)
        return np.argmin(x)
    c="gray"
    plt.axvline(x=HSJA_TIMIT_MEAN[-1], c=c,linewidth=2.0,linestyle="-.")  # 添加水平直线
    plt.axhline(y=get_v_point(HSJA_TIMIT_MEAN[-1],MY_TIMIT_MEAN), ls="-.", c=c)
    plt.axvline(x=SIGN_TIMIT_MEAN[-1], c=c, linewidth=2.0,)  # 添加水平直线
    plt.axhline(y=get_v_point(SIGN_TIMIT_MEAN[-1], MY_TIMIT_MEAN), c=c,linewidth=2.0)

    plt.ylabel("Queries",fontsize=fontsize)
    plt.xlabel("$L_2$ Distance",fontsize=fontsize)
    plt.title(r"TIMIT",fontsize=fontsize)

    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(psdi,'TIMIT_query_efficient.pdf'), bbox_inches='tight', dpi=800)
    plt.show()
elif IS_MID==6:
    #绘制图5
    # 展示查询效率
    # #plt.fill_between(x,func(x),fund(x),color='blue',alpha=0.25)
    fig=plt.figure(figsize=(7,6))
    ax=fig.add_subplot(111)
    #ax.set_yticks([0,3,6,9,12,15])
    ax.set_xticks([15, 12, 9, 6, 3, 0])
    ax.set_yticks([0,5000,10000,15000,20000,25000])
    ax.set_xticklabels(["15", "12", "9", "6", "3", "0"], fontsize=fontsize)
    ax.set_yticklabels([0,"5k","10k","15k","20k","25k"],fontsize=fontsize)
    plt.xlim(0,15)
    plt.ylim(0,25000)
    # plt.fill_between(list(range(1,25001)),HSJA_TIMIT_MIN,HSJA_TIMIT_MAX,color='pink',alpha=0.5)
    # plt.fill_between(list(range(1,25001)),SIGN_TIMIT_MIN,SIGN_TIMIT_MAX,color='green',alpha=0.2)
    # plt.fill_between(list(range(1, 25001)), MY_TIMIT_MIN, MY_TIMIT_MAX, color='orange',alpha=0.5)
    temp_y_aaaa=list(range(25000,0,-1))
    def ssort(l):
        l=np.array(l)
        l=sorted(l)
        return l
    p1,=plt.plot(ssort(HSJA_LIB_MEAN),temp_y_aaaa,c='black',linewidth=2.0,linestyle="-.")
    p2,=plt.plot(ssort(SIGN_LIB_MEAN),temp_y_aaaa,c='black',linewidth=2.0)
    p3,=plt.plot(ssort(MY_LIB_MEAN),temp_y_aaaa,c='red',linewidth=2.0)
    plt.legend([p1, p2,p3], ["HSJA","SIGN-OPT","Proposed"], loc='upper left',fontsize=fontsize)
    def get_v_point(point,x):
        x=np.array(x)
        x=np.abs(x-point)
        return np.argmin(x)
    c="gray"
    plt.axvline(x=HSJA_LIB_MEAN[-1], c=c,linewidth=2.0,linestyle="-.")  # 添加水平直线
    plt.axhline(y=get_v_point(HSJA_LIB_MEAN[-1],MY_LIB_MEAN), ls="-.", c=c)
    plt.axvline(x=SIGN_LIB_MEAN[-1], c=c, linewidth=2.0,)  # 添加水平直线
    plt.axhline(y=get_v_point(SIGN_LIB_MEAN[-1], MY_LIB_MEAN), c=c,linewidth=2.0)

    plt.ylabel("Queries",fontsize=fontsize)
    plt.xlabel("$L_2$ Distance",fontsize=fontsize)
    plt.title(r"LibriSpeech",fontsize=fontsize)

    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(psdi,'LibriSpeech_query_efficient.pdf'), bbox_inches='tight', dpi=800)
    plt.show()







