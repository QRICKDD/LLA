import os
import shutil
import itertools
import soundfile as sf
YOU_NEED_SUM=50

MODE="Lib"
rootd="F:\SR-ATK\sampleaudio"
if MODE=="TIMIT":
    ad=os.listdir(os.path.join(rootd,"timit-attack-audio"))
    td=os.listdir(os.path.join(rootd,"timit-target-audio"))
    save_a=os.path.join(rootd,"timit-attack-audio")
    save_t = os.path.join(rootd, "timit-target-audio")
    def gname(s):
        return s.split('-')[-1].split('.')[0]
if MODE=="Lib":
    ad=os.listdir(os.path.join(rootd,"lib-attack-audio"))
    td=os.listdir(os.path.join(rootd,"lib-target-audio"))
    save_a = os.path.join(rootd, "lib-attack-audio")
    save_t = os.path.join(rootd, "lib-target-audio")
    def gname(s):
        return s.split('-')[2]

#save attack-target
old_a_t_dict={}
for tpa,tpb in zip(ad,td):
    x=list(old_a_t_dict.keys())
    if gname(tpa) in x:
        old_a_t_dict[gname(tpa)] = old_a_t_dict[gname(tpa)]+[gname(tpb)]
    else:
        old_a_t_dict[gname(tpa)]=[gname(tpb)]


isrecoder={}
for ai,bi in zip(ad,td):
    isrecoder[ai]=bi

all_com=[]
for i in itertools.product(ad, td):
    all_com.append(i)

new_comp=[]

init_len=int(sorted(ad, key=lambda x: int(x.split('-')[0]))[-1].split('-')[0])+1

for (a,t) in all_com:
    if init_len>YOU_NEED_SUM:
        break
    if isrecoder[a]==t:
        continue
    if gname(a)==gname(t):
        continue
    old_t=old_a_t_dict.get(gname(a),None)
    if old_t!=None:
        if gname(t) in old_t:
            continue
        else:
            old_a_t_dict[gname(a)]=old_a_t_dict[gname(a)]+[gname(t)]
    if MODE=="TIMIT":
        newa=str(init_len)+"-attack-"+gname(a)+".wav"
        newt=str(init_len)+"-attack-"+gname(t)+".wav"
    elif MODE=="Lib":
        newa = str(init_len) + "-attack-" + "-".join(a.split("-")[2:])
        newt = str(init_len) + "-attack-" + "-".join(t.split("-")[2:])
    init_len+=1
    new_comp.append((newa, newt ,a , t))

print(new_comp)
for (newa,newt,a,t) in new_comp:
    xa,sr=sf.read(os.path.join(save_a,a))
    sf.write(os.path.join(save_a, newa), xa, sr)

    xt, sr = sf.read(os.path.join(save_t, t))
    sf.write(os.path.join(save_t, newt), xt, sr)

# confrim secure
if MODE=="TIMIT":
    ad=os.listdir(os.path.join(rootd,"timit-attack-audio"))
    td=os.listdir(os.path.join(rootd,"timit-target-audio"))
if MODE=="Lib":
    ad=os.listdir(os.path.join(rootd,"lib-attack-audio"))
    td=os.listdir(os.path.join(rootd,"lib-target-audio"))

for a,t in zip(ad,td):
    if gname(a)==gname(t):
        print("error!!!!!!!!!!!!!!")