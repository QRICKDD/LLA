# -- coding: utf-8 --
# -- coding: utf-8 --
# -- coding: utf-8 --
from matplotlib.pyplot import MultipleLocator

import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
import os
#全局设置画图
plt.figure(figsize=(6,7))


#定义
o_audio_file=r"F:\SR-ATK\sampleaudio\lib-attack-audio\35-attack-121-121726-0000.flac"
t_audio_file=r"F:\SR-ATK\sampleaudio\lib-target-audio\35-attack-6930-75918-0012.flac"



#source-aduio and target-audio
ox,sr=sf.read(o_audio_file)
tx,sr=sf.read(t_audio_file)
ox/=np.linalg.norm(ox,ord=np.inf)
tx/=np.linalg.norm(tx,ord=np.inf)
ox*=0.95
tx*=0.95
dx=tx-ox
plt.subplot(4,3,1)
plt.title("source-audio")
plt.xlim(0, 16000)
plt.xticks([])
plt.yticks([])
plt.plot(ox)
plt.subplot(4,3,2)
plt.title("target-audio")
plt.xlim(0, 16000)
plt.xticks([])
plt.yticks([])
plt.plot(tx)
plt.subplot(4,3,3)
plt.title("residual")
plt.xlabel("$L_2$=23.446")
plt.xlim(0, 16000)
plt.xticks([])
plt.yticks([])
plt.plot(dx)




S1=r"F:\SR-ATK\expMY\fake_4969_2.4772760717441633.wav"
S2=r"F:\SR-ATK\expMY\fake_14801_1.8635933807562668.wav"
S3=r"F:\SR-ATK\expMY\fake_24818_1.7154136692430864.wav"
s1,sr=sf.read(S1)
s2,sr=sf.read(S2)
s3,sr=sf.read(S3)
s1=s1-ox
s2=s2-ox
s3=s3-ox

plt.subplot(4,3,4)
plt.xlim(0, 16000)
plt.ylabel("Proposed")
plt.xlabel("$L_2$=2.47")
plt.xticks([])
plt.yticks([])
plt.plot(dx)
plt.plot(s1)
plt.subplot(4,3,5)
plt.xlabel("$L_2$=1.86")
plt.xlim(0, 16000)
plt.xticks([])
plt.yticks([])
plt.plot(dx)
plt.plot(s2)
plt.subplot(4,3,6)
plt.xlim(0, 16000)
plt.xlabel("$L_2$=1.71")
plt.xticks([])
plt.yticks([])
plt.plot(dx)
plt.plot(s3)


HSJ1=r"F:\SR-ATK\expH\fake_4873_8.769917071998243.wav"
HSJ2=r"F:\SR-ATK\expH\fake_14937_5.161145803447437.wav"
HSJ3=r"F:\SR-ATK\expH\fake_24356_3.381679619493476.wav"
H1,sr=sf.read(HSJ1)
H2,sr=sf.read(HSJ2)
H3,sr=sf.read(HSJ3)
h1=H1-ox
h2=H2-ox
h3=H3-ox

plt.subplot(4,3,7)
plt.xlim(0, 16000)
plt.ylabel("HSJA")
plt.xlabel("$L_2$=8.76")
plt.xticks([])
plt.yticks([])
plt.plot(dx)
plt.plot(h1)
plt.subplot(4,3,8)
plt.xlabel("$L_2$=5.16")
plt.xlim(0, 16000)
plt.xticks([])
plt.yticks([])
plt.plot(dx)
plt.plot(h2)
plt.subplot(4,3,9)
plt.xlabel("$L_2$=3.38")
plt.xlim(0, 16000)
plt.xticks([])
plt.yticks([])
plt.plot(dx)
plt.plot(h3)


MYF1=r"F:\SR-ATK\exppath\fake_4966_11.575861188678418.wav"
MYF2=r"F:\SR-ATK\exppath\fake_15407_8.784853907569275.wav"
MYF3=r"F:\SR-ATK\exppath\fake_24727_7.306666499908197.wav"
mf1,sr=sf.read(MYF1)
mf2,sr=sf.read(MYF2)
mf3,sr=sf.read(MYF3)
mf1=mf1-ox
mf2=mf2-ox
mf3=mf3-ox
plt.subplot(4,3,10)
plt.xlim(0, 16000)
plt.xticks([])
plt.yticks([])
plt.ylabel("SIGN-OPT")
plt.xlabel("$L_2$=11.57\nQueries: 5K")
plt.plot(dx)
plt.plot(mf1)
plt.subplot(4,3,11)
plt.xlim(0, 16000)
plt.xlabel("$L_2$=8.78\nQueries: 15K")
plt.xticks([])
plt.yticks([])
plt.plot(dx)
plt.plot(mf2)
plt.subplot(4,3,12)
plt.xlim(0, 16000)
plt.xlabel("$L_2$=5.48\nQueries: 25K")
plt.xticks([])
plt.yticks([])
plt.plot(dx)
plt.plot(mf3)

#plt.tight_layout()
plt.tight_layout(pad=1, w_pad=0, h_pad=0)
plt.savefig(os.path.join("F:\SR-ATK\picture",'CP.pdf'), dpi=800)
plt.show()


