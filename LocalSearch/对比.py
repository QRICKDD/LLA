# -- coding: utf-8 --
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
a=r"F:\SR-ATK\用于画图sampleaudio\timit-attack-audio\1-attack-MMDS0.wav"
t=r"F:\SR-ATK\用于画图sampleaudio\timit-target-audio\1-target-MPRD0.wav"
adv=r"F:\SR-ATK\测试啦lla\timitaudio\adv-1.wav"
a,sr=sf.read(a)
t,sr=sf.read(t)
adv,sr=sf.read(adv)
a/=np.linalg.norm(a)
a*=0.95
t/=np.linalg.norm(t)
t*=0.95

plt.plot(adv-a)
plt.plot(t-a)
plt.show()