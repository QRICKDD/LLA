import os
import soundfile as sf
def getf(id,root,filenames,dirnew):
    for item in filenames:
        if item.split('-')[0]==str(id):
            x,sr=sf.read(os.path.join(root,item))
            x=x[:16000]
            assert len(x)==16000
            sf.write(os.path.join(dirnew,item),x,16000)
            x, sr = sf.read(os.path.join(dirnew,item))
            assert len(x)==16000
    return
filenames=os.listdir(r"F:\SR-ATK\lib-new\attack")
dirnew=r"F:\SR-ATK\lib-new\aa"
root=r"F:\SR-ATK\lib-new\attack"

start=list(range(51,62))
for id in start:
    getf(id,root,filenames,dirnew)
