# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import sys,os
import numpy as np
import lib
from lib import dicto, glob, getArgvDic, findints,pathjoin
from lib import show, loga, logl, imread, imsave
from lib import Evalu,diceEvalu
from lib import *
from configManager import (getImgGtNames, indexOf, readgt, readimg, 
                           setMod, togt, toimg, makeValEnv, doc)
from train import c, cf, args
setMod('val')

args.out = pathjoin(c.tmpdir,'val/png')

# =============================================================================
# config BEGIN
# =============================================================================
args.update(
        restore=-1,
        step=None,
        
        )
# =============================================================================
# config END
# =============================================================================

if args.restore == -1:
    pas = [p[len(args.prefix):] for p in glob(args.prefix+'*')]
    args.restore = len(pas) and max(map(lambda s:len(findints(s)) and findints(s)[-1],pas))

makeValEnv(args)

if __name__ == '__main__':
    import predictInterface 
    c.predictInterface = predictInterface
    predict = predictInterface.predict 
#    c.predict = predict
    e = Evalu(diceEvalu,
#              evaluName='restore-%s'%restore,
              valNames=c.names,
#              loadcsv=1,
              logFormat='dice:{dice:.3f}, loss:{loss:.3f}',
              sortkey='loss',
#              loged=False,
#              saveResoult=False,
              )
    c.names.sort(key=lambda x:readgt(x).shape[0])
    for name in c.names[:]:
        img,gt = readimg(name),readgt(name)
        prob = predict(toimg(name))
        re = prob.argmax(2)
        ind = re==2
        re[re==1] = 2
        re[ind]=1
        e.evalu(re,gt,name)
        gtc = labelToColor(gt,[[0,0,0],[1.,0,0],[1,1,1]])
        rec = labelToColor(re,[[0,0,0],[1.,0,0],[1,1,1]])
        show(img,gtc,rec)
#        diff = binaryDiff(re,gt)
#        show(img,diff,re)
#        show(img,diff)
#        show(diff)
#        yellowImg=gt[...,None]*img+(npa-[255,255,0]).astype(np.uint8)*~gt[...,None]
#        show(yellowImg,diff)
#        imsave(pathjoin(args.out,name+'.png'),uint8(re))
    
    print args.restore,e.loss.mean()


#map(lambda n:show(readimg(n),e[n],readgt(n)),e.low(80).index[:])















