# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import sys,os
import numpy as np
import lib
from lib import dicto, glob, getArgvDic, findints,pathjoin
from lib import show, loga, logl, imread, imsave
from lib import Evalu,diceEvalu
from configManager import (getImgGtNames, indexOf, readgt, readimg, 
                           setMod, togt, toimg, makeTestEnv, doc)
from train import c, cf, args
setMod('test')

args.out = pathjoin(c.tmpdir,'test/png')

# =============================================================================
# config BEGIN
# =============================================================================
args.update(
        restore=-1,
#        step=None,
        
        )
# =============================================================================
# config END
# =============================================================================

if args.restore == -1:
    pas = [p[len(args.prefix):] for p in glob(args.prefix+'*')]
    args.restore = len(pas) and max(map(lambda s:len(findints(s)) and findints(s)[-1],pas))

makeTestEnv(args)


if __name__ == '__main__':
    import predictInterface 
    c.predictInterface = predictInterface
    predict = predictInterface.predict 
#    c.predict = predict
    e = Evalu(diceEvalu,
#              evaluName='restore-%s'%restore,
              testNames=c.names,
#              loadcsv=1,
              logFormat='dice:{dice:.3f}, loss:{loss:.3f}',
              sortkey='loss',
#              loged=False,
#              saveResoult=False,
              )
    for name in c.names:
        prob = predict(toimg(name))
        divInt8 = ((prob[...,1]-prob[...,0])*127).clip(-127,127).astype(np.int8)
        re = divInt8>0
        img,gt = readimg(name),readgt(name)>.5
        e.evalu(re,gt,name)
        show(img,gt,re)
#        diff = binaryDiff(re,gt)
#        show(img,diff,re)
#        show(img,diff)
#        show(diff)
#        yellowImg=gt[...,None]*img+(npa-[255,255,0]).astype(np.uint8)*~gt[...,None]
#        show(yellowImg,diff)
#        imsave(pathjoin(args.out,name+'.png'),uint8(re))
    
    print args.restore,e.loss.mean()


#map(lambda n:show(readimg(n),e[n],readgt(n)),e.low(80).index[:])















