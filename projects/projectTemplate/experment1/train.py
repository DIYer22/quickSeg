# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from lib import *
import sys,os
import lib
from lib import dicto, glob, getArgvDic,filename
from lib import show, loga, logl, imread, imsave
from configManager import (getImgGtNames, indexOf, readgt, readimg, 
                           setMod, togt, toimg, makeTrainEnv)

from config import c, cf

setMod('train')

from configManager import args
args.names = getImgGtNames(c.names)[:]
args.prefix = c.weightsPrefix
args.classn = 2
args.window = (64*10,64*10)
[ 20.     ,  29.96875]
# =============================================================================
# config BEGIN
# =============================================================================
args.update(
        batch=2,
        epoch=30,
        resume=0,
        epochSize = 10000,
        )
# =============================================================================
# config END
# =============================================================================




argListt, argsFromSys = getArgvDic()
args.update(argsFromSys)

makeTrainEnv(args)

if __name__ == '__main__':
    import trainInterface as train
    train.train()
    pass

