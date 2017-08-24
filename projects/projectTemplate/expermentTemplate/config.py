# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from lib import dicto,dirname, basename,os,log,fileJoinPath
from lib import show, loga, logl, imread, imsave

from configManager import (getImgGtNames, indexOf, readgt, readimg, 
                           setMod, togt, toimg)
from configManager import cf,c
# =============================================================================
# config BEGIN
# =============================================================================
cf.netdir = 'unet'
cf.jobName = None
cf.expName = None




#cf.trainArgs = dicto(
##        batch=1,
##        epoch=50,
##        resume=None,
##        lr=None,
#        )

cf.trainGlob =  u'G:\\experiment\\Data\\HKU-IS\\Imgs\\*.jpg'
cf.toGtPath = lambda path:path.replace('.jpg','.png')

cf.test = 0.1
cf.toTestGtPath = None
#cf.testArgs = None

cf.predictGlob = u'G:\\experiment\\Data\\HKU-IS\\Imgs\\*.jpg'

# =============================================================================
# config END
# =============================================================================

filePath = fileJoinPath(__file__)
jobDir = (os.path.split(dirname(filePath))[-1])
expDir = (os.path.split((filePath))[-1])

cf.jobName = cf.jobName or jobDir
cf.expName = cf.expName or expDir

cf.savename = '%s_%s_%s'%(cf.netdir,cf.expName,cf.jobName)

cf.toTestGtPath = cf.toTestGtPath or cf.toGtPath
#cf.testArgs = cf.testArgs or cf.trainArgs

c.update(cf)

setMod('train')
#setMod('test')

if __name__ == '__main__':
    if c.mod == 'train':
        from train import main as trainMain
        trainMain()
    if c.mod == 'test':
        from train import testMain
        testMain()
    if c.mod == 'predict':
        from predict import predictMain
        predictMain()
        
    pass




