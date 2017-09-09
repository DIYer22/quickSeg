# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import lib
from lib import dicto,dirname, basename,os,log,fileJoinPath, pathjoin
from lib import show, loga, logl, imread, imsave

from configManager import (getImgGtNames, indexOf, readgt, readimg, 
                           setMod, togt, toimg)
from configManager import cf,c
# =============================================================================
# config BEGIN
# =============================================================================
cf.netdir = 'res-unet1'
cf.project = None
cf.experment = None


cf.trainGlob = u'/home/dl/experiment/salDataset/SalBenchmark-master/Data/HKU-IS/Imgs/*.jpg'
cf.toGtPath = lambda path:path.replace('.jpg','.png')

cf.test = 0.1
cf.toTestGtPath = None

#cf.predictGlob = u'G:/experiment/Data/HKU-IS/Imgs/*.jpg'
#cf.predictGlob = '/home/dl/datasOnWindows/carMaskData/test/*.jpg'
# =============================================================================
# config END
# =============================================================================


filePath = fileJoinPath(__file__)
jobDir = (os.path.split(dirname(filePath))[-1])
expDir = (os.path.split((filePath))[-1])

cf.project = cf.project or jobDir
cf.experment = cf.experment or expDir

cf.savename = '%s-%s-%s'%(cf.netdir,cf.experment,cf.project)

cf.toTestGtPath = cf.toTestGtPath or cf.toGtPath
#cf.testArgs = cf.testArgs or cf.trainArgs



c.update(cf)
c.cf = cf


c.weightsPrefix = fileJoinPath(__file__,pathjoin(c.tmpdir,'weihgts/%s-%s'%(c.netdir,c.experment)))
#show- map(readimg,c.names[:10])
if __name__ == '__main__':
        
    pass




