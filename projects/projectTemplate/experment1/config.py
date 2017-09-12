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
cf.netdir = 'res-unet1-simg'
cf.project = None
cf.experment = None


cf.trainGlob = u'/home/dl/datasOnWindows/sealandship/train/*.jpg'
cf.toGtPath = lambda path:path.replace('.jpg','.png')

cf.val = u'/home/dl/datasOnWindows/sealandship/test/*.jpg'
cf.toValGtPath = None

#cf.testGlob = u'G:/experiment/Data/HKU-IS/Imgs/*.jpg'
#cf.testGlob = '/home/dl/datasOnWindows/carMaskData/test/*.jpg'
# =============================================================================
# config END
# =============================================================================


filePath = fileJoinPath(__file__)
jobDir = (os.path.split(dirname(filePath))[-1])
expDir = (os.path.split((filePath))[-1])

cf.project = cf.project or jobDir
cf.experment = cf.experment or expDir

cf.savename = '%s-%s-%s'%(cf.netdir,cf.experment,cf.project)

cf.toValGtPath = cf.toValGtPath or cf.toGtPath
#cf.valArgs = cf.valArgs or cf.trainArgs



c.update(cf)
c.cf = cf


c.weightsPrefix = fileJoinPath(__file__,pathjoin(c.tmpdir,'weihgts/%s-%s'%(c.netdir,c.experment)))
#show- map(readimg,c.names[:10])
if __name__ == '__main__':
        
    pass




