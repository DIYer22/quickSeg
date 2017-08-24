# -*- coding: utf-8 -*-
import sys
from os.path import abspath,join,dirname
absLibpPath = None
libpath = absLibpPath or abspath(join(dirname(abspath(__file__)),'../../../lib/'))
if libpath not in sys.path:
    sys.path = [join(libpath,'yl'),libpath]+sys.path

import  tool 
import  ylimg as imglib
import  ylnp as nplib
from tool import *
from ylimg import *
from ylnp import *

#import lib517
#from lib517 import *

import configManager


if __name__ == '__main__':

    pass
