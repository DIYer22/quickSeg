# -*- coding: utf-8 -*-
import sys
from os.path import abspath,join,dirname

yllibPath = abspath(join(dirname(abspath(__file__)),'./yl'))
if yllibPath not in sys.path:
    sys.path = [yllibPath] + sys.path

import  tool 
import  ylimg as imglib
import  ylml as mllib
import  ylnp as nplib
from tool import *
from ylimg import *
from ylml import *
from ylnp import *


if __name__ == '__main__':

    pass
