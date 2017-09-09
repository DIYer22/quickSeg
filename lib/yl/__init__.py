# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import tool
import yldb
import py3
import undetermined

try:
    import ylimg
    import ylnp
    import ylml
except Exception,e:
    ylnp = "there may not GUI on this system, can't use ylnp"
    ylimg= "there may not GUI on this system, can't use ylimg"
    ylml= "there may not GUI on this system, can't use ylml"
if __name__ == '__main__':
    print (yldb, py3, tool, ylimg, ylnp)
    tool.importAllFunCode('yl')
    pass

