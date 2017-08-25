# -*- coding: utf-8 -*-
import sys,os

if __name__ == '__main__':
    from shutil import rmtree,move
    if os.path.isdir('yl'):
        rmtree('yl',True)
    if os.path.isdir('yllib'):
        rmtree('yllib',True)
    cmd = '''
git clone https://github.com/DIYer22/yllib/
move yllib/python/yl yl
rmdir /s/q yllib
echo "newst yllib ok!"

'''
    sh = '''git clone https://github.com/DIYer22/yllib/
mv yllib/python/yl yl
rm  yllib -rf
echo "newst yllib ok!"
'''
    win = 'win' in sys.platform 
    if win:
        os.system(cmd) # 失效了
    else :
        os.system(sh)
    pass
