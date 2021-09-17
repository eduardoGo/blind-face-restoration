from PIL import Image                                              
import os, sys                       

path = "C:\\Users\\eduardo\\Documents\\9 periodo\\image processing\\projeto\\ab2-final\\GFPGAN\\datasets\\ffhq\\ffhq_512\\"
dirs = os.listdir( path )                                       

def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((512,512), Image.ANTIALIAS)
            print("saving on", f+'.png')
            imResize.save(f+'.png', 'png', quality=80)


resize()