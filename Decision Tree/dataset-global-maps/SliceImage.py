from PIL import Image
from itertools import product
import os

def tile(filename, dir_in, dir_out, d):
    name, ext = os.path.splitext(filename)
    img = Image.open(os.path.join(dir_in, filename))
    w, h = img.size

    a = 0
    
    grid = product(range(0, h-h%d, d), range(0, w-w%d, d))
    for i, j in grid:
        box = (j, i, j+d, i+d)
        os.system(f"mkdir {dir_out}/{a}")
        out = os.path.join(f'{dir_out}/{a}', f'{name}_{a}{ext}')
        img.crop(box).save(out)

        a += 1

tile("botanic_map.png", '/home/admsistemas/Documents/DatasetGlobalMaps', '/home/admsistemas/Documents/DatasetGlobalMaps', 111)