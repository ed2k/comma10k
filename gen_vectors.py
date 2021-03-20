import numpy as np
import matplotlib.pyplot as plt

from skimage import measure
from skimage.io import imread
from skimage.color import rgb2gray,rgba2rgb

MASK = dict(road=(1,32), back=(1,128), me=(0,204), lane=(0,255), car=(1,255))



# rgba image in, return 1 for the interested, 0 the rest
def mask_out(r, color_type):
    #print(r.shape)
    h,w,_ = r.shape
    r=r[:,:,:3]
    r2=np.zeros((h,w))
    # road 1:32 back 1:128 me 0:204 lane 0:255 car 1:255
    mask = MASK
    for x in range(w):
        for y in range(h):
            if r[y,x,mask[color_type][0]] == mask[color_type][1]:
                r2[y,x] = 1 # road
            elif color_type == 'road' and r[y,x,mask['lane'][0]] == mask['lane'][1]:
                r2[y,x] = 1 # convert lane to road
            elif color_type == 'road' and r[y,x,mask['me'][0]] == mask['me'][1]:
                r2[y,x] = 1 # convert me to road
            else:
                r2[y,x] = 0
    return r2

# car 255, lane 192, road 127 , back 64 , me 0
#r[r==(64,32,32)] = 127 # back   0
#r[r==(128,128,96)] = 0 # road .25
#r[r==(204,0,255)] = 64 # me    .5
#r[r==(255,0,0)] = 192 # lane   .75
#r[r==(0,255,102)] = 255 # car 1


def uniq_color(r):
    v = {}
    for x in range(w):
        for y in range(h):
            a = r[y,x]         
            if a not in v:
                v[a] = 1
    return list(v.keys())


def get_approx_contour(conts):
    result = {}
    for color_type in ['lane', 'road', 'me', 'car']:
        approx_c = []
        for c in conts[color_type]:
            contour = np.array(c)
            coords = measure.approximate_polygon(contour, tolerance=2)
            approx_c.append(coords.tolist())
        result[color_type] = approx_c
    return result


def one_fname(img):
    result = {}
    for color_type in ['lane', 'road', 'me', 'car']:
        r = mask_out(img, color_type)
        contours = measure.find_contours(r, 0.1)
        print(color_type, len(contours))
        print([len(c) for c in contours])
        result[color_type] = [c.tolist() for c in contours]
    return result

import os
import json
import psutil
import random

def cpu_balance():
    per = psutil.sensors_temperatures()['coretemp'][1:]
    mask = []
    min_temp = 100
    min_idx = -1
    for idx in range(5):
       if per[idx].current < min_temp:
           min_temp = per[idx].current
           min_idx = idx
    if min_idx > -1:
      p = psutil.Process()
      mask = min_idx *2 + random.randint(0, 1)
      p.cpu_affinity([mask])
      print('affinity', mask)

for fname in os.listdir('masks'):
    if fname[-4:] != '.png':
        continue
    data = {}
    gen_contour = True
    j_path = os.path.join('vectors', fname[:-4]+'.json')
    if os.path.exists(j_path):
        with open(j_path) as fjson:
            data = json.load(fjson)
            if 'contours' in data:
                gen_contour = False
    print(fname)
    #cpu_balance()
    img = imread(os.path.join('masks', fname))
    if gen_contour:
        j_format = one_fname(img)
        data['contours'] = j_format
    data['approx_c'] = get_approx_contour(data['contours'])
    with open(j_path, 'w') as fjson:
        json.dump(data, fjson)
    #break
        
        

        


