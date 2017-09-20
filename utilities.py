import numpy as np
from PIL import Image

#Ensure integrity of boundary conditions for 3x3 window
def testmap(map):
    #Assumes equal size images
    map_h = map.shape[0]
    map_w = map.shape[1]
    for y in range(map_h):
        for x in range(map_w):
            newpix = map[y][x]
            if (newpix[0] == 0):
                if (y != 0):
                    raise AssertionError
            if (newpix[0] == map_h-1):
                if (y != map_h-1):
                    raise AssertionError
            if (newpix[1] == 0):
                if (x != 0):
                    raise AssertionError
            if (newpix[1] == map_w-1):
                if (x != map_h-1):
                    raise AssertionError
            
            if (newpix[0] < 0) or (newpix[0] >= map_h):
                raise AssertionError
            if (newpix[1] < 0) or (newpix[1] >= map_w):
                raise AssertionError

#Create blocky upsampling of map and save as an image
def upmaptest(map, source, doubletimes, means, name):
    map_h = map.shape[0]
    map_w = map.shape[1]
    scalingfactor = 2**doubletimes
    target = np.zeros((map_h*scalingfactor,map_w*scalingfactor,3),
        dtype=np.int)
    for y in range(map_h):
        for x in range(map_w):
            p = map[y][x]
            p = [p[0]*scalingfactor, p[1]*scalingfactor]
            for offset1 in range(scalingfactor):
                for offset2 in range(scalingfactor):
                    target[y*scalingfactor+offset1][x*scalingfactor+offset2] =(
                        source[0][p[0]+offset1][p[1]+offset2])
    
    A_size = (map_w*scalingfactor,map_h*scalingfactor)
    target = target.transpose((2,0,1))
    blue = target[0] + means[0]
    green = target[1] + means[1]
    red = target[2] + means[2]
    target = np.array([red,green,blue])
    target = target.transpose(1,2,0)
    target = target.reshape((map_h*map_w*scalingfactor**2,3))
    target = np.uint8(target)
    samplearray = []
    for pixel in target:
        samplearray.append(tuple(pixel))
    img = Image.new('RGB', A_size)
    img.putdata(samplearray)
    img.save(name+".png", 'PNG')

#Visualize strength of activations in a feature map
def visF(F, name):
    F_h = F.shape[1]
    F_w = F.shape[2]
    F = np.linalg.norm(F, ord=2, axis=3, keepdims=True)
    F = F - F.min()
    F = F / F.max()
    F = F.reshape((F_h*F_w))
    F = np.uint8(F*255)
    img = Image.new('L', (F_w, F_h))
    img.putdata(F)
    img.save(name+".png", 'PNG')
