import tensorflow as tf
import vgg19conv
from PIL import Image
import numpy as np
import os
import sys

if (len(sys.argv) < 2) or (not os.path.exists(sys.argv[1])):
    print("Error: Must supply a valid image path as the first argument.")
    quit()

if not os.path.exists("./showtau/"):
    os.makedirs("./showtau/")

target_layer = 3

image_A = Image.open(sys.argv[1])
image_A = image_A.convert('RGB')

#Convert to BGR
means = vgg19conv.Vgg19Conv.VGG_MEAN
red = np.array(image_A.getdata(band=0)) - means[2]
green = np.array(image_A.getdata(band=1)) - means[1]
blue = np.array(image_A.getdata(band=2)) - means[0]
A_raw = np.concatenate((blue, green, red))
A_raw = A_raw.reshape((1,3)+(image_A.size[1],image_A.size[0]))
A_raw = A_raw.transpose(0,2,3,1)

#Load VGG-19 network
model_weights = np.load("vgg19.npy", encoding="latin1").item()
vgg = vgg19conv.Vgg19Conv(model=model_weights)

#Prepare convolution block
full_conv_A = []
A_size = A_raw.shape[1:3]
A_raw_placeholder = tf.placeholder(tf.float32, (None,)+A_size+(3,))
newblockAfull = vgg.get_block(A_raw_placeholder, 1)
for L in range(1,target_layer+1):
    if (L > 1):
        newblockAfull = vgg.get_block(full_conv_A[L-2], L)
    full_conv_A.append(newblockAfull)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    #Extract feature layer
    F = full_conv_A[-1].eval(feed_dict={A_raw_placeholder:A_raw})
    
    #Display results of different thresholds
    F_h = F.shape[1]
    F_w = F.shape[2]
    for x in range(20):
        if (x < 5):
            tau = (x+1)*.01
        elif (x < 10):
            tau = (x-4)*.02+.05
        elif (x < 15):
            tau = (x-9)*.03+.15
        else:
            tau = (x-14)*.04+.3
        F_x = np.linalg.norm(F, ord=2, axis=3, keepdims=True)
        F_x = F_x * F_x
        F_x = F_x - F_x.min()
        F_x = F_x / F_x.max()
        
        F_x = F_x > tau
        F_x = F_x.reshape((F_h*F_w))
        F_x = np.uint8(F_x*255)
        img = Image.new('L', (F_w, F_h))
        img.putdata(F_x)
        
        img.save("showtau/{:.2f}.png".format(tau), 'PNG')
