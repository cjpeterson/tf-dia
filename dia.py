from PIL import Image
import numpy as np
import vgg19conv
import tensorflow as tf
import time
import os
import utilities

def saveimg(arr, name):
    if (arr.shape[0] == 1):
        arr = arr.reshape(arr.shape[1:])
    size = (arr.shape[1], arr.shape[0])
    arr = arr.transpose((2,0,1))
    means = vgg19conv.Vgg19Conv.VGG_MEAN
    blue = arr[0] + means[0]
    green = arr[1] + means[1]
    red = arr[2] + means[2]
    arr = np.array([red,green,blue])
    arr = arr.transpose(1,2,0)
    arr = arr.reshape((arr.shape[0]*arr.shape[1],3))
    arr = np.uint8(arr)
    samplearray = []
    for pixel in arr:
        samplearray.append(tuple(pixel))
    img = Image.new('RGB', size)
    img.putdata(samplearray)
    img.save(name+".png", 'PNG')
    
    return img
'''
def usemap(map, source):
    map_h = map.shape[0]
    map_w = map.shape[1]
    target = [None]*map_h
    for y in range(map_h):
        targetcolumn = [None]*map_w
        for x in range(map_w):
            p = map[y][x]
            newfeat = source[0][p[0]][p[1]]
            targetcolumn[x] = newfeat
        target[y] = targetcolumn
    target = np.array(target)
    return target.reshape((1,)+target.shape)
'''
def upsample(map, newsize):
    map_h = map.shape[0]
    map_w = map.shape[1]
    newmap = map*2
    #Cover edge cases of bad patches
    '''
    for m1 in range(map_h):
        p = newmap[m1][0]
        newmap[m1][0] = [p[0],p[1]+1]
    for m2 in range(map_w):
        p = newmap[0][m2]
        newmap[0][m2] = [p[0]+1,p[1]]
    '''
    newmap = np.repeat(newmap, 2, axis=0)
    newmap = np.repeat(newmap, 2, axis=1)
    while (newsize[0] < newmap.shape[0]):
        newmap = np.delete(newmap, newmap.shape[0]-1, axis=0)
    while (newsize[1] < newmap.shape[1]):
        newmap = np.delete(newmap, newmap.shape[1]-1, axis=1)
    #newmap = np.pad(newmap, pad_width=[[0,newsize[0]-newmap.shape[0]],
    #    [0,newsize[1]-newmap.shape[1]],[0,0]], mode='edge')
    
    return newmap

def average_map(map, source, patch_width):
    map_h = map.shape[0]
    map_w = map.shape[1]
    map_d = source.shape[3]
    source_h = source.shape[1]
    source_w = source.shape[2]
    target = np.zeros((map_h, map_w, map_d))
    for m1 in range(map_h):
        for m2 in range(map_w):
            #Define patch
            lowerh = 0-min((patch_width-1)//2,m1)
            upperh = min((patch_width-1)//2,map_h-m1-1)
            lowerw = 0-min((patch_width-1)//2,m2)
            upperw = min((patch_width-1)//2,map_w-m2-1)
            
            args = []
            for offset1 in range(lowerh, upperh+1):
                for offset2 in range(lowerw, upperw+1):
                    args.append((m1+offset1,m2+offset2,offset1,offset2))
            
            features = np.zeros(map_d)
            p_total = 0
            for arg in args:
                p = map[arg[0]][arg[1]]
                p = [p[0]-arg[2],p[1]-arg[3]]
                if (p[0] >= 0 and p[0] < source_h and
                    p[1] >= 0 and p[1] < source_w):
                    features = features + source[0][p[0]][p[1]]
                    p_total += 1
            features = features / p_total
            
            target[m1][m2] = features
    
    target = target.reshape((1,)+target.shape)
    return target
'''
def altreconstruct(map, source):
    map_h = map.shape[0]
    map_w = map.shape[1]
    target = np.zeros((map_h, map_w, 3))
    pwidth = 5
    for m1 in range(map_h):
        for m2 in range(map_w):
            #Define patch
            lowerh = 0-min((pwidth-1)//2,m1)
            upperh = min((pwidth-1)//2,map_h-m1-1)
            lowerw = 0-min((pwidth-1)//2,m2)
            upperw = min((pwidth-1)//2,map_w-m2-1)
            
            args = []
            for offset1 in range(lowerh, upperh+1):
                for offset2 in range(lowerw, upperw+1):
                    args.append((m1+offset1,m2+offset2,offset1,offset2))
            
            color = np.zeros(3)
            for arg in args:
                p = map[arg[0]][arg[1]]
                p = [p[0]-arg[2],p[1]-arg[3]]
                color = color + source[0][p[0]][p[1]]
            color = color / len(args)
            
            target[m1][m2] = color
    
    target = target.reshape((1,)+target.shape)
    return target
'''
def patchmatch(map, ffrom, fto, fpfrom, fpto, pwidth, searchrad, numiters=5):
    map_h = map.shape[0]
    map_w = map.shape[1]
    ito_h = fto.shape[1]
    ito_w = fto.shape[2]
    
    for iter in range(numiters):
        print('Iteration: {}/{}'.format(iter+1, numiters))
        
        mapchanged = False
        coordmod = (iter%2)*2 - 1
        for m1 in range(map_h):
            for m2 in range(map_w):
                #Get potential map arguments
                arg1 = [m1, m2]
                arg2 = [max(0,min(m1+coordmod,map_h-1)), m2]
                arg3 = [m1, max(0,min(m2+coordmod,map_w-1))]
                
                #Get corresponding pixels in to
                p1 = np.copy(map[arg1[0]][arg1[1]])
                p2 = np.copy(map[arg2[0]][arg2[1]])
                p2[0] = p2[0] + (m1-arg2[0])
                p3 = np.copy(map[arg3[0]][arg3[1]])
                p3[1] = p3[1] + (m2-arg3[1])
                
                #Define patch
                lowerh = 0-min((pwidth-1)//2,m1)
                upperh = min((pwidth-1)//2,map_h-m1-1)
                lowerw = 0-min((pwidth-1)//2,m2)
                upperw = min((pwidth-1)//2,map_w-m2-1)
                
                #Check if pixels yield valid patches
                #if (p2[0]+lowerh < 0) or (p2[0]+upperh >= ito_h):
                #    p2 = p1
                #if (p3[1]+lowerw < 0) or (p3[1]+upperw >= ito_w):
                #    p3 = p1
                
                #Check if pixel exists
                if (p2[0] < 0) or (p2[0] >= ito_h):
                    p2 = p1
                if (p3[1] < 0) or (p3[1] >= ito_w):
                    p3 = p1
                
                #Calculate loss
                ps = [p1,p2,p3]
                losses = [0,0,0]
                for i in range(3):
                    if (p1 == p2).all() and (p1 == p3).all():
                        break
                    if (i>0) and (ps[i] == p1).all():
                        losses[i] = losses[0]
                        continue
                    p = ps[i]
                    loss = 0
                    num_pix = 0
                    for offset1 in range(lowerh,upperh+1):
                        if ((p[0]+offset1 < 0) or (p[0]+offset1 >= ito_h)):
                            continue
                        for offset2 in range(lowerw,upperw+1):
                            if ((p[1]+offset2 < 0) or (p[1]+offset2 >= ito_w)):
                                continue
                            
                            fromy = m1+offset1
                            fromx = m2+offset2
                            toy = p[0]+offset1
                            tox = p[1]+offset2
                            f1 = ffrom[0][fromy][fromx] - fto[0][toy][tox]
                            f2 = fpfrom[0][fromy][fromx] - fpto[0][toy][tox]
                            loss += np.dot(f1,f1) + np.dot(f2,f2)
                            num_pix += 1
                    losses[i] = loss / num_pix
                
                #Take argument with lowest loss
                i = np.argmin(losses)
                map[m1][m2] = ps[i]
                if (i > 0):
                    mapchanged = True
                
                
                #Random search
                #We do a square search block to make it easier
                radius = searchrad
                curloss = losses[i]
                while (radius >= 1):
                    #Get pixel
                    center = map[m1][m2]
                    lowerrandh = round(max(0,
                        center[0]-radius))
                    upperrandh = round(min(ito_h-1,
                        center[0]+radius))
                    lowerrandw = round(max(0,
                        center[1]-radius))
                    upperrandw = round(min(ito_w-1,
                        center[1]+radius))
                    testpix = [np.random.randint(lowerrandh,upperrandh+1),
                        np.random.randint(lowerrandw,upperrandw+1)]
                    
                    #Test pixel
                    loss = 0
                    num_pix = 0
                    for offset1 in range(lowerh,upperh+1):
                        if ((testpix[0]+offset1 < 0) or
                            (testpix[0]+offset1 >= ito_h)):
                            continue
                        for offset2 in range(lowerw,upperw+1):
                            if ((testpix[1]+offset2 < 0) or
                                (testpix[1]+offset2 >= ito_w)):
                                continue
                            
                            fromy = m1+offset1
                            fromx = m2+offset2
                            toy = testpix[0]+offset1
                            tox = testpix[1]+offset2
                            f1 = ffrom[0][fromy][fromx] - fto[0][toy][tox]
                            f2 = fpfrom[0][fromy][fromx] - fpto[0][toy][tox]
                            loss += np.dot(f1,f1) + np.dot(f2,f2)
                            num_pix += 1
                    loss = loss / num_pix
                    if (loss < curloss):
                        curloss = loss
                        mapchanged = True
                        map[m1][m2] = testpix
                    radius *= 0.5
        if (not mapchanged):
            return
        

def get_deep_image_analogy(image_A_path, image_Bp_path, **options):
    
    debug = options.get('debug', False)
    full = options.get('full', True)
    iterations = options.get('iterations', 1000)
    pm_iters = options.get('pm_iters', 20)
    zero_img = options.get('zero_img', False)
    weights_path = options.get('weights_path', './vgg19_conv_partial.npy')
    tau_A = options.get('tau_A', 0.05)
    tau_Bp = options.get('tau_Bp', 0.05)
    
    #TODO: behavior not fully implemented
    skip = False
    
    starttime = int(time.time())
    if (debug):
        np.random.seed(123)
    
    if not os.path.exists("./output/"):
        os.makedirs("./output/")
    if not os.path.exists("./visualizations/"):
        os.makedirs("./visualizations/")
    
    print("Preparing data")
    
    #Load Images
    image_A = Image.open(image_A_path)
    image_A = image_A.convert('RGB')
    image_Bp = Image.open(image_Bp_path)
    image_Bp = image_Bp.convert('RGB')
    
    #Must be BGR
    means = vgg19conv.Vgg19Conv.VGG_MEAN
    red = np.array(image_A.getdata(band=0)) - means[2]
    green = np.array(image_A.getdata(band=1)) - means[1]
    blue = np.array(image_A.getdata(band=2)) - means[0]
    A_raw = np.concatenate((blue, green, red))
    A_raw = A_raw.reshape((1,3)+(image_A.size[1],image_A.size[0]))
    A_raw = A_raw.transpose(0,2,3,1)
    
    red = np.array(image_Bp.getdata(band=0)) - means[2]
    green = np.array(image_Bp.getdata(band=1)) - means[1]
    blue = np.array(image_Bp.getdata(band=2)) - means[0]
    Bp_raw = np.concatenate((blue, green, red))
    Bp_raw = Bp_raw.reshape((1,3)+(image_Bp.size[1],image_Bp.size[0]))
    Bp_raw = Bp_raw.transpose(0,2,3,1)
    
    
    #Load VGG-19 network
    model_weights = np.load(weights_path, encoding="latin1").item()
    vgg = vgg19conv.Vgg19Conv(model=model_weights)
    
    #Prepare convolution blocks
    single_conv_A = []
    single_conv_B = []
    full_conv_A = []
    full_conv_B = []
    A_size = A_raw.shape[1:3]
    B_size = Bp_raw.shape[1:3]
    feat = 3
    A_raw_placeholder = tf.placeholder(tf.float32, (None,)+A_size+(feat,))
    B_raw_placeholder = tf.placeholder(tf.float32, (None,)+B_size+(feat,))
    newblockAfull = vgg.get_block(A_raw_placeholder, 1)
    newblockBfull = vgg.get_block(B_raw_placeholder, 1)
    for L in range(1,6):
        A_side_placeholder = tf.placeholder(tf.float32, (None,)+A_size+(feat,))
        B_side_placeholder = tf.placeholder(tf.float32, (None,)+B_size+(feat,))
        newblockA = vgg.get_block(A_side_placeholder, L)
        newblockB = vgg.get_block(B_side_placeholder, L)
        if (L > 1):
            A_size = (int(np.ceil(A_size[0]/2.0)),int(np.ceil(A_size[1]/2.0)))
            B_size = (int(np.ceil(B_size[0]/2.0)),int(np.ceil(B_size[1]/2.0)))
        if (L > 3):
            feat = 512
        else:
            feat = 64*(2**(L-1))
        single_conv_A.append((A_side_placeholder, newblockA, A_size, feat))
        single_conv_B.append((B_side_placeholder, newblockB, B_size, feat))
        if (L > 1):
            newblockAfull = vgg.get_block(full_conv_A[L-2], L)
            newblockBfull = vgg.get_block(full_conv_B[L-2], L)
        full_conv_A.append(newblockAfull)
        full_conv_B.append(newblockBfull)
    
    
    #Prepare reconstruction operation
    if (not full):
        R_A = [None]*4
        R_Bp = [None]*4
        for i in range(4):
            R_A_L = tf.Variable(tf.random_normal(
                (1,)+single_conv_B[i][2]+(single_conv_B[i][3],), mean=10.0))
            warp_A = tf.placeholder(tf.float32,
                (1,)+single_conv_B[i+1][2]+(single_conv_B[i+1][3],))
            
            cost_A = vgg.get_block(R_A_L, i+2)
            cost_A = cost_A - warp_A
            cost_A = tf.reduce_sum(tf.multiply(cost_A, cost_A))
            op_A = tf.train.GradientDescentOptimizer(
                learning_rate=5e-3).minimize(cost_A, var_list=[R_A_L])
            clip_op = tf.assign(R_A_L, tf.clip_by_value(R_A_L, 0, 1e6))
            R_A[i] = (op_A, warp_A, R_A_L, cost_A, clip_op)
            
            R_Bp_L = tf.Variable(tf.random_normal(
                (1,)+single_conv_A[i][2]+(single_conv_A[i][3],), mean=10.0))
            warp_Bp = tf.placeholder(tf.float32,
                (1,)+single_conv_A[i+1][2]+(single_conv_A[i+1][3],))
            cost_Bp = vgg.get_block(R_Bp_L, i+2)
            cost_Bp = cost_Bp - warp_Bp
            cost_Bp = tf.reduce_sum(tf.multiply(cost_Bp, cost_Bp))
            op_Bp = tf.train.GradientDescentOptimizer(
                learning_rate=5e-3).minimize(cost_Bp, var_list=[R_Bp_L])
            clip_op = tf.assign(R_Bp_L, tf.clip_by_value(R_Bp_L, 0, 1e6))
            R_Bp[i] = (op_Bp, warp_Bp, R_Bp_L, cost_Bp, clip_op)
    
    
    if (full):
        #Prepare full reconstruction operation
        R_A_full = [None]*5
        R_Bp_full = [None]*5
        for i in range(5):
            R_A_full_L = tf.Variable(tf.random_normal(
                (1,)+single_conv_B[0][2]+(3,)))
            warp_A_full = tf.placeholder(tf.float32,
                (1,)+single_conv_B[i][2]+(single_conv_B[i][3],))
            
            full_block = R_A_full_L
            for x in range(i+1):
                full_block = vgg.get_block(full_block, x+1)
            cost_A_full = warp_A_full - full_block
            cost_A_full = tf.reduce_sum(tf.multiply(cost_A_full, cost_A_full))
            R_A_full_len = single_conv_B[0][2][0]*single_conv_B[0][2][1]
            bnds = ([[[[-means[0],-means[1],-means[2]]]]],
                [[[[255-means[0],255-means[1],255-means[2]]]]])
            var_to_bnds={R_A_full_L:bnds}
            lbfgs_A_full = tf.contrib.opt.ScipyOptimizerInterface(cost_A_full,
                var_to_bounds=var_to_bnds, var_list=[R_A_full_L],
                method='L-BFGS-B', options={'maxcor':20, 'disp':True,
                'maxiter':iterations})
            R_A_full[i] = (lbfgs_A_full, warp_A_full, R_A_full_L, cost_A_full)
            
            R_Bp_full_L = tf.Variable(tf.random_normal(
                (1,)+single_conv_A[0][2]+(3,)))
            warp_Bp_full = tf.placeholder(tf.float32,
                (1,)+single_conv_A[i][2]+(single_conv_A[i][3],))
            
            full_block = R_Bp_full_L
            for x in range(i+1):
                full_block = vgg.get_block(full_block, x+1)
            cost_Bp_full = warp_Bp_full - full_block
            cost_Bp_full = tf.reduce_sum(tf.multiply(cost_Bp_full,
                cost_Bp_full))
            R_Bp_full_len = single_conv_A[0][2][0]*single_conv_A[0][2][1]
            bnds = ([[[[-means[0],-means[1],-means[2]]]]],
                [[[[255-means[0],255-means[1],255-means[2]]]]])
            var_to_bnds={R_Bp_full_L:bnds}
            lbfgs_Bp_full = tf.contrib.opt.ScipyOptimizerInterface(
                cost_Bp_full, var_to_bounds=var_to_bnds,
                var_list=[R_Bp_full_L], method='L-BFGS-B',
                options={'maxcor':20, 'disp':True, 'maxiter':iterations})
            R_Bp_full[i] = (lbfgs_Bp_full, warp_Bp_full, R_Bp_full_L,
                cost_Bp_full)
    
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        #Use VGG-19 to extract feature layers
        print("Extracting feature layers")
        F_A = []
        F_Bp = []
        newF_A_L = single_conv_A[0][1].eval(
            feed_dict={single_conv_A[0][0]:A_raw})
        newF_Bp_L = single_conv_B[0][1].eval(
            feed_dict={single_conv_B[0][0]:Bp_raw})
        F_A.append(newF_A_L)
        F_Bp.append(newF_Bp_L)
        for L in range(2,6):
            newF_A_L = single_conv_A[L-1][1].eval(
                feed_dict={single_conv_A[L-1][0]:F_A[L-2]})
            newF_Bp_L = single_conv_B[L-1][1].eval(
                feed_dict={single_conv_B[L-1][0]:F_Bp[L-2]})
            F_A.append(newF_A_L)
            F_Bp.append(newF_Bp_L)
        
        
        #Initialize other feature containers
        F_Ap = [None]*5
        F_B = [None]*5
        F_A_normed = [None]*5
        F_Bp_normed = [None]*5
        for i in range(5):
            F_Ap[i] = F_A[i]
            F_B[i] = F_Bp[i]
            F_A_normed[i] = F_A[i]/np.linalg.norm(F_A[i],ord=2,axis=3,
                keepdims=True)
            F_Bp_normed[i] = F_Bp[i]/np.linalg.norm(F_Bp[i],ord=2,axis=3,
                keepdims=True)
        F_Ap_normed = [None]*5
        F_B_normed = [None]*5
        
        #Initialize phi_a_b and phi_b_a
        phi_a_b = np.zeros(A_size[0]*A_size[1]*2, dtype=np.int)
        phi_a_b = phi_a_b.reshape(A_size+(2,))
        for y in range(A_size[0]):
            for x in range(A_size[1]):
                phi_a_b[y][x] = np.array([
                    np.random.randint(0,B_size[0]),
                    np.random.randint(0,B_size[1])])
        phi_b_a = np.zeros(B_size[0]*B_size[1]*2, dtype=np.int)
        phi_b_a = phi_b_a.reshape(B_size+(2,))
        for y in range(B_size[0]):
            for x in range(B_size[1]):
                phi_b_a[y][x] = np.array([
                    np.random.randint(0,A_size[0]),
                    np.random.randint(0,A_size[1])])
        
        
        #Main loop
        pwidths = [5,5,3,3,3]
        maxrad = np.max([A_size[0], A_size[1], B_size[0], B_size[1]])
        searchrads = [4,4,6,6,maxrad]
        alphas = [0.1,0.6,0.7,0.8]
        numchecks = 10
        for n in range(5):
            L = 5-n
            print("Processing layer {}".format(L))
            
            #NNF search to update phi_a_b and phi_b_a
            F_Ap_normed[L-1] = F_Ap[L-1]/np.linalg.norm(F_Ap[L-1],ord=2,axis=3,
                keepdims=True)
            F_B_normed[L-1] = F_B[L-1]/np.linalg.norm(F_B[L-1],ord=2,axis=3,
                keepdims=True)
            print("Patchmatch phi_a_b:")
            patchmatch(phi_a_b, F_A_normed[L-1], F_B_normed[L-1],
                F_Ap_normed[L-1], F_Bp_normed[L-1], pwidths[L-1],
                searchrads[L-1], pm_iters)
            print("Patchmatch phi_b_a:")
            patchmatch(phi_b_a, F_B_normed[L-1], F_A_normed[L-1],
                F_Bp_normed[L-1], F_Ap_normed[L-1], pwidths[L-1],
                searchrads[L-1], pm_iters)
            
            if (debug):
                utilities.visF(F_A[L-1], "./visualizations/{}_F_A_{}".format(
                    starttime, L))
                utilities.visF(F_Bp[L-1], "./visualizations/{}_F_Bp_{}".format(
                    starttime, L))
            
            #TODO: consider making this L > skip+1
            if (L > 2 or (L > 1 and not skip)):
                if (debug):
                    utilities.upmaptest(phi_a_b, Bp_raw, L-1, means,
                        './visualizations/{}_upmap_{}'.format(starttime, L))
                    utilities.upmaptest(phi_b_a, A_raw, L-1, means,
                        './visualizations/{}_upmap_{}'.format(starttime, L))
                
                #Iteratively find RBp
                warp_Bp = average_map(phi_a_b, F_Bp[L-1], pwidths[L-1])
                if (debug):
                    utilities.visF(warp_Bp, "./visualizations/{}_F_warp_Bp_{}"
                        .format(starttime, L))
                
                if (full):
                    R_Bp_full[L-1][0].minimize(session=sess,
                        feed_dict={R_Bp_full[L-1][1]:warp_Bp})
                    R_Bp_full_L = sess.run(R_Bp_full[L-1][2])
                    saveimg(R_Bp_full_L, "./visualizations/{}_R_Bp_full_{}"
                        .format(starttime, L-1))
                    R_Bp_L = sess.run(full_conv_A[L-2-skip],
                        feed_dict={A_raw_placeholder:R_Bp_full_L})
                else:
                    print("Iterating on R_Bp")
                    #Per layer R
                    totalcost = 0
                    for iter in range(iterations):
                        _, _, cost = sess.run([R_Bp[L-2][0], R_Bp[L-2][4],
                            R_Bp[L-2][3]], feed_dict={R_Bp[L-2][1]:warp_Bp})
                        totalcost += cost
                        if (iter == 0):
                            print("Initial cost: {}".format(cost))
                            oldcost = cost
                            totalcost = 0
                        elif (iter%numchecks==0):
                            avgcost = totalcost / numchecks
                            delta = oldcost - avgcost
                            percentdiff = delta / oldcost
                            print("%diff: {:.10f} , iter {}".format(
                                percentdiff, iter))
                            oldcost = avgcost
                            totalcost = 0
                            #if (percentdiff < -0.01):
                            #    break
                    print("Final cost: " + repr(cost))
                    R_Bp_L = sess.run(R_Bp[L-2][2])
                if (debug):
                    utilities.visF(R_Bp_L, "./visualizations/{}_F_R_Bp_"
                        .format(starttime, L-1))
                
                #Calc WA
                featurenorms = np.linalg.norm(F_A[L-2-skip], ord=2, axis=3,
                    keepdims=True)
                featurenorms = featurenorms*featurenorms
                featurenorms = featurenorms - featurenorms.min()
                featurenorms = featurenorms / featurenorms.max()
                M_A = featurenorms>tau_A
                W_A = alphas[L-2-skip]*M_A
                #Assign FAp_L
                F_Ap[L-2-skip] = F_A[L-2-skip]*W_A + R_Bp_L*(1-W_A)
                
                
                #Iteratively find RA
                warp_A = average_map(phi_b_a, F_A[L-1], pwidths[L-1])
                if (debug):
                    utilities.visF(warp_A, "./visualizations/{}_F_warp_A_{}"
                        .format(starttime, L))
                
                if (full):
                    R_A_full[L-1][0].minimize(session=sess,
                        feed_dict={R_A_full[L-1][1]:warp_A})
                    R_A_full_L = sess.run(R_A_full[L-1][2])
                    saveimg(R_A_full_L, "./visualizations/{}_R_A_full_{}"
                        .format(starttime, L-1))
                    R_A_L = sess.run(full_conv_B[L-2-skip],
                        feed_dict={B_raw_placeholder:R_A_full_L})
                else:
                    print("Iterating on R_A")
                    totalcost = 0
                    for iter in range(iterations):
                        _, _, cost = sess.run([R_A[L-2][0], R_A[L-2][4],
                            R_A[L-2][3]], feed_dict={R_A[L-2][1]:warp_A})
                        totalcost += cost
                        if (iter == 0):
                            print("Initial cost: {}".format(cost))
                            oldcost = cost
                            totalcost = 0
                        elif (iter%numchecks==0):
                            avgcost = totalcost / numchecks
                            delta = oldcost - avgcost
                            percentdiff = delta / oldcost
                            print("%diff: {:.10f} , iter {}".format(
                                percentdiff, iter))
                            oldcost = avgcost
                            totalcost = 0
                            #if (percentdiff < -0.01):
                            #    break
                    print("Final cost: {}".format(cost))
                    R_A_L = sess.run(R_A[L-2][2])
                if (debug):
                    utilities.visF(R_A_L, "./visualizations/{}_F_R_A_".format(
                        starttime, L-1))
                
                #Calc WBp
                featurenorms = np.linalg.norm(F_Bp[L-2-skip], ord=2, axis=3,
                    keepdims=True)
                featurenorms = featurenorms*featurenorms
                featurenorms = featurenorms - featurenorms.min()
                featurenorms = featurenorms / featurenorms.max()
                M_Bp = featurenorms>tau_Bp
                W_Bp = alphas[L-2-skip]*M_Bp
                #Assign FB_L
                F_B[L-2-skip] = F_Bp[L-2-skip]*W_Bp + R_A_L*(1-W_Bp)
            
            elif (zero_img and (L == skip+1)):
                warp_Bp = average_map(phi_a_b, F_Bp[L-1], pwidths[L-1])
                
                if (full):
                    R_Bp_full[L-1][0].minimize(session=sess,
                        feed_dict={R_Bp_full[L-1][1]:warp_Bp})
                    R_Bp_full_L = sess.run(R_Bp_full[L-1][2])
                    saveimg(R_Bp_full_L, "./visualizations/{}_R_Bp_full_{}"
                        .format(starttime, L-1))
                
                warp_A = average_map(phi_b_a, F_A[L-1], pwidths[L-1])
                
                if (full):
                    R_A_full[L-1][0].minimize(session=sess,
                        feed_dict={R_A_full[L-1][1]:warp_A})
                    R_A_full_L = sess.run(R_A_full[L-1][2])
                    saveimg(R_A_full_L, "./visualizations/{}_R_A_full_{}"
                        .format(starttime, L-1))
            
            if (L > 1):
                #Upsample phi_a_b and phi_b_a
                phi_a_b = upsample(phi_a_b, single_conv_A[L-2][2])
                phi_b_a = upsample(phi_b_a, single_conv_B[L-2][2])
        
        #Calc image_Ap
        Ap_raw = average_map(phi_a_b, Bp_raw, 5)
        try:
            image_Ap = saveimg(Ap_raw, './output/{}_Ap'.format(starttime))
        except IOError:
            print("Error while saving final image. Attempting backup.")
            image_Ap = saveimg(Ap_raw, './{}_Ap'.format(starttime))
        
        #Calc image_B
        B_raw = average_map(phi_b_a, A_raw, 5)
        try:
            image_B = saveimg(B_raw, './output/{}_B'.format(starttime))
        except:
            print("Error while saving final image. Attempting backup.")
            image_B = saveimg(B_raw, './{}_B'.format(starttime))
        
        endtime = int(time.time())
        seconds_elapsed = endtime-starttime
        minutes_elapsed = seconds_elapsed // 60
        seconds_elapsed -= minutes_elapsed*60
        hours_elapsed = minutes_elapsed // 60
        minutes_elapsed -= hours_elapsed*60
        print("Time taken: {}:{:02d}:{:02d}".format(hours_elapsed,
            minutes_elapsed, seconds_elapsed))
        
        return (image_Ap, image_B, phi_a_b, phi_b_a)
