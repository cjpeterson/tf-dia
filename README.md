# tf-dia
"Deep Image Analogies" implemented using tensorflow

The authors of the paper have released a version that runs on CUDA. If you have a modern NVIDIA GPU, it is recommended that you use their program: [Link](https://github.com/msracver/Deep-Image-Analogy)

Paper: https://arxiv.org/abs/1705.01088

All code is licensed under the AGPL v3.0

### Pre-requisites

* Python (only 3.5 has been tested)
* SciPy
* TensorFlow
* numpy (TensorFlow pre-requisite)

It is recommended to use pip to install these packages. Once they're installed, in a command prompt or terminal, navigate to a folder where you would like to store the folder containing these files and type

`git clone --recursive github.com/cjpeterson/tf-dia.git`

(The --recursive option is not necessary here, but is recommended in general.) Navigate into the folder in your terminal.

**Lastly, you will need a copy** of some pre-trained VGG-19 weights in the .npy format. We only need the convolutional layers up to (and including) 5-1. If you don't have some already, you may download the full set [here (548 MB)](https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs), or the partial set [here (47 MB)](https://mega.nz/#!ExIzTZxA!3nAgY2soeU5HGcah2HYTWuIQYz48b-xp5Gf2r6Q3I0Q). The partial set must be unzipped. (These are both copies of the weights trained by Simonyan and Zisserman.)

### Usage

The primary command you'll use is

`python.exe main.py ./pathtoimageA.png ./pathtoimageBp.png`

Where the two paths point to the images you'd like to make the analogy on. The default options are set up to give good results without unnecessary computation time. If you'd like to see a list of options, input

`python.exe main.py --help`

Recommended image sizes are 448x448 or 224x224. The program should accept either image at an arbitrary resolution, but please note that these are being passed through pre-trained convolutions, so scaling up a picture of a head may cause the network to no longer recognize it as a head.

### Performance

The program is CPU-bound. The deconvolutions are multi-threaded, but patchmatch is currently single-threaded. My machine has an i5-6600k processer (3.5 GHz). The default operation, running on two 448x448 images, takes **7 hours and 35 minutes**. Every 20 deconvolution iterations takes ~6 minutes, and every 1 patchmatch iteration takes ~6 minutes. The default is 1000 deconv. iter. and 20 patchmatch iter.

If you'd like a preview of the final images, it's recommended to set deconvolutions to 30 iterations and patchmatch to 2 iterations. Like this

`python.exe main.py ./pathtoimageA.png ./pathtoimageB.png --d_iters 30 --pm_iters 2`

For two 448x448 images, this takes **26 minutes** on my machine.
