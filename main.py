import dia
import argparse

#Parse input
parser = argparse.ArgumentParser()
parser.add_argument("image_A_path", help="Path to the first input image")
parser.add_argument("image_Bp_path", help="Path to the second input image")
parser.add_argument("--weights_path", default=argparse.SUPPRESS,
    help="Path to vgg19 weights (default: './vgg19_conv_partial.npy')")
parser.add_argument("--d_iters", type=int, default=argparse.SUPPRESS,
    help="Number of iterations to use in each deconvolution operation " +
    "(default=1000)")
parser.add_argument("--pm_iters", type=int, default=argparse.SUPPRESS,
    help="Number of iterations to perform in each patchmatch operation " +
    "(default=20)")
parser.add_argument("--partial", action="store_true",
    help="Use single-layer deconvolution, as in the paper (default: " +
    "deconvolution targets the image layer, and is finally passed upwards " +
    "to the target network layer)")
parser.add_argument("--debug", action="store_true",
    help="Enable seed and extra visualizations")
parser.add_argument("--zero_img", action="store_true",
    help="Perform a final deconvolution from Layer 1 and save as a " +
    "visualization (Note: only works if partial is disabled)")
parser.add_argument("--tau_A", type=int, default=argparse.SUPPRESS,
    help="Threshold for preserving content structures from A (default: 0.05)")
parser.add_argument("--tau_Bp", type=int, default=argparse.SUPPRESS,
    help="Threshold for preserving content structures from Bp (default: 0.05)")
args = parser.parse_args()
args = vars(args)

#Prepare parsed input
options = {}
options['debug'] = args.get('debug', False)
options['full'] = not args.get('partial', False)
options['iterations'] = args.get('d_iters', 1000)
options['pm_iters'] = args.get('pm_iters', 20)
options['zero_img'] = args.get('zero_img', False)
options['weights_path'] = args.get('weights_path', './vgg19_conv_partial.npy')
options['tau_A'] = args.get('tau_A', 0.05)
options['tau_Bp'] = args.get('tau_Bp', 0.05)

#Execute program
dia.get_deep_image_analogy(args['image_A_path'], args['image_Bp_path'],
    **options)
