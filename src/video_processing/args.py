import argparse
from openpifpaf.network import nets
from openpifpaf import decoder

def cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Predict (2D pose and/or 3D location from images)
    # General
    parser.add_argument('--networks', nargs='+', help='Run pifpaf and/or monoloco', default=['monoloco'])
    parser.add_argument('images', nargs='*', help='input images')
    parser.add_argument('--glob', help='glob expression for input images (for many images)')
    parser.add_argument('-o', '--output-directory', help='Output directory')
    parser.add_argument('--output_types', nargs='+', default=['json'],
                                help='what to output: json keypoints skeleton for Pifpaf'
                                     'json bird front combined for Monoloco')
    parser.add_argument('--show', help='to show images', action='store_true')

    # Pifpaf
    nets.cli(parser)
    decoder.cli(parser, force_complete_pose=True, instance_threshold=0.15)
    parser.add_argument('--scale', default=1.0, type=float, help='change the scale of the image to preprocess')

    # Monoloco
    parser.add_argument('--model', help='path of MonoLoco model to load', required=False)
    parser.add_argument('--hidden_size', type=int, help='Number of hidden units in the model', default=512)
    parser.add_argument('--path_gt', help='path of json file with gt 3d localization',
                                default='data/arrays/names-kitti-190513-1754.json')
    parser.add_argument('--transform', help='transformation for the pose', default='None')
    parser.add_argument('--draw_box', help='to draw box in the images', action='store_true')
    parser.add_argument('--predict', help='whether to make prediction', action='store_true')
    parser.add_argument('--z_max', type=int, help='maximum meters distance for predictions', default=22)
    parser.add_argument('--n_dropout', type=int, help='Epistemic uncertainty evaluation', default=0)
    parser.add_argument('--dropout', type=float, help='dropout parameter', default=0.2)
    parser.add_argument('--webcam', help='monoloco streaming', action='store_true')

    args = parser.parse_args()
    return args