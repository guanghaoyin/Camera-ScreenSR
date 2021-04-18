import argparse
# import template

parser = argparse.ArgumentParser(description='SISR')

parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')
parser.add_argument('--template', default='.',
                    help='You can set various templates in option.py')

# Hardware specifications
#store_ture 只要运行时该变量有传参就将该变量设为True
parser.add_argument('--n_threads', type=int, default=2,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=2,
                    help='number of GPUs')
parser.add_argument('--GPU_ID', type=int, default=0,
                    help='ID of GPU')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# Data specifications
# parser.add_argument('--dir_data', type=str, default='../../../Dataset/SISR/Set14',
#                     help='dataset directory')
parser.add_argument('--dir_data', type=str, default='./dataset/SISR/DIV2K',
                    help='dataset directory')
parser.add_argument('--data_train', type=str, default='DIV2K',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='DIV2K',
                    help='test dataset name')
parser.add_argument('--dir_hr', type=str, default='HR',
                    help='hr dir name')
parser.add_argument('--dir_lr', type=str, default='LRLR',
                    help='lr dir name')
parser.add_argument('--n_train', type=int, default=800,
                    help='number of training set')
parser.add_argument('--n_val', type=int, default=5,
                    help='number of validation set')
parser.add_argument('--offset_val', type=int, default=800,
                    help='validation index offest')
parser.add_argument('--ext', type=str, default='img',
                    help='dataset file extension:sep_reset|img')
parser.add_argument('--scale', default='1',
                    help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=162,
                    help='output patch size')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--chop', action='store_true',
                    help='enable memory-efficient forward')

# Model specifications
parser.add_argument('--model', default='D',
                    help='model name')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')

# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=2000,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=112,
                    help='input batch size for training')
parser.add_argument('--split_batch', type=int, default=1,
                    help='split the batch into smaller chunks')
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')

# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--lr_decay', type=int, default=200,
                    help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='step',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')

# Loss specifications
parser.add_argument('--print_model', action='store_true',
                    help='print model')
parser.add_argument('--print_every', type=int, default=20,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save', type=str, default='.',
                    help='file name to save')
parser.add_argument('--save_results', action='store_true',
                    help='save output results')
# parser.add_argument('--save_models', action='store_true',
#                     help='save all intermediate models')
# parser.add_argument('--loss', type=str, default='1*L1+0.006*VGG54',
#                     help='loss function configuration')
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function configuration')
parser.add_argument('--skip_threshold', type=float, default='1e6',
                    help='skipping batch that has large error')
parser.add_argument('--load', type=str, default='.',
                    help='file name to load')
parser.add_argument('--load_model', type=str, default='.',
                    help='model name to load')
args = parser.parse_args()
# template.set_template(args)