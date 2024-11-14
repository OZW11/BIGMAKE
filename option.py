# 时间： 2024/11/14 20:25
import argparse

# 创建一个解析器对象，并设置描述信息
parser = argparse.ArgumentParser(description='lj_net')
parser.add_argument('--model', type=str, default='lj', help='choose a type of model')
# 地址
parser.add_argument('--dir_data', type=str, default='train_real', help='dataset directory')
parser.add_argument('--dir_test_ori_img', type=str, default='test_real', help='Original image address of test data')
parser.add_argument('--dir_test_noi_img', type=str, default='train_noise', help='Noise image address of test data')
# 分块
parser.add_argument('--patch_size', type=int, default=48, help='output patch size')
parser.add_argument('--n_pat_per_image', type=int, default=256,help='a image produce n patches')
parser.add_argument('--sigma', type=int, default=25,
                    help='sigma == 100 means blind, sigma == 200 means realnoise')
# 训练
parser.add_argument('--epoch', type=int, default=300, help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='input batch size for training')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--optimizer', default='ADAM', choices=('SGD', 'ADAM', 'RMSprop'), help='optimizer to use (SGD | '
                                                                                            'ADAM | RMSprop)')

args = parser.parse_args()
