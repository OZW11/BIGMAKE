# 时间： 2024/11/14 20:25
import argparse

# 创建一个解析器对象，并设置描述信息
parser = argparse.ArgumentParser(description='zwo_net')
parser.add_argument('--model', type=str, default='zwo', help='choose a type of model')
parser.add_argument('--model_dir', default='models/zwo_Ures', help='directory of the model')
parser.add_argument('--model_name', default='final_model.pth', help='the model name')
# 地址
parser.add_argument('--dir_train_ori_img', type=str, default='train_real', help='Original image address of train data')
parser.add_argument('--dir_train_noi_img', type=str, default='train_noise', help='Noise image address of train data')
parser.add_argument('--dir_test_ori_img', type=str, default='test_real', help='Original image address of test data')
parser.add_argument('--dir_test_noi_img', type=str, default='test_noise', help='Noise image address of test data')
parser.add_argument('--save_test_dir', type=str, default='save_test_image', help='save test image address')
# 分块
parser.add_argument('--patch_size', type=int, default=48, help='output patch size')
parser.add_argument('--n_pat_per_image', type=int, default=256,help='a image produce n patches')
parser.add_argument('--sigma', type=int, default=25, help='Gaussian noise variance')
parser.add_argument('--noise_type', type=str, default='gaussian', help='人工噪声类型：'
                                                                       '高斯gaussian，椒盐salt，泊松poisson，均匀uniform')
# 训练
parser.add_argument('--epoch', type=int, default=300, help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=40, help='input batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='input batch size for training')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--optimizer', default='ADAM', choices=('SGD', 'ADAM', 'RMSprop'), help='optimizer to use (SGD | '
                                                                                            'ADAM | RMSprop)')
parser.add_argument('--loss_func', type=str, default='l2', help='choose the loss function')
parser.add_argument('--start_epoch', type=int, default=0, help='the state is saved to here')
parser.add_argument('--save_model_epoch', type=int, default=10, help='训练几个epoch保存一次模型')
parser.add_argument('--training_mode', type=str, default='art', help='人工噪声用art，采集噪声用real')
parser.add_argument('--testing_mode', type=str, default='real', help='人工噪声用art，采集噪声用real')

args = parser.parse_args()
