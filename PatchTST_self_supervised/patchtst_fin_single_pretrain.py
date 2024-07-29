import glob
import logging
import wandb
from src.models.patchTST import PatchTST
from src.learner import Learner, transfer_weights
from src.callback.tracking import *
from src.callback.patch_mask import *
from src.callback.transforms import *
from src.metrics import *
from src.basics import set_device
from datautils import *
from collections import deque
import random
import argparse
import sys

exp_id = 4
wandb.init(project=f"pretrain_{exp_id}")


parser = argparse.ArgumentParser()

if not os.path.exists('pretrain_log'):
    os.makedirs('pretrain_log')
logging.basicConfig(filename=f'pretrain_log/pretrain_{exp_id}.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


# Dataset and dataloader
parser.add_argument('--dset_pretrain', type=str, default='finance', help='dataset name')
parser.add_argument('--context_points', type=int, default=160, help='sequence length')
parser.add_argument('--target_points', type=int, default=40, help='forecast horizon')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--virtual_batch_size', type=int, default=5, help='virtual batch size')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for DataLoader')
parser.add_argument('--scaler', type=str, default='standard', help='scale the input data')
parser.add_argument('--features', type=str, default='M', help='for multivariate model or univariate model')
# Patch
parser.add_argument('--patch_len', type=int, default=8, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride between patch')
# RevIN
parser.add_argument('--revin', type=int, default=1, help='reversible instance normalization')

# Model args
parser.add_argument('--n_layers', type=int, default=6, help='number of Transformer layers')
parser.add_argument('--n_heads', type=int, default=32, help='number of Transformer heads')
parser.add_argument('--d_model', type=int, default=256, help='Transformer d_model')
parser.add_argument('--d_ff', type=int, default=1024, help='Tranformer MLP dimension')
parser.add_argument('--dropout', type=float, default=0.2, help='Transformer dropout')
parser.add_argument('--head_dropout', type=float, default=0.2, help='head dropout')
parser.add_argument('--activation', type=str, default='relu', help='activation function')
parser.add_argument('--learn_pe', type=bool, default=True, help='whether to learn positional encoding')
parser.add_argument('--shared_embedding', type=bool, default=True, help='whether to use shared embedding')
parser.add_argument('--res_attention', type=bool, default=False, help='whether to use residual attention')
parser.add_argument('--pre_norm', type=bool, default=True, help='whether to use pre-norm')
parser.add_argument('--pe', type=str, default='zeros',
                    choices=['zero', 'zeros', 'normal', 'gauss', 'uniform', 'sincos'],
                    help='type of positional encoding. Options: zero, zeros, normal, gauss, uniform, sincos')


# Pretrain mask
parser.add_argument('--mask_ratio', type=float, default=0.2, help='masking ratio for the input')
# Optimization args
parser.add_argument('--n_epochs_pretrain', type=int, default=15, help='number of pre-training epochs')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
parser.add_argument('--n_iterations', type=int, default=5, help="number of iterations ")
# model id to keep track of the number of models saved
parser.add_argument('--pretrained_model_id', type=int, default=exp_id, help='id of the saved pretrained model')
parser.add_argument('--model_type', type=str, default='based_model', help='for multivariate model or univariate model')


# 新加的内容
parser.add_argument('--data_path', type=str, default='None', help='data file')
parser.add_argument('--root_path', type=str, default='./dataset/processed_pretrain_single/', help='root path of the data file')
parser.add_argument('--freq', type=str, default='t',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

parser.add_argument("--loss_boundary", type=float, default=10.0, help="loss boundary of each batch for monitoring the model pretraining process")

parser.add_argument('--continue_train', action='store_true', help='continue training from last checkpoint')

args = parser.parse_args()
print('args:', args)


args.save_pretrained_model = 'patchtst_pretrained_cw ' + str(args.context_points) + '_patch ' + str \
    (args.patch_len) + '_stride ' + str(args.stride) + '_epochs-pretrain' + str(args.n_epochs_pretrain) + '_mask' + str \
                                 (args.mask_ratio) + '_model' + str(args.pretrained_model_id)

args.save_path = 'saved_models/' + args.dset_pretrain + '/masked_patchtst/' + args.model_type + '/'
if not os.path.exists(args.save_path): os.makedirs(args.save_path)

# get available GPU devide
set_device()


def get_model(c_in, args):
    """
    c_in: number of variables
    """
    # get number of patches
    num_patch = (max(args.context_points, args.patch_len) - args.patch_len) // args.stride + 1
    print('number of patches:', num_patch)

    # get model
    model = PatchTST(c_in=c_in,
                target_dim=args.target_points,
                patch_len=args.patch_len,
                stride=args.stride,
                num_patch=num_patch,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                d_model=args.d_model,
                shared_embedding=args.shared_embedding,
                d_ff=args.d_ff,
                dropout=args.dropout,
                head_dropout=args.head_dropout,
                act=args.activation,
                head_type='pretrain',
                res_attention=args.res_attention,
                learn_pe=args.learn_pe,
                pre_norm=args.pre_norm,
                pre_act=args.pe
                )
    # print out the model size
    print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model


def find_lr(dls):

    model = get_model(dls.vars, args)
    loss_func = torch.nn.MSELoss(reduction='mean')
    cbs = [RevInCB(dls.vars, denorm=False)] if args.revin else []
    cbs += [PatchMaskCB(patch_len=args.patch_len, stride=args.stride, mask_ratio=args.mask_ratio)]

    learn = Learner(dls, model, loss_func, lr=args.lr, cbs=cbs, loss_boundary=args.loss_boundary,
                    virtual_batch_size=args.virtual_batch_size)
    suggested_lr = learn.lr_finder()
    return suggested_lr


def pretrain_func(dls, lr=args.lr, model=None):
    # get model if not provided
    if model is None:
        model = get_model(dls.vars, args)
    # get loss
    loss_func = torch.nn.MSELoss(reduction='mean')
    # get callbacks
    cbs = [RevInCB(dls.vars, denorm=False)] if args.revin else []
    cbs += [
        PatchMaskCB(patch_len=args.patch_len, stride=args.stride, mask_ratio=args.mask_ratio),
        SaveModelCB(monitor='valid_loss', fname=args.save_pretrained_model,
                    path=args.save_path)
    ]
    # define learner
    learn = Learner(dls, model,
                    loss_func,
                    lr=lr,
                    cbs=cbs,
                    # metrics=[mse]
                    loss_boundary=args.loss_boundary,
                    virtual_batch_size=args.virtual_batch_size
                    )
    # fit the data to the model
    learn.fit_one_cycle(n_epochs=args.n_epochs_pretrain, lr_max=lr)

    train_loss = learn.recorder['train_loss']
    valid_loss = learn.recorder['valid_loss']
    df = pd.DataFrame(data={'train_loss': train_loss, 'valid_loss': valid_loss})
    df.to_csv(args.save_path + args.save_pretrained_model + '_losses.csv', float_format='%.6f', index=False)
    return model


def get_file_size(file_path):
    return os.path.getsize(file_path) / (1024 * 1024 * 1024)  # Convert to GB


def get_batch_files(pkl_files, max_batch_size=0.1):
    batch_files = []
    current_batch_size = 0

    for file in pkl_files:
        file_size = get_file_size(file)
        if current_batch_size + file_size > max_batch_size and batch_files:
            yield batch_files
            batch_files = []
            current_batch_size = 0

        batch_files.append(file)
        current_batch_size += file_size

    if batch_files:
        yield batch_files


if __name__ == '__main__':
    args.dset = args.dset_pretrain

    # 获取所有 pkl 文件
    all_pkl_files = sorted(glob.glob(os.path.join(args.root_path, '*.pkl')))

    model = None
    num_iterations = args.n_iterations  # 设置训练轮数

    for iteration in range(num_iterations):
        logging.info(f"Starting iteration {iteration + 1}/{num_iterations}")

        # 打乱文件顺序
        pkl_files = all_pkl_files.copy()
        random.shuffle(pkl_files)

        total_files = len(pkl_files)
        processed_files = 0

        for batch_files in get_batch_files(pkl_files, max_batch_size=0.03):
            processed_files += len(batch_files)
            logging.info(f"Iteration {iteration + 1}: Training on files {processed_files}/{total_files}")

            # 更新 args.data_paths
            args.data_paths = [os.path.relpath(file, args.root_path) for file in batch_files]
            logging.info(f"Data paths: {args.data_paths}")
            batch_size = sum(get_file_size(file) for file in batch_files)
            logging.info(f"Batch size: {batch_size:.2f} GB")

            try:
                # 创建 Dataset_Multi_Fin 实例
                dls = get_dls(args)

                suggested_lr = find_lr(dls)

                # 预训练
                model = pretrain_func(dls, suggested_lr, model)
                logging.info(
                    f'Iteration {iteration + 1}: Pretraining completed for batch {processed_files // len(batch_files)}')

            except Exception as e:
                logging.error(f"Error occurred while processing files: {args.data_paths}")
                logging.error(f"Error message: {str(e)}")
                continue  # Skip to the next batch of files


    logging.info('All pretraining completed')
    print('All pretraining completed')