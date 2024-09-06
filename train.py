import sys
import time
import numpy as np
from torch import optim
import torchvision.transforms as T
from data_loading import BasicDataset
from path_hyperparameter import ph
import torch
from losses import FCCDN_loss_without_seg
import os
import logging
import random
import wandb

from models.model import BaseNet
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score, JaccardIndex, CohenKappa
from hack import train_val
from dataset_process import compute_mean_std
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator




class DataLoaderX(DataLoader):
    """Using prefetch_generator to accelerate data loading

    原本 PyTorch 默认的 DataLoader 会创建一些 worker 线程来预读取新的数据，但是除非这些线程的数据全部都被清空，这些线程才会读下一批数据。
    使用 prefetch_generator，我们可以保证线程不会等待，每个线程都总有至少一个数据在加载。

    Parameter:
        DataLoader(class): torch.utils.data.DataLoader.
    """

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def random_seed(SEED):
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = True  # keep convolution algorithm deterministic

    torch.backends.cudnn.benchmark = True

def auto_experiment():
    random_seed(SEED=ph.random_seed)
    try:
        train_net(dataset_name=ph.dataset_name)
    except KeyboardInterrupt:
        logging.info('Interrupt')
        sys.exit(0)


def train_net(dataset_name):
    """
    This is the workflow of training model and evaluating model,
    note that the dataset should be organized as
    :obj:`dataset_name`/`train` or `val`/`t1` or `t2` or `label`

    Parameter:
        dataset_name(str): name of dataset

    Return:
        return nothing
    """

    t1_mean, t1_std = compute_mean_std(images_dir=f'./{dataset_name}/train/t1/')
    t2_mean, t2_std = compute_mean_std(images_dir=f'./{dataset_name}/train/t2/')

    dataset_args = dict(t1_mean=t1_mean, t1_std=t1_std, t2_mean=t2_mean, t2_std=t2_std)
    train_dataset = BasicDataset(t1_images_dir=f'./{dataset_name}/train/t1/',
                                 t2_images_dir=f'./{dataset_name}/train/t2/',
                                 labels_dir=f'./{dataset_name}/train/label/',
                                 train=True, **dataset_args)
    val_dataset = BasicDataset(t1_images_dir=f'./{dataset_name}/val/t1/',
                               t2_images_dir=f'./{dataset_name}/val/t2/',
                               labels_dir=f'./{dataset_name}/val/label/',
                               train=False, **dataset_args)

    n_train = len(train_dataset)
    n_val = len(val_dataset)

    loader_args = dict(num_workers=8,
                       prefetch_factor=5,
                       persistent_workers=True,
                       # pin_memeory=True,
                       )
    train_loader = DataLoaderX(train_dataset, shuffle=True, drop_last=False, batch_size=ph.batch_size, **loader_args)
    val_loader = DataLoaderX(val_dataset, shuffle=False, drop_last=False,
                             batch_size=ph.batch_size * ph.inference_ratio, **loader_args)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # working device
    logging.basicConfig(level=logging.INFO)
    localtime = time.asctime(time.localtime(time.time()))
    hyperparameter_dict = ph.state_dict()
    hyperparameter_dict['time'] = localtime

    log_wandb = wandb.init(project=ph.log_wandb_project, resume='allow', anonymous='must',
                           settings=wandb.Settings(start_method='thread'),
                           config=hyperparameter_dict)
    logging.info(f'''Starting training:
        Epochs:          {ph.epochs}
        Batch size:      {ph.batch_size}
        Learning rate:   {ph.learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {ph.save_checkpoint}
        save best model: {ph.save_best_model}
        Device:          {device.type}
        Mixed Precision: {ph.amp}
    ''')

    net = BaseNet(3, 1)  # change detection model
    net = net.to(device=device)
    optimizer = optim.AdamW(net.parameters(), lr=ph.learning_rate,
                            weight_decay=ph.weight_decay)  # optimizer
    warmup_lr = np.arange(1e-7, ph.learning_rate,
                          (ph.learning_rate - 1e-7) / ph.warm_up_step)  # warm up learning rate

    grad_scaler = torch.cuda.amp.GradScaler()


    if ph.load:
        checkpoint = torch.load(ph.load, map_location=device)
        net.load_state_dict(checkpoint['net'])
        logging.info(f'Model loaded from {ph.load}')
        if 'optimizer' in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer'])
            for g in optimizer.param_groups:
                g['lr'] = ph.learning_rate
            optimizer.param_groups[0]['capturable'] = True

    total_step = 0  # logging step
    lr = ph.learning_rate  # learning rate

    criterion = FCCDN_loss_without_seg  # loss function

    best_metrics = dict.fromkeys(['best_f1score', 'lowest loss'], 0)  # best evaluation metrics
    metric_collection = MetricCollection({
        'accuracy': Accuracy().to(device=device),
        'precision': Precision().to(device=device),
        'recall': Recall().to(device=device),
        'f1score': F1Score().to(device=device),
        'JaccardIndex': JaccardIndex(2).to(device=device),
        'CohenKappa':CohenKappa(2).to(device=device),
    })

    to_pilimg = T.ToPILImage()


    checkpoint_path = f'./{dataset_name}_checkpoint/'
    best_f1score_model_path = f'./{dataset_name}_best_f1score_model/'
    best_loss_model_path = f'./{dataset_name}_best_loss_model/'

    non_improved_epoch = 0

    for epoch in range(ph.epochs):

        log_wandb, net, optimizer, grad_scaler, total_step, lr = \
            train_val(
                mode='train', dataset_name=dataset_name,
                dataloader=train_loader, device=device, log_wandb=log_wandb, net=net,
                optimizer=optimizer, total_step=total_step, lr=lr, criterion=criterion,
                metric_collection=metric_collection, to_pilimg=to_pilimg, epoch=epoch,
                warmup_lr=warmup_lr, grad_scaler=grad_scaler
            )

        if epoch >= ph.evaluate_epoch:
            with torch.no_grad():
                log_wandb, net, optimizer, total_step, lr, best_metrics, non_improved_epoch = \
                    train_val(
                        mode='val', dataset_name=dataset_name,
                        dataloader=val_loader, device=device, log_wandb=log_wandb, net=net,
                        optimizer=optimizer, total_step=total_step, lr=lr, criterion=criterion,
                        metric_collection=metric_collection, to_pilimg=to_pilimg, epoch=epoch,
                        best_metrics=best_metrics, checkpoint_path=checkpoint_path,
                        best_f1score_model_path=best_f1score_model_path, best_loss_model_path=best_loss_model_path,
                        non_improved_epoch=non_improved_epoch
                    )

    wandb.finish()


if __name__ == '__main__':

    auto_experiment()

