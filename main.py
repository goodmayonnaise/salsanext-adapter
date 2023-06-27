

import os 
import time 
from datetime import datetime

from model.segmentor import EncoderDecoder
from model.salsanext import SalsaNextEncoder, SalsaNextDecoder
from data_loader.kitti import KITTI
from utils.pytorchtools import EarlyStopping
from utils.gpus import setup
from losses.loss import FocalLosswithDiceRegularizer
from train import train 

import torch
import torch.distributed as dist
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler, AdamW
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from datetime import timedelta



def main():

    # # gpu setting -----------------------------------------------------------------------------
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # gpus = os.environ["CUDA_VISIBLE_DEVICES"]
    # device = set_device(gpus)
    # use_gpu = torch.cuda.is_available()

    torch.cuda.manual_seed_all(777)
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    gpus = os.environ["CUDA_VISIBLE_DEVICES"]
    num_gpu = list(range(torch.cuda.device_count()))
    num_workers = len(gpus.split(",")) * 2
    timeout=timedelta(seconds=86400)
    dist.init_process_group(backend='nccl', rank=0, world_size=1, timeout=timeout)

    # setting model params --------------------------------------------------------------------
    epochs = 100
    batch_size = len(num_gpu)*10
    nclasses = 20 
    img_size = (256, 1024)

    # setting model ---------------------------------------------------------------------------
    backbone = SalsaNextEncoder(nclasses=nclasses)
    decode_head = SalsaNextDecoder(nclasses=nclasses)
    model = EncoderDecoder(backbone=backbone, decode_head=decode_head)
    model = DataParallel(model.cuda(), device_ids=num_gpu)
    # print(summary(model, (5, 3, 256, 1024)))
    # optimizer = Adam(model.to(device).parameters(), lr=1e-3)
    optimizer = AdamW(model.to(device).parameters(), lr=2e-5)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.001)
    criterion = FocalLosswithDiceRegularizer(reduction="mean")

    # setting data ----------------------------------------------------------------------------
    path = '/vit-adapter-kitti/data/kitti'
    train_dataset = KITTI(path, img_size, nclasses, 'train')
    train_loader = DataLoader(train_dataset, batch_size, num_workers=num_workers)
    val_dataset = KITTI(path, img_size, nclasses, 'val')
    val_loader = DataLoader(val_dataset, batch_size, num_workers=num_workers)

    # create dir for weight --------------------------------------------------------------------------------
    configs = "{}_batch{}_epoch{}_{}_{}".format(path[5:], batch_size, epochs, str(criterion).split('(')[0], str(optimizer).split( )[0])
    print("Configs:", configs)
    now = time.strftime('%m%d%H%M') 
    model_path = os.path.join("weights", configs, str(now))
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    earlystop = EarlyStopping(patience=10, verbose=True, path=os.path.join(model_path, 'earlystop.pt'))

    # write log --------------------------------------------------------------------------------
    metrics = {'train_loss':[], 'train_miou':[], 'train_acc':[],
               'val_loss':[], 'val_miou':[], 'val_acc':[]}
    
    if not os.path.exists(os.path.join(model_path, 'train')):
        os.makedirs(os.path.join(model_path, 'train'))
    if not os.path.exists(os.path.join(model_path, 'val')):
        os.makedirs(os.path.join(model_path, 'val'))
 
    writer_train = SummaryWriter(log_dir=os.path.join(model_path, 'train'))
    writer_val = SummaryWriter(log_dir = os.path.join(model_path, 'val'))

    with open(f'{model_path}/result.csv', 'a') as epoch_log:
        epoch_log.write('\n-----------------new-----------------\nepoch\ttrain loss\t val loss\ttrain miou\tval miou\ttrain acc\tval acc')

    t_s = datetime.now()
    print(f'\ntrain start time : {t_s}')
    train(model, epochs, train_loader, val_loader, optimizer, criterion, nclasses, scheduler, 
          model_path, earlystop, device, metrics, writer_train, writer_val)
    print(f'\n[train time information]\n\ttrain start time\t{t_s}\n\tend of train\t\t{datetime.now()}\n\ttotal train time\t{datetime.now()-t_s}')


if __name__ == "__main__":
    main()

    # input = torch.rand([5, 3, 1024, 256]).to(device)
    # from torchviz import make_dot 
    # make_dot(model(input), params=dict(model.named_parameters())).render("graph", format="png")
    # exit()
