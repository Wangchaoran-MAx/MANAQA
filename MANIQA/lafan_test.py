import os
import torch
import numpy as np
import logging
import time
import torch.nn as nn
import random

from torchvision import transforms
from torch.utils.data import DataLoader
from models.manaqa import MANAQA
from config import Config
from utils.process import RandCrop, ToTensor, RandHorizontalFlip, Normalize, five_point_crop
from scipy.stats import spearmanr, pearsonr
from data.Lafan1 import Lafan1
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# 设置随机数种子：如果不使用seed设置种子值，则每次输出的随机数都是不同的
def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 参数int，随机数种子。如果seed为负值，则会映射为正值。
    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cudnn：GPU加速库
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def set_logging(config):
    if not os.path.exists(config.log_path):
        os.makedirs(config.log_path)
    filename = os.path.join(config.log_path, config.log_file)
    logging.basicConfig(
        filename=filename,
        level=logging.INFO,
        filemode='w',
        format='[%(asctime)s %(levelname)-8s] %(message)s',
        datefmt='%Y%m%d %H:%M:%S'
    )

def train_epoch(epoch, net, criterion, optimizer, scheduler, train_loader):
    losses = []
    net.train()
    # save data for one epoch

    # pred：预测的  labels：真实的
    pred_epoch = []
    labels_epoch = []

    for data in tqdm(train_loader):
        # x_d：data数据中的图片列  labels：得分列
        x_d = data['d_motion_org'].cuda()
        labels = data['score']
        labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()
        #print("pred_d = net(x_d)",np.shape(x_d))
        pred_d = net(x_d)

        optimizer.zero_grad()
        loss = criterion(torch.squeeze(pred_d), labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()

        # save results in one epoch
        pred_batch_numpy = pred_d.data.cpu().numpy()
        labels_batch_numpy = labels.data.cpu().numpy()
        pred_epoch = np.append(pred_epoch, pred_batch_numpy)
        labels_epoch = np.append(labels_epoch, labels_batch_numpy)

    # compute correlation coefficient
    rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

    ret_loss = np.mean(losses)
    logging.info('train epoch:{} / loss:{:.4} / SRCC:{:.4} / PLCC:{:.4}'.format(epoch + 1, ret_loss, rho_s, rho_p))

    return ret_loss, rho_s, rho_p,pred_epoch, labels_epoch 

def eval_epoch(config, epoch, net, criterion, test_loader):
    with torch.no_grad():
        losses = []
        net.eval()
        # save data for one epoch
        pred_epoch = []
        labels_epoch = []

        for data in tqdm(test_loader):
            pred = 0
            for i in range(config.num_avg_val):
                x_d = data['d_motion_org'].cuda()
                labels = data['score']
                labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()
              #  x_d = five_point_crop(i, d_img=x_d, config=config)
                pred += net(x_d)

            pred /= config.num_avg_val
            # compute loss
            loss = criterion(torch.squeeze(pred), labels)
            losses.append(loss.item())

            # save results in one epoch
            pred_batch_numpy = pred.data.cpu().numpy()
            labels_batch_numpy = labels.data.cpu().numpy()
            #print(pred_batch_numpy)
           
        
            pred_epoch = np.append(pred_epoch, pred_batch_numpy)
            labels_epoch = np.append(labels_epoch, labels_batch_numpy)

        # compute correlation coefficient
        rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

        logging.info(
            'Epoch:{} ===== loss:{:.4} ===== SRCC:{:.4} ===== PLCC:{:.4}'.format(epoch + 1, np.mean(losses), rho_s,
                                                                                 rho_p))
        return np.mean(losses), rho_s, rho_p,pred_epoch, labels_epoch


if __name__ == '__main__':
    cpu_num = 1
    # 释放缓存分配器当前持有的且未占用的缓存显存
    
    torch.cuda.empty_cache()
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)
    # 深度学习训练的线程数
    setup_seed(20)

    # config file
    train_set = ['subject1', 'subject2', 'subject3']
    val_set = ['subject4']
    test_set = ['subject5']
    try_train_set = ['subject1']
    try_val_set = ['subject2']
    config = Config({
        # dataset path
        #"db_name": "PIPAL",
        #训练图像的文件夹路径
        #"train_dis_path": "/hy-tmp/lafan_test/",
        # 验证图像的文件夹路径
       # "val_dis_path": "/hy-tmp/lafan_test/",
        #训练的图像带有标签的名字
        "train_txt_file_name": "/hy-tmp/final_test/try_train_score.txt",
        # 验证的图像带有标签的名字
        "val_txt_file_name": "/hy-tmp/final_test/try_val_score.txt",
        

        # optimization
        "batch_size":8,
        "learning_rate": (1e-4)/4,
        "weight_decay": 1e-5,
        "n_epoch": 1000,
        "val_freq": 1,

        "T_max": 50,
        "eta_min": 0,
        "num_avg_val": 5,
        "crop_size": 224,
        "num_workers": 0,

        # model
        "patch_size":8,
        "img_size": 224,
        # 嵌入向量的大小
        "embed_dim": 768,
         #MLP（前馈）层的维数
        "dim_mlp": 768,
        "num_heads": [4, 4],
        "window_size": 4,
        "depths": [2, 2],
        "num_outputs": 1,
        "num_tab": 2,
        "scale": 0.13,

        # load & save checkpoint
        "model_name": "model_manaqa",
        "output_path": "./output",
        "snap_path": "./output/models/",  # directory for saving checkpoint
        "log_path": "./output/log/manaqa/",
        "log_file": ".txt",
        "tensorboard_path": "./output/tensorboard_manaqa/"
    })

    if not os.path.exists(config.output_path):
        os.mkdir(config.output_path)
    # snap 一种包的管理工具
    if not os.path.exists(config.snap_path):
        os.mkdir(config.snap_path)

    if not os.path.exists(config.tensorboard_path):
        os.mkdir(config.tensorboard_path)

    config.snap_path += config.model_name
    config.log_file = config.model_name + config.log_file
    config.tensorboard_path += config.model_name

    set_logging(config)
    logging.info(config)
    #SummaryWriter一般是用来记录训练过程中的学习率和损失函数的变化，通过命令行可以得到一个网址，查看起来也很方便
    writer = SummaryWriter(config.tensorboard_path)

    # data load
    train_dataset = Lafan1(
        #dis_path=config.train_dis_path,
        actors=try_train_set,
        txt_file_name=config.train_txt_file_name,
        transform=transforms.Compose(
            [
                #RandCrop(config.crop_size),
                Normalize(0.5, 0.5),
                #RandHorizontalFlip(),
                ToTensor()
            ]
        ),

    )
    print("train_data读取完")
    val_dataset = Lafan1(
        #dis_path=config.val_dis_path,
        actors=try_val_set,
        txt_file_name=config.val_txt_file_name,
        transform=transforms.Compose([Normalize(0.5, 0.5), ToTensor()]),
        
    )
    print("val_data读取完")
    logging.info('number of train scenes: {}'.format(len(train_dataset)))
    logging.info('number of val scenes: {}'.format(len(val_dataset)))

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        drop_last=True,
        shuffle=True
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        drop_last=True,
        shuffle=False
    )
    net = MANAQA(
        embed_dim=config.embed_dim,
        num_outputs=config.num_outputs,
        dim_mlp=config.dim_mlp,
        patch_size=config.patch_size,
        img_size=config.img_size,
        window_size=config.window_size,
        depths=config.depths,
        num_heads=config.num_heads,
        num_tab=config.num_tab,
        scale=config.scale
    )
    net = nn.DataParallel(net)
    net = net.cuda()

    # loss function
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=config.eta_min)

    # make directory for saving weights
    if not os.path.exists(config.snap_path):
        os.mkdir(config.snap_path)

    # train & validation
    losses, scores = [], []
    best_srocc = 0
    best_plcc = 0
    

    for epoch in range(0, config.n_epoch):
        start_time = time.time()
        pred_epoch,labels_epoch=[],[]
        logging.info('Running training epoch {}'.format(epoch + 1))
        loss_val, rho_s, rho_p, pred_epoch,labels_epoch  = train_epoch(epoch, net, criterion, optimizer, scheduler, train_loader)
        file=open('/hy-tmp/MANIQA/output/log/train/'+str(epoch)+'.txt','w')
        for i in range(0,len(labels_epoch)):
            file.write(str(labels_epoch[i]) + ' ' + str(pred_epoch[i]) + '\n')
    
        writer.add_scalar("Train_loss", loss_val, epoch)
        writer.add_scalar("SRCC", rho_s, epoch)
        writer.add_scalar("PLCC", rho_p, epoch)
        pred_epoch1, labels_epoch1=[],[]
        if (epoch + 1) % config.val_freq == 0:
            logging.info('Starting eval...')
            logging.info('Running testing in epoch {}'.format(epoch + 1))
            loss, rho_s, rho_p,pred_epoch1, labels_epoch1 = eval_epoch(config, epoch, net, criterion, val_loader)
            logging.info('Eval done...')
            file1=open('/hy-tmp/MANIQA/output/log/eval/'+str(epoch)+'.txt','w')
            for i in range(0,len(labels_epoch1)):
                file1.write(str(labels_epoch1[i]) + ' ' + str(pred_epoch1[i]) + '\n')
            if rho_s > best_srocc or rho_p > best_plcc:
                best_srocc = rho_s
                best_plcc = rho_p
                # save weights
                model_name = "epoch{}".format(epoch + 1)
                model_save_path = os.path.join(config.snap_path, model_name)
                torch.save(net, model_save_path)
                logging.info(
                    'Saving weights and model of epoch{}, SRCC:{}, PLCC:{}'.format(epoch + 1, best_srocc, best_plcc))
        file.close()
        file1.close()

        logging.info('Epoch {} done. Time: {:.2}min'.format(epoch + 1, (time.time() - start_time) / 60))