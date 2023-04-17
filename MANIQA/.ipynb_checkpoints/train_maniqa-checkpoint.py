import os
import torch
import numpy as np
import logging
import time
import torch.nn as nn
import random


from torchvision import transforms
from torch.utils.data import DataLoader
from models.maniqa import MANIQA
from config import Config
from utils.process import RandCrop, ToTensor, RandHorizontalFlip, Normalize, five_point_crop
from scipy.stats import spearmanr, pearsonr
from data.pipal21 import PIPAL21
from torch.utils.tensorboard import SummaryWriter 
from tqdm import tqdm

#os.environ：获取环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#设置随机数种子：如果不使用seed设置种子值，则每次输出的随机数都是不同的
def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    #参数int，随机数种子。如果seed为负值，则会映射为正值。
    torch.manual_seed(seed)
    
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    #cudnn：GPU加速库
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

#日志初始化
def set_logging(config):
    #创建路径
    if not os.path.exists(config.log_path): 
        os.makedirs(config.log_path)
    filename = os.path.join(config.log_path, config.log_file)
    logging.basicConfig(
        level=logging.INFO,
        filename=filename,
        filemode='w',
        format='[%(asctime)s %(levelname)-8s] %(message)s',
        datefmt='%Y%m%d %H:%M:%S'
    )

#、optimizer 优化器  
def train_epoch(epoch, net, criterion, optimizer, scheduler, train_loader):
    losses = []
    net.train()
    # save data for one epoch
    
    #pred：预测的  labels：真实的
    pred_epoch = []
    labels_epoch = []
    
    for data in tqdm(train_loader):
        #x_d：data数据中的图片列  labels：得分列
        x_d = data['d_img_org'].cuda()
        labels = data['score']
        #去除size为1的维度,包括行和列。
        labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()  
        #对图片进行预测
        pred_d = net(x_d)
        #清空过往梯度
        optimizer.zero_grad()
        #输入图像和标签，计算损失函数
        loss = criterion(torch.squeeze(pred_d), labels)
        losses.append(loss.item())
        #反向传播，计算当前梯度；
        loss.backward()
        #根据梯度更新网络参数
        optimizer.step()
        scheduler.step()

        # save results in one epoch
        #预测得分和真实得分压入对应的epoch中
        pred_batch_numpy = pred_d.data.cpu().numpy()
        labels_batch_numpy = labels.data.cpu().numpy()
        pred_epoch = np.append(pred_epoch, pred_batch_numpy)
        labels_epoch = np.append(labels_epoch, labels_batch_numpy)
    
    # compute correlation coefficient
    #自定义函数计算SRCC与PLCC的值
    rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    #求取平均值
    ret_loss = np.mean(losses)
    #日志打印
    logging.info('train epoch:{} / loss:{:.4} / SRCC:{:.4} / PLCC:{:.4}'.format(epoch + 1, ret_loss, rho_s, rho_p))

    return ret_loss, rho_s, rho_p


def eval_epoch(config, epoch, net, criterion, test_loader):
    #表明当前计算不需要反向传播，使用之后，强制后边的内容不进行计算图的构建
    with torch.no_grad():
        losses = []
        net.eval()
        # save data for one epoch
        pred_epoch = []
        labels_epoch = []

        for data in tqdm(test_loader):
            pred = 0
            for i in range(config.num_avg_val):
                x_d = data['d_img_org'].cuda()
                labels = data['score']
                labels =torch.squeeze(labels.type(torch.FloatTensor)).cuda()
                #五点截取，（总共有五种情况）比如说一张图片分成了九块，取最中间的那块和四角的四块
                x_d = five_point_crop(i, d_img=x_d, config=config)
                #将所有块的预测值加起来
                pred += net(x_d)
            #最终的预测值再除以块数
            pred /= config.num_avg_val
            # compute loss
            loss = criterion(torch.squeeze(pred), labels)
            losses.append(loss.item())

            # save results in one epoch
            pred_batch_numpy = pred.data.cpu().numpy()
            labels_batch_numpy = labels.data.cpu().numpy()
            pred_epoch = np.append(pred_epoch, pred_batch_numpy)
            labels_epoch = np.append(labels_epoch, labels_batch_numpy)
        
        # compute correlation coefficient
        rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

        logging.info('Epoch:{} ===== loss:{:.4} ===== SRCC:{:.4} ===== PLCC:{:.4}'.format(epoch + 1, np.mean(losses), rho_s, rho_p))
        return np.mean(losses), rho_s, rho_p


if __name__ == '__main__':
    cpu_num = 1
    #配置环境变量
    torch.cuda.empty_cache()
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    setup_seed(20)

    # config file
    config = Config({
        # dataset path    这里我做了修改

        "db_name": "PIPAL",
        "train_dis_path": "/hy-tmp/mydata/distorted_images/",
        "val_dis_path": "/hy-tmp/mydata/val_image/",
        "train_txt_file_name": "/hy-tmp/mydata/mos_with_names.txt",
        "val_txt_file_name": "/hy-tmp/mydata/val_mos.txt",

        # optimization
        "batch_size": 8,
        "learning_rate": 1e-5,
        "weight_decay": 1e-5,
        
        "n_epoch": 300,
        #模型验证频率（通信频率）
        "val_freq": 1,
        #T_max：Cosine是个周期函数，这里的T_max就是这个周期的一半，如果你将T_max设置为10，则学习率衰减的周期是20个epoch，其中前10个epoch从学习率的初值（也是最大值）下降到最低值，后10个epoch从学习率的最低值上升到最大值
        "T_max": 50,
        "eta_min": 0,
        "num_avg_val": 5,
        "crop_size": 224,
        #GPU的工作数
        "num_workers": 0,

        # model
        "patch_size": 8,
        "img_size": 224,
        #嵌入向量的大小
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
        "model_name": "model_maniqa",
        "output_path": "./output",
        "snap_path": "./output/models/",               # directory for saving checkpoint
        "log_path": "./output/log/maniqa/",
        "log_file": ".txt",
        "tensorboard_path": "./output/tensorboard/"
    })

    if not os.path.exists(config.output_path):
        os.mkdir(config.output_path)
    #snap 一种包的管理工具
    if not os.path.exists(config.snap_path):
        os.mkdir(config.snap_path)
    
    if not os.path.exists(config.tensorboard_path):
        os.mkdir(config.tensorboard_path)

    config.snap_path += config.model_name
    config.log_file = config.model_name + config.log_file
    config.tensorboard_path += config.model_name

    set_logging(config)
    logging.info(config)

    writer = SummaryWriter(config.tensorboard_path)

    # data load
    train_dataset = PIPAL21(
        dis_path=config.train_dis_path,
        txt_file_name=config.train_txt_file_name,
        transform=transforms.Compose(
            [
                #图片进行随机裁剪
                RandCrop(config.crop_size),
                #将图片标准化
                Normalize(0.5, 0.5),
                #图像一半的概率翻转，一半的概率不翻转
                RandHorizontalFlip(),
                #将shape为(H, W, C)的numpy.ndarray或img转为shape为(C, H, W)的tensor
                ToTensor()
            ]
        ),
    )
    val_dataset = PIPAL21(
        dis_path=config.val_dis_path,
        txt_file_name=config.val_txt_file_name,
        transform=transforms.Compose([Normalize(0.5, 0.5), ToTensor()]),
    )

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
    net = MANIQA(
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
    #均方损失函数
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
        logging.info('Running training epoch {}'.format(epoch + 1))
        loss_val, rho_s, rho_p = train_epoch(epoch, net, criterion, optimizer, scheduler, train_loader)

        writer.add_scalar("Train_loss", loss_val, epoch)
        writer.add_scalar("SRCC", rho_s, epoch)
        writer.add_scalar("PLCC", rho_p, epoch)

        if (epoch + 1) % config.val_freq == 0:
            logging.info('Starting eval...')
            logging.info('Running testing in epoch {}'.format(epoch + 1))
            loss, rho_s, rho_p = eval_epoch(config, epoch, net, criterion, val_loader)
            logging.info('Eval done...')

            if rho_s > best_srocc or rho_p > best_plcc:
                best_srocc = rho_s
                best_plcc = rho_p
                # save weights
                model_name = "epoch{}".format(epoch + 1)
                model_save_path = os.path.join(config.snap_path, model_name)
                torch.save(net, model_save_path)
                logging.info('Saving weights and model of epoch{}, SRCC:{}, PLCC:{}'.format(epoch + 1, best_srocc, best_plcc))
        
        logging.info('Epoch {} done. Time: {:.2}min'.format(epoch + 1, (time.time() - start_time) / 60))
        

            