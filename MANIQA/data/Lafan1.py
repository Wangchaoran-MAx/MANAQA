import os,cv2
import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf
from models.lafan import extract_motion
class FrameEncoder(nn.Module):
    def __init__(self, input_dim, block_num):
        super(FrameEncoder, self).__init__()
        self.block_num = block_num
        self.conv1 = nn.Conv1d(input_dim, input_dim, kernel_size=1, stride=1)

        # encoder block
        self.enc_pi1 = nn.Conv1d(input_dim, input_dim // 2, kernel_size=1, stride=1)
        self.enc_pi2 = nn.Conv1d(input_dim // 2, input_dim, kernel_size=1, stride=1)
        self.enc_tao = nn.Conv1d(input_dim, input_dim, kernel_size=1, stride=1)

        self.conv2 = nn.Conv1d(input_dim, input_dim, kernel_size=1, stride=1)
        self.conv_z_attention = nn.Conv1d(input_dim, input_dim, kernel_size=1, stride=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.conv1(x)

        for i in range(self.block_num):
            pi = self.relu(h)
            pi = self.enc_pi1(pi)
            pi = self.relu(pi)
            pi = self.enc_pi2(pi)

            tau = self.enc_tao(h)
            tau = self.sigmoid(tau)
            h = h * (1 - tau) + pi * tau

        z = self.conv2(h)

        z_attention = self.conv_z_attention(h)
        z_attention = self.sigmoid(z_attention)

        z = torch.multiply(z, z_attention)

        return z

# 一个数据加载器，这个数据加载器每次可以返回一个 Batch 的数据供模型训练使用。
class Lafan1(torch.utils.data.Dataset):
    def __init__(self,actors, txt_file_name, transform):
        super(Lafan1, self).__init__()
        self.actors=actors
        self.txt_file_name = txt_file_name
        self.transform = transform
        self.res=[]
        
        #motion = ['/hy-tmp/final_test/motion_0/', '/hy-tmp/final_test/motion_1/', '/hy-tmp/final_test/motion_2/','/hy-tmp/final_test/motion_3/','/hy-tmp/final_test/motion_4/', '/hy-tmp/final_test/motion_5/']
        motion = ['/hy-tmp/final_test/motion_try_0/','/hy-tmp/final_test/motion_try_1/','/hy-tmp/final_test/motion_try_2/','/hy-tmp/final_test/motion_try_3/','/hy-tmp/final_test/motion_try_4/','/hy-tmp/final_test/motion_try_5/']
        dis_files_data, score_data,res = [], [],[]
        with open(self.txt_file_name, 'r') as listFile:
            for line in listFile:
                # 每一行字符串进行切片，默认为空格
                dis, score = line.split()
                # [start:end:step]
                dis = dis[:]  # 可能需要更改
                score = float(score)
                dis_files_data.append(dis)
                score_data.append(score)

        # reshape score_list (1xn -> nx1)
        score_data = np.array(score_data)
        #score_data = self.normalization(score_data)
        score_data = score_data.astype('float').reshape(-1, 1)
        # dis_files_data：图片数据的名称 score_list：图片分数的名称
        self.data_dict = {'d_motion_list': dis_files_data, 'score_list': score_data}
        X,self.res = extract_motion.get_lafan1_set(motion, self.actors, window=50, offset=20)
        self.res=self.res.reshape(self.res.shape[0],self.res.shape[1],-1)
        #print(np.shape(self.res))
        self.res=torch.from_numpy(self.res).to(torch.float32)
        enc = FrameEncoder(input_dim=50, block_num=3)
        self.res = enc(self.res)

    def normalization(self, data):
        range = np.max(data) - np.min(data)
        return (data - np.min(data)) / range

    
    
    def __len__(self):
        return len(self.data_dict['d_motion_list'])

    def __getitem__(self, idx):
        '''
        d_img_name = self.data_dict['d_img_list'][idx]
        d_img = cv2.imread(os.path.join(self.dis_path, d_img_name), cv2.IMREAD_COLOR)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype('float32') / 255
        d_img = np.transpose(d_img, (2, 0, 1))
        '''
        sum_out1 = []
        tensor = []
        inputs = []
# sum_out=np.random.rand(3,224, 224)
        d_motion_name = int(self.data_dict['d_motion_list'][idx])
        d_motion = self.res[d_motion_name]
        sum_out1 = d_motion.detach().numpy()
        sum_out1= np.expand_dims(sum_out1, 0)
        sum_out1 = np.concatenate([sum_out1, sum_out1, sum_out1], axis=0)
        sum_out1= np.expand_dims(sum_out1, 0)
        #print(np.shape(sum_out1))
        conv_0=torch.nn.Conv2d(in_channels=3,out_channels=3,kernel_size=(1,39),stride=1,padding=1)
        conv_1=torch.nn.Conv2d(in_channels=3,out_channels=3,kernel_size=5,stride=1,padding=4)
        inputs = torch.from_numpy(sum_out1).to(torch.float32)
        #print(np.shape(inputs))
        inputs = conv_0(inputs)
        for i in range((224-52)//(4*2+1-5)):
            inputs = conv_1(inputs)
        inputs= inputs.detach().numpy()
        #print(np.shape(inputs))
        d_motion = inputs[0]

        #print(np.shape(d_motion))

        score = self.data_dict['score_list'][idx]
        # print(np.shape(sum_out))
        sample = {
            'd_motion_org': d_motion,
            'score': score
        }

        if self.transform:
            sample = self.transform(sample)
        return sample