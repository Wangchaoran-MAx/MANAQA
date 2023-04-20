import os,cv2
import torch
import torch.nn as nn
import numpy as np
from models.lafan import extract_motion


# 一个数据加载器，这个数据加载器每次可以返回一个 Batch 的数据供模型训练使用。
class Lafan1(torch.utils.data.Dataset):
    def __init__(self,actors, txt_file_name, transform):
        super(Lafan1, self).__init__()
        self.actors=actors
        self.txt_file_name = txt_file_name
        self.transform = transform
        self.res=[]
        
        #motion = ['/hy-tmp/final_test/motion_0/', '/hy-tmp/final_test/motion_1/', '/hy-tmp/final_test/motion_2/','/hy-tmp/final_test/motion_3/','/hy-tmp/final_test/motion_4/', '/hy-tmp/final_test/motion_5/']
        motion = ['/hy-tmp/final_test/motion_try_0/','/hy-tmp/final_test/motion_try_4/','/hy-tmp/final_test/motion_try_6/']
        dis_files_data, score_data,res = [], [],[]
        with open(self.txt_file_name, 'r') as listFile:
            for line in listFile:
                # 每一行字符串进行切片，默认为空格
                dis, score = line.split()
                # [start:end:step]
                dis = dis[:]  
                score = float(score)
                dis_files_data.append(dis)
                score_data.append(score)

        # reshape score_list (1xn -> nx1)
        score_data = np.array(score_data)
        #score_data = self.normalization(score_data)
        score_data = score_data.astype('float').reshape(-1, 1)
        # dis_files_data：图片数据的名称 score_list：图片分数的名称
        self.data_dict = {'d_motion_list': dis_files_data, 'score_list': score_data}
        self.res,Q = extract_motion.get_lafan1_set(motion, self.actors, window=50, offset=20)
        
        #print("self.res",np.shape(self.res))
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
    

        d_motion_name = int(self.data_dict['d_motion_list'][idx])
        d_motion = self.res[d_motion_name]
       
        #print("d_motion",np.shape(d_motion))
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