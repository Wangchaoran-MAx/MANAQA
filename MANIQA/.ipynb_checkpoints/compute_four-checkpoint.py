import torch,os,ntpath
import numpy as np
from models.lafan import extract_every1,extract_every2,extract1,extract
from models.lafan import utils
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 假设有一组BVH数据，每个数据包含10个骨骼节点，每个节点有三个坐标值


def fast_npss(gt_seq, pred_seq):
    """
    Computes Normalized Power Spectrum Similarity (NPSS).
    计算归一化功率谱相似度
    This is the metric proposed by Gropalakrishnan et al (2019).
    This implementation uses numpy parallelism for improved performance.

    :param gt_seq: ground-truth array of shape : (Batchsize, Timesteps, Dimension)
    :param pred_seq: shape : (Batchsize, Timesteps, Dimension)
    :return: The average npss metric for the batch
    """
    # Fourier coefficients along the time dimension
    gt_fourier_coeffs = np.real(np.fft.fft(gt_seq, axis=1))
    pred_fourier_coeffs = np.real(np.fft.fft(pred_seq, axis=1))

    # Square of the Fourier coefficients
    gt_power = np.square(gt_fourier_coeffs)
    pred_power = np.square(pred_fourier_coeffs)

    # Sum of powers over time dimension
    gt_total_power = np.sum(gt_power, axis=1)
    pred_total_power = np.sum(pred_power, axis=1)

    # Normalize powers with totals
    gt_norm_power = gt_power / gt_total_power[:, np.newaxis, :]
    pred_norm_power = pred_power / pred_total_power[:, np.newaxis, :]

    # Cumulative sum over time
    cdf_gt_power = np.cumsum(gt_norm_power, axis=1)
    cdf_pred_power = np.cumsum(pred_norm_power, axis=1)

    # Earth mover distance
    emd = np.linalg.norm((cdf_pred_power - cdf_gt_power), ord=1, axis=1)

    # Weighted EMD
    power_weighted_emd = np.average(emd, weights=gt_total_power)

    return power_weighted_emd
def flatjoints(x):
    """
    Shorthand for a common reshape pattern. Collapses all but the two first dimensions of a tensor.
    :param x: Data tensor of at least 3 dimensions.
    :return: The flattened tensor.
    """
    return x.reshape((x.shape[0], x.shape[1], -1))

#关于jitter的一些计算
'''
window=50
offset=20
f=30
#filename='E:/aiming1_subject1_5.bvh'
motion = ['/hy-tmp/final_test/motion_0/', '/hy-tmp/final_test/motion_1/', '/hy-tmp/final_test/motion_2/', '/hy-tmp/final_test/motion_3/',
              '/hy-tmp/final_test/motion_4/', '/hy-tmp/final_test/motion_5/']
bvh_folder='/hy-tmp/all_motion/'
jitter_path='/hy-tmp/final_test/jitter_all.txt'
#jitter_path='E:/jitter1.txt'
train_set = ['subject1', 'subject2', 'subject3', 'subject4','subject5']
#anim=extract1.read_bvh(filename)


write_file=open(jitter_path,'w')
#write_file.write("数据名称        "+'jkp_mean'+' '+'jkp_std:'+'\n')

for i in range(0,6):
    files = os.listdir(motion[i])
    for file in tqdm(files):
        file_path=os.path.join(motion[i], file)
        xtrain, qtrain, parents = extract_every1.get_lafan1_set(file_path, train_set,window=window,offset=offset)
        q_glbl, x_glbl = utils.quat_fk(qtrain, xtrain, parents)
#x:位置 q：四元数
        joint_p=torch.from_numpy(x_glbl)
        jkp = ((joint_p[3:] - 3 * joint_p[2:-1] + 3 * joint_p[1:-2] - joint_p[:-3]) * (f ** 3)).norm(dim=2)   # N, J
        jkp_mean=jkp.mean()/100/(f ** 3)
        jkp_std=jkp.std(dim=0).mean()/100/(f ** 3)
        #print(jkp_mean,jkp_std)
        a=np.round(jkp_mean.numpy(),5)
        b=np.round(jkp_std.numpy(),5)
        write_file=open(jitter_path,'a')
        write_file.write(file+' '+str(a)+' '+str(b)+'\n')
write_file.close()
'''
'''

for file in tqdm(os.listdir(bvh_folder)):
    file_path=os.path.join(bvh_folder, file)
    #print(file_path)
    xtrain, qtrain, parents = extract_every1.get_lafan1_set(file_path, train_set,window=window,offset=offset)
    q_glbl, x_glbl = utils.quat_fk(qtrain, xtrain, parents)
    #print(x_glbl)
    print(np.shape(x_glbl))
#x:位置 q：四元数
    joint_p=torch.from_numpy(x_glbl)
    print(np.shape(joint_p))
    jkp = ((joint_p[:,3:] - 3 * joint_p[:,2:-1] + 3 * joint_p[:,1:-2] - joint_p[:,:-3]) * (f ** 3)).norm(dim=2)  # N, J
    #print(joint_p[3:] ,joint_p[2:-1],joint_p[1:-2],joint_p[:-3])

    jkp_mean=jkp.mean()/100/(f ** 3)
    jkp_std=jkp.std(dim=0).mean()/100/(f ** 3)
    #print(jkp.mean(),jkp.std())
    a=np.round(jkp_mean.numpy(),5)
    b=np.round(jkp_std.numpy(),5)

    write_file=open(jitter_path,'a')
    write_file.write(file+' '+str(a)+'\n')

write_file.close()

'''
#最新方法计算jitter

window=50
offset=20
f=30
#filename='E:/aiming1_subject1_5.bvh'
#motion = ['/hy-tmp/final_test/motion_0/', '/hy-tmp/final_test/motion_1/', '/hy-tmp/final_test/motion_2/', '/hy-tmp/final_test/motion_3/','/hy-tmp/final_test/motion_4/', '/hy-tmp/final_test/motion_5/']
motion = ['/hy-tmp/final_test/motion_try_0/','/hy-tmp/final_test/motion_try_1/','/hy-tmp/final_test/motion_try_2/','/hy-tmp/final_test/motion_try_3/','/hy-tmp/final_test/motion_try_4/','/hy-tmp/final_test/motion_try_5/']
bvh_folder='/hy-tmp/all_motion/'
jitter_path='/hy-tmp/final_test/try_train_jitter.txt'
#jitter_path='E:/jitter1.txt'
train_set = ['subject1', 'subject2', 'subject3']
val_set = ['subject4']
test_set = ['subject5']
try_train_set = ['subject1']
try_val_set = ['subject2']
#anim=extract1.read_bvh(filename)


write_file=open(jitter_path,'w')
for j in range(0,len(motion)):
    xtrain, qtrain, parents = extract1.get_lafan1_set(motion[j], try_train_set,window=window,offset=offset)
    q_glbl, x_glbl = utils.quat_fk(qtrain, xtrain, parents)
#print(x_glbl)
    print(np.shape(x_glbl))
#x:位置 q：四元数
    for i in range(x_glbl.shape[0]):
        joint_p=torch.from_numpy(x_glbl[i])
    #print(np.shape(joint_p))
        jkp = ((joint_p[3:] - 3 * joint_p[2:-1] + 3 * joint_p[1:-2] - joint_p[:-3]) * (f ** 3)).norm(dim=2)  # N, J
    #print(joint_p[3:] ,joint_p[2:-1],joint_p[1:-2],joint_p[:-3])

        jkp_mean=jkp.mean()/100/(f ** 3)
    #jkp_std=jkp.std(dim=0).mean()/100/(f ** 3)
    #print(jkp.mean(),jkp.std())
        a=np.round(jkp_mean.numpy(),7)
    #b=np.round(jkp_std.numpy(),5)

        write_file=open(jitter_path,'a')
        write_file.write(str(j*(x_glbl.shape[0])+i)+' '+str(a)+'\n')

write_file.close()

#在服务器上的一些参数


'''
    bvh_folder = '/hy-tmp/lafan'
    motion = ['/hy-tmp/final_test/motion_0/', '/hy-tmp/final_test/motion_1/', '/hy-tmp/final_test/motion_2/', '/hy-tmp/final_test/motion_3/',
              '/hy-tmp/final_test/motion_4/', '/hy-tmp/final_test/motion_5/']
    train_actors = ['subject1', 'subject2','subject3','subject4','subject5']
    test_actors = ['subject3']
    n_past=10
    n_future=10
    trans_lengths = [25,45]
    n_joints = 22
    res = {}
    out_path='/hy-tmp/final_test'
'''

#其余所有指标


'''
if __name__ == "__main__":
    bvh_folder = 'E:\原本打分BVH'
    motion = ['E:/final_text\motion_0/', 'E:/final_text\motion_1/', 'E:/final_text\motion_2/', 'E:/final_text\motion_3/',
              'E:/final_text\motion_4/', 'E:/final_text\motion_5/']
    train_actors = ['subject1', 'subject2', 'subject3', 'subject4','subject5']#5ren
    trans_lengths = [50]
    n_joints = 22
    res = {}
    out_path='E:/'
    if out_path is not None:
        #res_txt_file = open(os.path.join(out_path + '50three_2.txt'), "w")
        res_txt_file = open(os.path.join(out_path + '50three_2.txt'), "w")
        res_txt_file.write( '三个值分别是 quat_loss、pos_loss、fast_npss1' + '\n')


    for n_trans in trans_lengths:
        print('Computing errors for transition length = {}...'.format(n_trans))
        for file in tqdm(os.listdir(bvh_folder)):
            name=file[:-4]
        #seq_name, subject = ntpath.basename(file[:-4]).split('_')

            file_path = os.path.join(bvh_folder, file)
            print(file_path)
            x_mean, x_std, offsets = extract_every2.get_train_stats(file_path, train_actors, 50, 20)
            X, Q, parents = extract_every2.get_lafan1_set(file_path, train_actors, window=50, offset=20)
        # Format the data for the current transition lengths. The number of samples and the offset stays unchanged.
            curr_window = n_trans
            curr_x = X[:, :curr_window, ...]
            curr_q = Q[:, :curr_window, ...]
            batchsize = curr_x.shape[0]

        # Ground-truth positions/quats/eulers
            gt_local_quats = curr_q
            gt_roots = curr_x[:, :, 0:1, :]
            gt_offsets = np.tile(offsets, [batchsize, curr_window, 1, 1])
            gt_local_poses = np.concatenate([gt_roots, gt_offsets], axis=2)
            trans_gt_local_poses = gt_local_poses
            trans_gt_local_quats = gt_local_quats
        # Local to global with Forward Kinematics (FK)
            trans_gt_global_quats, trans_gt_global_poses = utils.quat_fk(trans_gt_local_quats, trans_gt_local_poses, parents)
            trans_gt_global_poses = trans_gt_global_poses.reshape((trans_gt_global_poses.shape[0], -1, n_joints * 3)).transpose([0, 2, 1])
        # Normalize
            trans_gt_global_poses = (trans_gt_global_poses - x_mean) / x_std
            trans_myself_global_quats=[]
            trans_myself_global_poses=[]
            for i in range(0,6):
                for file in tqdm(os.listdir(motion[i])):
                    name1=file[:-6]
                    if(name1==name):
                        name_temp=file
                file_path = os.path.join(motion[i], name_temp)
                x_meani, x_stdi, offsetsi = extract_every1.get_train_stats(file_path, train_actors, 50, 20)
                Xi, Qi, parentsi= extract_every1.get_lafan1_set(file_path, train_actors, window=50,
                                                                            offset=20)
            # Format the data for the current transition lengths. The number of samples and the offset stays unchanged.
                curr_windowi = n_trans
                curr_xi = Xi[:, :curr_windowi, ...]
                curr_qi = Qi[:, :curr_windowi, ...]
                batchsizei = curr_xi.shape[0]

            # Ground-truth positions/quats/eulers
                gt_local_quatsi = curr_qi
                gt_rootsi = curr_xi[:, :, 0:1, :]
                gt_offsetsi = np.tile(offsetsi, [batchsizei, curr_windowi, 1, 1])
                gt_local_posesi = np.concatenate([gt_rootsi, gt_offsetsi], axis=2)
                trans_gt_local_posesi = gt_local_posesi
                trans_gt_local_quatsi = gt_local_quatsi
            # Local to global with Forward Kinematics (FK)
                trans_gt_global_quatsi, trans_gt_global_posesi = utils.quat_fk(trans_gt_local_quatsi, trans_gt_local_posesi,
                                                                         parentsi)
                trans_gt_global_posesi = trans_gt_global_posesi.reshape(
                    (trans_gt_global_posesi.shape[0], -1, n_joints * 3)).transpose([0, 2, 1])
            # Normalize
                trans_gt_global_posesi = (trans_gt_global_posesi - x_meani) / x_stdi
                # trans_myself_global_quats.append(trans_gt_global_quatsi)
                #trans_myself_global_poses.append(trans_gt_global_posesi)
                quat_loss=np.round(np.mean(np.sqrt(np.sum((trans_gt_global_quatsi - trans_gt_global_quats) ** 2.0, axis=(2, 3)))),5)
                pos_loss=np.round(np.mean(np.sqrt(np.sum((trans_gt_global_posesi - trans_gt_global_poses) ** 2.0, axis=1))),5)
                fast_npss1=np.round(fast_npss(flatjoints(trans_gt_global_quats), flatjoints(trans_gt_global_quatsi)),5)


                if out_path is not None:
                   # res_txt_file = open(os.path.join(out_path, str(n_trans) + 'three.txt'), "a")
                    res_txt_file = open(os.path.join(out_path, str(n_trans)+'three_2.txt'), "a")
                    res_txt_file.write(name_temp+' '+str(quat_loss)+' '+str(pos_loss)+' '+str(fast_npss1)+'\n')
                    res_txt_file.close()
'''

'''
if __name__ == "__main__":
    bvh_folder = '/hy-tmp/lafan'
    #motion = ['/hy-tmp/final_test/motion_0/', '/hy-tmp/final_test/motion_1/', '/hy-tmp/final_test/motion_2/', '/hy-tmp/final_test/motion_3/','/hy-tmp/final_test/motion_4/', '/hy-tmp/final_test/motion_5/']
    try_bvh='/hy-tmp/final_test/BVH_try/'
    motion = ['/hy-tmp/final_test/motion_try_0/','/hy-tmp/final_test/motion_try_1/','/hy-tmp/final_test/motion_try_2/','/hy-tmp/final_test/motion_try_3/','/hy-tmp/final_test/motion_try_4/','/hy-tmp/final_test/motion_try_5/']
    train_set = ['subject1', 'subject2', 'subject3']
    val_set = ['subject4']
    test_set = ['subject5']
    try_train_set = ['subject1']
    try_val_set = ['subject2']
    trans_lengths = [50]
    n_joints = 22
    res = {}
    out_path='/hy-tmp/final_test/'
    
        #res_txt_file = open(os.path.join(out_path + '50three_2.txt'), "w")
    
        #res_txt_file.write( '三个值分别是 quat_loss、pos_loss、fast_npss1' + '\n')
    res_txt_file = open(os.path.join(out_path + 'try_val_three.txt'), "w")
    gt_global_poses=[]
    gt_global_quats=[]
    for n_trans in trans_lengths:

        X, Q, parents = extract.get_lafan1_set(try_bvh, try_val_set, window=50, offset=20)

        q_glbl, x_glbl = utils.quat_fk(Q, X, parents)
        for i in range (x_glbl.shape[0]):
            x_mean = np.mean(x_glbl[i].reshape([x_glbl[i].shape[0], -1]).transpose([1, 0]), axis=(1),
                         keepdims=True)
            x_std = np.std(x_glbl[i].reshape( [x_glbl[i].shape[0], -1]).transpose([1,0]), axis=(1),
                       keepdims=True)

            trans_gt_global_quats=q_glbl[i]
            trans_gt_global_poses=x_glbl[i]
        # Local to global with Forward Kinematics (FK)
            trans_gt_global_poses = trans_gt_global_poses.reshape(( -1, n_joints * 3)).transpose([1,0])
            #print(np.shape(trans_gt_global_poses), np.shape(x_mean))
        # Normalize
            trans_gt_global_poses = (trans_gt_global_poses - x_mean) / x_std
            gt_global_poses.append(trans_gt_global_poses)
            gt_global_quats.append(trans_gt_global_quats)

        myself_global_quats=[[],[],[],[],[],[]]
        myself_global_poses=[[],[],[],[],[],[]]
        for i in range(0,len(motion)):
            print(i)
            Xi, Qi, parentsi= extract1.get_lafan1_set(motion[i], try_val_set, window=50, offset=20)
            q_glbli, x_glbli = utils.quat_fk(Qi, Xi, parentsi)
            for j in range(x_glbli.shape[0]):
                x_meani = np.mean(x_glbli[j].reshape([x_glbli[j].shape[0], -1]).transpose([1, 0]), axis=(1),
                                 keepdims=True)
                x_stdi = np.std(x_glbli[j].reshape([x_glbli[j].shape[0], -1]).transpose([1, 0]), axis=(1),
                               keepdims=True)

                trans_gt_global_quatsi = q_glbli[j]
                trans_gt_global_posesi = x_glbli[j]
                # Local to global with Forward Kinematics (FK)
                trans_gt_global_posesi = trans_gt_global_posesi.reshape((-1, n_joints * 3)).transpose([1, 0])
                #print(np.shape(trans_gt_global_poses), np.shape(x_mean))
                # Normalize
                trans_gt_global_posesi = (trans_gt_global_posesi - x_meani) / x_stdi
                myself_global_poses[i].append(trans_gt_global_posesi)
                myself_global_quats[i].append(trans_gt_global_quatsi)
           # print(np.shape(myself_global_poses[i]))
       
        for j in range(0, len(motion)):
            print(i)
            for i in range(len(gt_global_quats)):
            
                quat_loss=np.round(np.mean(np.sqrt(np.sum((myself_global_quats[j][i] - gt_global_quats[i]) ** 2.0, axis=1))),5)
                pos_loss=np.round(np.mean(np.sqrt(np.sum((myself_global_poses[j][i] - gt_global_poses[i]) ** 2.0, axis=1))),5)
                fast_npss1=np.round(fast_npss(flatjoints(gt_global_quats[i]), flatjoints(myself_global_quats[j][i])),5)
                res_txt_file = open(os.path.join(out_path + 'try_val_three.txt'), "a")
                res_txt_file.write( str(j*len(gt_global_quats)+i) + ' ' + str(quat_loss) + ' ' + str(pos_loss) + ' ' + str(fast_npss1) + '\n')
            
        res_txt_file.close()
'''