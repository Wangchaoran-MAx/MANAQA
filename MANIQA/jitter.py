import torch,os
import numpy as np
from models.lafan import extract_every1
from models.lafan import utils
from tqdm import tqdm

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

window=50
offset=20
f=30
#filename='E:/aiming1_subject1_5.bvh'
motion = ['/hy-tmp/final_test/motion_0/', '/hy-tmp/final_test/motion_1/', '/hy-tmp/final_test/motion_2/', '/hy-tmp/final_test/motion_3/',
              '/hy-tmp/final_test/motion_4/', '/hy-tmp/final_test/motion_5/']

jitter_path='/hy-tmp/final_test/jitter1.txt'
#train_set = ['subject1', 'subject2', 'subject3', 'subject4','subject5']
train_set = ['subject1', 'subject2', 'subject3']
val_set = ['subject4']
test_set = ['subject5']
#anim=extract1.read_bvh(filename)


write_file=open(jitter_path,'w')
write_file.write("数据名称        "+'jkp_mean'+' '+'jkp_std:'+'\n')

for i in range(0,6):
    files = os.listdir(motion[i])
    for file in tqdm(files):
        file_path=os.path.join(motion[i], file)
        xtrain, qtrain, parents = extract_every1.get_lafan1_set(file_path, train_set,window=window,offset=offset)
        q_glbl, x_glbl = utils.quat_fk(qtrain, xtrain, parents)
#x:位置 q：四元数
        joint_p=torch.from_numpy(x_glbl)
        print(np.shape(x_glbl))
        jkp = ((joint_p[3:] - 3 * joint_p[2:-1] + 3 * joint_p[1:-2] - joint_p[:-3]) * (f ** 3)).norm(dim=2)   # N, J
        jkp_mean=jkp.mean()/100/(f ** 3)
        jkp_std=jkp.std(dim=0).mean()/100/(f ** 3)
        #print(jkp_mean,jkp_std)
        a=np.round(jkp_mean.numpy(),5)
        b=np.round(jkp_std.numpy(),5)
        write_file=open(jitter_path,'a')
        write_file.write(file+' '+str(a)+' '+str(b)+'\n')
write_file.close()