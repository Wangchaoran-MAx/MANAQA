import torch,os,ntpath
import numpy as np
from models.lafan  import extract_every1,extract_every2,extract1,extract
from models.lafan  import utils
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

if __name__ == "__main__":
    bvh_folder = '/hy-tmp/lafan'
    motion = ['/hy-tmp/final_test/motion_0/', '/hy-tmp/final_test/motion_1/', '/hy-tmp/final_test/motion_2/', '/hy-tmp/final_test/motion_3/',
              '/hy-tmp/final_test/motion_4/', '/hy-tmp/final_test/motion_5/']
    #train_actors = ['subject1', 'subject2', 'subject3', 'subject4','subject5']#5ren
    train_set = ['subject1', 'subject2', 'subject3']
    val_set = ['subject4']
    test_set = ['subject5']
    trans_lengths = [50]
    n_joints = 22
    res = {}
    out_path='/hy-tmp/final_test/'
    if out_path is not None:
        #res_txt_file = open(os.path.join(out_path + '50three_2.txt'), "w")
        res_txt_file = open(os.path.join(out_path + 'train_three_all.txt'), "w")
        #res_txt_file.write( '三个值分别是 quat_loss、pos_loss、fast_npss1' + '\n')

    gt_global_poses=[]
    gt_global_quats=[]
    for n_trans in trans_lengths:
        for file in tqdm(os.listdir(bvh_folder)):
            name1 = file[:-4]
        X, Q, parents = extract.get_lafan1_set(bvh_folder, train_set, window=50, offset=20)
        #X, Q, parents = extract.get_lafan1_set(bvh_folder, val_set, window=50, offset=20)
        # X, Q, parents = extract.get_lafan1_set(bvh_folder, test_set, window=50, offset=20)
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
        for i in range(0,6):
            Xi, Qi, parentsi= extract1.get_lafan1_set(motion[i], train_set, window=50, offset=20)
            #Xi, Qi, parentsi = extract1.get_lafan1_set(motion[i], val_set, window=50, offset=20)
            #Xi, Qi, parentsi = extract1.get_lafan1_set(motion[i], test_set, window=50, offset=20)

            q_glbli, x_glbli = utils.quat_fk(Qi, Xi, parentsi)
            for j in range(x_glbli.shape[0]):
                x_meani = np.mean(x_glbli[i].reshape([x_glbli[j].shape[0], -1]).transpose([1, 0]), axis=(1),
                                 keepdims=True)
                x_stdi = np.std(x_glbli[i].reshape([x_glbli[j].shape[0], -1]).transpose([1, 0]), axis=(1),
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
        for j in range(0, 6):
            for i in range(len(gt_global_quats)):
                quat_loss=np.round(np.mean(np.sqrt(np.sum((myself_global_quats[j][i] - gt_global_quats[i]) ** 2.0, axis=1))),5)
                pos_loss=np.round(np.mean(np.sqrt(np.sum((myself_global_poses[j][i] - gt_global_poses[i]) ** 2.0, axis=1))),5)
                fast_npss1=np.round(fast_npss(flatjoints(gt_global_quats[i]), flatjoints(myself_global_quats[j][i])),5)
                res_txt_file = open(os.path.join(out_path + 'train_three_all.txt'), "a")
                res_txt_file.write( str(i*6+j) + ' ' + str(quat_loss) + ' ' + str(pos_loss) + ' ' + str(fast_npss1) + '\n')

        res_txt_file.close()


