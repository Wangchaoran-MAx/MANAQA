import torch,os
import numpy as np
from models.lafan import extract1,extract
from models.lafan import utils

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
'''
window=50
offset=20
f=30
#filename='E:/aiming1_subject1_5.bvh'
motion=['E:/final_text\motion_0','E:/final_text\motion_1','E:/final_text\motion_2','E:/final_text\motion_3','E:/final_text\motion_4','E:/final_text\motion_5']
bvh_folder='E:/test'
jitter_path='E:/final_text/jitter.txt'
train_set = ['subject1', 'subject2', 'subject3', 'subject4']
#anim=extract1.read_bvh(filename)
for i in range(0,6):
    xtrain, qtrain, parents, contacts_l, contacts_r = extract1.get_lafan1_set(motion[i], train_set,window=window,offset=offset)
    q_glbl, x_glbl = utils.quat_fk(qtrain, xtrain, parents)
#x:位置 q：四元数
    joint_p=torch.from_numpy(x_glbl)
    jkp = ((joint_p[3:] - 3 * joint_p[2:-1] + 3 * joint_p[1:-2] - joint_p[:-3]) * (f ** 3)).norm(dim=2)   # N, J
    jkp_mean=jkp.mean()/100/(f ** 3)
    jkp_std=jkp.std(dim=0).mean()/100/(f ** 3)
    print(jkp_mean,jkp_std)
    if(i==0):
        file=open(jitter_path,'w')
    else:
        file = open(jitter_path, 'a')
    file.write("失真第"+str(i)+"等级： "+'jkp_mean:'+str(jkp_mean)+'\t'+'jkp_std:'+str(jkp_std)+'\n')
    print("ok")
file.close()
'''

#def benchmark_interpolation(X, Q, x_mean, x_std, offsets, parents, out_path=None, ):

if __name__ == "__main__":
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

    for n_trans in trans_lengths:
        print('Computing errors for transition length = {}...'.format(n_trans))
        x_mean, x_std, offsets = extract.get_train_stats(bvh_folder, train_actors, 50, 20)
        X, Q, parents, contacts_l, contacts_r = extract.get_lafan1_set(bvh_folder, train_actors, window=65, offset=40)
        # Format the data for the current transition lengths. The number of samples and the offset stays unchanged.
        curr_window = n_trans + n_past + n_future
        curr_x = X[:, :curr_window, ...]
        curr_q = Q[:, :curr_window, ...]
        batchsize = curr_x.shape[0]

        # Ground-truth positions/quats/eulers
        gt_local_quats = curr_q
        gt_roots = curr_x[:, :, 0:1, :]
        gt_offsets = np.tile(offsets, [batchsize, curr_window, 1, 1])
        gt_local_poses = np.concatenate([gt_roots, gt_offsets], axis=2)
        trans_gt_local_poses = gt_local_poses[:, n_past: -n_future, ...]
        trans_gt_local_quats = gt_local_quats[:, n_past: -n_future, ...]
        # Local to global with Forward Kinematics (FK)
        trans_gt_global_quats, trans_gt_global_poses = utils.quat_fk(trans_gt_local_quats, trans_gt_local_poses, parents)
        trans_gt_global_poses = trans_gt_global_poses.reshape((trans_gt_global_poses.shape[0], -1, n_joints * 3)).transpose([0, 2, 1])
        # Normalize
        trans_gt_global_poses = (trans_gt_global_poses - x_mean) / x_std
        trans_myself_global_quats=[]
        trans_myself_global_poses=[]
        for i in range(0,6):
            x_meani, x_stdi, offsetsi = extract1.get_train_stats(motion[i], train_actors, 50, 20)
            Xi, Qi, parentsi, contacts_li, contacts_ri = extract1.get_lafan1_set(motion[i], train_actors, window=65,
                                                                            offset=40)
            # Format the data for the current transition lengths. The number of samples and the offset stays unchanged.
            curr_windowi = n_trans + n_past + n_future
            curr_xi = Xi[:, :curr_windowi, ...]
            curr_qi = Qi[:, :curr_windowi, ...]
            batchsizei = curr_xi.shape[0]

            # Ground-truth positions/quats/eulers
            gt_local_quatsi = curr_qi
            gt_rootsi = curr_xi[:, :, 0:1, :]
            gt_offsetsi = np.tile(offsetsi, [batchsizei, curr_windowi, 1, 1])
            gt_local_posesi = np.concatenate([gt_rootsi, gt_offsetsi], axis=2)
            trans_gt_local_posesi = gt_local_posesi[:, n_past: -n_future, ...]
            trans_gt_local_quatsi = gt_local_quatsi[:, n_past: -n_future, ...]
            # Local to global with Forward Kinematics (FK)
            trans_gt_global_quatsi, trans_gt_global_posesi = utils.quat_fk(trans_gt_local_quatsi, trans_gt_local_posesi,
                                                                         parentsi)
            trans_gt_global_posesi = trans_gt_global_posesi.reshape(
                (trans_gt_global_posesi.shape[0], -1, n_joints * 3)).transpose([0, 2, 1])
            # Normalize
            trans_gt_global_posesi = (trans_gt_global_posesi - x_meani) / x_stdi
            trans_myself_global_quats.append(trans_gt_global_quatsi)
            trans_myself_global_poses.append(trans_gt_global_posesi)
        '''
        # Zero-velocity pos/quats
        zerov_trans_local_quats, zerov_trans_local_poses = np.zeros_like(trans_gt_local_quats), np.zeros_like(trans_gt_local_poses)
        zerov_trans_local_quats[:, :, :, :] = gt_local_quats[:, n_past - 1:n_past, :, :]
        zerov_trans_local_poses[:, :, :, :] = gt_local_poses[:, n_past - 1:n_past, :, :]
        # To global
        trans_zerov_global_quats, trans_zerov_global_poses = utils.quat_fk(zerov_trans_local_quats, zerov_trans_local_poses, parents)
        trans_zerov_global_poses = trans_zerov_global_poses.reshape((trans_zerov_global_poses.shape[0], -1, n_joints * 3)).transpose([0, 2, 1])
        # Normalize
        trans_zerov_global_poses = (trans_zerov_global_poses - x_mean) / x_std

        # Interpolation pos/quats
        r, q = curr_x[:, :, 0:1], curr_q
        inter_root, inter_local_quats = utils.interpolate_local(r, q, n_past, n_future)
        trans_inter_root = inter_root[:, 1:-1, :, :]
        trans_inter_offsets = np.tile(offsets, [batchsize, n_trans, 1, 1])
        trans_inter_local_poses = np.concatenate([trans_inter_root, trans_inter_offsets], axis=2)
        inter_local_quats = inter_local_quats[:, 1:-1, :, :]
        # To global
        trans_interp_global_quats, trans_interp_global_poses = utils.quat_fk(inter_local_quats, trans_inter_local_poses, parents)
        trans_interp_global_poses = trans_interp_global_poses.reshape((trans_interp_global_poses.shape[0], -1, n_joints * 3)).transpose([0, 2, 1])
        # Normalize
        trans_interp_global_poses = (trans_interp_global_poses - x_mean) / x_std
        '''
         #Local quaternion loss
        res[('motion_0_quat_loss', n_trans)] = np.mean(np.sqrt(np.sum((trans_myself_global_quats[0] - trans_gt_global_quats) ** 2.0, axis=(2, 3))))
        res[('motion_1_quat_loss', n_trans)] = np.mean(np.sqrt(np.sum((trans_myself_global_quats[1] - trans_gt_global_quats) ** 2.0, axis=(2, 3))))
        res[('motion_2_quat_loss', n_trans)] = np.mean(np.sqrt(np.sum((trans_myself_global_quats[2] - trans_gt_global_quats) ** 2.0, axis=(2, 3))))
        res[('motion_3_quat_loss', n_trans)] = np.mean(np.sqrt(np.sum((trans_myself_global_quats[3] - trans_gt_global_quats) ** 2.0, axis=(2, 3))))
        res[('motion_4_quat_loss', n_trans)] = np.mean(np.sqrt(np.sum((trans_myself_global_quats[4] - trans_gt_global_quats) ** 2.0, axis=(2, 3))))
        res[('motion_5_quat_loss', n_trans)] = np.mean(np.sqrt(np.sum((trans_myself_global_quats[5] - trans_gt_global_quats) ** 2.0, axis=(2, 3))))


        # Global positions loss
        res[('motion_0_pos_loss', n_trans)] = np.mean(np.sqrt(np.sum((trans_myself_global_poses[0] - trans_gt_global_poses)**2.0, axis=1)))
        res[('motion_1_pos_loss', n_trans)] = np.mean(np.sqrt(np.sum((trans_myself_global_poses[1]- trans_gt_global_poses)**2.0, axis=1)))
        res[('motion_2_pos_loss', n_trans)] = np.mean(np.sqrt(np.sum((trans_myself_global_poses[2] - trans_gt_global_poses)**2.0, axis=1)))
        res[('motion_3_pos_loss', n_trans)] = np.mean(np.sqrt(np.sum((trans_myself_global_poses[3]- trans_gt_global_poses)**2.0, axis=1)))
        res[('motion_4_pos_loss', n_trans)] = np.mean(np.sqrt(np.sum((trans_myself_global_poses[4] - trans_gt_global_poses)**2.0, axis=1)))
        res[('motion_5_pos_loss', n_trans)] = np.mean(np.sqrt(np.sum((trans_myself_global_poses[5]- trans_gt_global_poses)**2.0, axis=1)))

        # NPSS loss on global quaternions
        res[('motion_0_npss_loss', n_trans)] = fast_npss(flatjoints(trans_gt_global_quats), flatjoints(trans_myself_global_quats[0]))
        res[('motion_1_npss_loss', n_trans)] = fast_npss(flatjoints(trans_gt_global_quats), flatjoints(trans_myself_global_quats[1]))
        res[('motion_2_npss_loss', n_trans)] = fast_npss(flatjoints(trans_gt_global_quats),flatjoints(trans_myself_global_quats[2]))
        res[('motion_3_npss_loss', n_trans)] = fast_npss(flatjoints(trans_gt_global_quats),flatjoints(trans_myself_global_quats[3]))
        res[('motion_4_npss_loss', n_trans)] = fast_npss(flatjoints(trans_gt_global_quats), flatjoints(trans_myself_global_quats[4]))
        res[('motion_5_npss_loss', n_trans)] = fast_npss(flatjoints(trans_gt_global_quats),flatjoints(trans_myself_global_quats[5]))


    print()
    motion0_quat_losses  = [res[('motion_0_quat_loss', n)] for n in trans_lengths]
    motion1_quat_losses = [res[('motion_1_quat_loss', n)] for n in trans_lengths]
    motion2_quat_losses = [res[('motion_2_quat_loss', n)] for n in trans_lengths]
    motion3_quat_losses = [res[('motion_3_quat_loss', n)] for n in trans_lengths]
    motion4_quat_losses = [res[('motion_4_quat_loss', n)] for n in trans_lengths]
    motion5_quat_losses = [res[('motion_5_quat_loss', n)] for n in trans_lengths]

    print("=== Global quat losses ===")
    print("{0: <16} | {1:6d}    | {2:6d}  ".format("Lengths", 25, 45))
    print("{0: <16} | {1:6.2f}  | {2:6.2f}".format("motion_0", *motion0_quat_losses))
    print("{0: <16} | {1:6.2f}  | {2:6.2f} ".format("motion_1", *motion1_quat_losses))
    print("{0: <16} | {1:6.2f}  | {2:6.2f}".format("motion_2", *motion2_quat_losses))
    print("{0: <16} | {1:6.2f}  | {2:6.2f}".format("motion_3", *motion3_quat_losses))
    print("{0: <16} | {1:6.2f}  | {2:6.2f}".format("motion_4", *motion4_quat_losses))
    print("{0: <16} | {1:6.2f}  | {2:6.2f}".format("motion_5", *motion5_quat_losses))
    print()

    motion0_pos_losses = [res[('motion_0_pos_loss', n)] for n in trans_lengths]
    motion1_pos_losses = [res[('motion_1_pos_loss', n)] for n in trans_lengths]
    motion2_pos_losses = [res[('motion_2_pos_loss', n)] for n in trans_lengths]
    motion3_pos_losses = [res[('motion_3_pos_loss', n)] for n in trans_lengths]
    motion4_pos_losses = [res[('motion_4_pos_loss', n)] for n in trans_lengths]
    motion5_pos_losses = [res[('motion_5_pos_loss', n)] for n in trans_lengths]

    print("=== Global pos losses ===")
    print("{0: <16} | {1:6d}    | {2:6d} ".format("Lengths", 25, 45))
    print("{0: <16} | {1:6.2f}  | {2:6.2f}".format("motion_0", *motion0_pos_losses))
    print("{0: <16} | {1:6.2f}  | {2:6.2f}".format("motion_1", *motion1_pos_losses))
    print("{0: <16} | {1:6.2f}  | {2:6.2f}".format("motion_2", *motion2_pos_losses))
    print("{0: <16} | {1:6.2f}  | {2:6.2f}".format("motion_3", *motion3_pos_losses))
    print("{0: <16} | {1:6.2f}  | {2:6.2f}".format("motion_4", *motion4_pos_losses))
    print("{0: <16} | {1:6.2f}  | {2:6.2f}".format("motion_5", *motion5_pos_losses))
    print()

    motion0_npss_losses = [res[('motion_0_npss_loss', n)] for n in trans_lengths]
    motion1_npss_losses = [res[('motion_1_npss_loss', n)] for n in trans_lengths]
    motion2_npss_losses = [res[('motion_2_npss_loss', n)] for n in trans_lengths]
    motion3_npss_losses = [res[('motion_3_npss_loss', n)] for n in trans_lengths]
    motion4_npss_losses = [res[('motion_4_npss_loss', n)] for n in trans_lengths]
    motion5_npss_losses = [res[('motion_5_npss_loss', n)] for n in trans_lengths]

    print("=== NPSS on global quats ===")
    print("{0: <16} | {1:6d}    | {2:6d}".format("Lengths", 25, 45))
    print("{0: <16} | {1:6.2f}  | {2:6.2f}".format("motion_0", *motion0_npss_losses))
    print("{0: <16} | {1:6.2f}  | {2:6.2f}".format("motion_1", *motion1_npss_losses))
    print("{0: <16} | {1:6.2f}  | {2:6.2f}".format("motion_2", *motion2_npss_losses))
    print("{0: <16} | {1:6.2f}  | {2:6.2f}".format("motion_3", *motion3_npss_losses))
    print("{0: <16} | {1:6.2f}  | {2:6.2f}".format("motion_4", *motion4_npss_losses))
    print("{0: <16} | {1:6.2f}  | {2:6.2f}".format("motion_5", *motion5_npss_losses))
    print()

    # Write to file is desired
    if out_path is not None:
        res_txt_file = open(os.path.join(out_path, 'h36m_transitions_benchmark.txt'), "a")
        res_txt_file.write("\n=== Global quat losses ===\n")
        res_txt_file.write("{0: <16} | {1:6d}   | {2:6d}  \n".format("Lengths", 25, 45))
        res_txt_file.write("{0: <16} | {1:6.6f} | {2:6.6f}\n".format("motion_0", *motion0_quat_losses))
        res_txt_file.write("{0: <16} | {1:6.6f} | {2:6.6f}\n".format("motion_1", *motion1_quat_losses))
        res_txt_file.write("{0: <16} | {1:6.6f} | {2:6.6f}\n".format("motion_2", *motion2_quat_losses))
        res_txt_file.write("{0: <16} | {1:6.6f} | {2:6.6f}\n".format("motion_3", *motion3_quat_losses))
        res_txt_file.write("{0: <16} | {1:6.6f} | {2:6.6f}\n".format("motion_4", *motion4_quat_losses))
        res_txt_file.write("{0: <16} | {1:6.6f} | {2:6.6f}\n".format("motion_5", *motion5_quat_losses))


        res_txt_file.write("\n\n")
        res_txt_file.write("=== Global pos losses ===\n")
        res_txt_file.write("{0: <16} | {1:6d}   | {2:6d}  \n".format("Lengths", 25, 45))
        res_txt_file.write("{0: <16} | {1:6.6f} | {2:6.6f}\n".format("motion_0", *motion0_pos_losses))
        res_txt_file.write("{0: <16} | {1:6.6f} | {2:6.6f} \n".format("motion_1", *motion1_pos_losses))
        res_txt_file.write("{0: <16} | {1:6.6f} | {2:6.6f} \n".format("motion_2", *motion2_pos_losses))
        res_txt_file.write("{0: <16} | {1:6.6f} | {2:6.6f} \n".format("motion_3", *motion3_pos_losses))
        res_txt_file.write("{0: <16} | {1:6.6f} | {2:6.6f} \n".format("motion_4", *motion4_pos_losses))
        res_txt_file.write("{0: <16} | {1:6.6f} | {2:6.6f} \n".format("motion_5", *motion5_pos_losses))

        res_txt_file.write("\n\n")
        res_txt_file.write("=== NPSS on global quats ===\n")

        res_txt_file.write("{0: <16} | {1:6d}   | {2:6d} \n".format("Lengths", 25, 45))
        res_txt_file.write("{0: <16} | {1:6.6f} | {2:6.6f}\n".format("motion_0", *motion0_npss_losses))
        res_txt_file.write("{0: <16} | {1:6.6f} | {2:6.6f}\n".format("motion_1", *motion1_npss_losses))
        res_txt_file.write("{0: <16} | {1:6.6f} | {2:6.6f} \n".format("motion_2", *motion2_npss_losses))
        res_txt_file.write("{0: <16} | {1:6.6f} | {2:6.6f} \n".format("motion_3", *motion3_npss_losses))
        res_txt_file.write("{0: <16} | {1:6.6f} | {2:6.6f} \n".format("motion_4", *motion4_npss_losses))
        res_txt_file.write("{0: <16} | {1:6.6f} | {2:6.6f} \n".format("motion_5", *motion5_npss_losses))

        res_txt_file.write("\n\n\n\n")
        res_txt_file.close()




