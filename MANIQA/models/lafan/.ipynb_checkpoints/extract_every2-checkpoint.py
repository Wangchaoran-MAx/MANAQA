import re, os, ntpath
import numpy as np
from . import utils
from tqdm import tqdm

channelmap = {
    'Xrotation': 'x',
    'Yrotation': 'y',
    'Zrotation': 'z'
}

channelmap_inv = {
    'x': 'Xrotation',
    'y': 'Yrotation',
    'z': 'Zrotation',
}

ordermap = {
    'x': 0,
    'y': 1,
    'z': 2,
}


class Anim(object):
    """
    A very basic animation object
    """

    def __init__(self, quats, pos, offsets, parents, bones):
        """
        :param quats: local quaternions tensor四元张量
        :param pos: local positions tensor
        :param offsets: local joint offsets
        :param parents: bone hierarchy
        :param bones: bone names
        """
        self.quats = quats
        self.pos = pos
        self.offsets = offsets
        self.parents = parents
        self.bones = bones


def read_bvh(filename, start=None, end=None, order=None):
    """
    Reads a BVH file and extracts animation information.

    :param filename: BVh filename
    :param start: start frame
    :param end: end frame
    :param order: order of euler rotations
    :return: A simple Anim object conatining the extracted information.
    """

    f = open(filename, "r")

    i = 0
    active = -1
    end_site = False

    names = []
    # reshape 0是保持原来的维度不变
    orients = np.array([]).reshape((0, 4))
    offsets = np.array([]).reshape((0, 3))
    parents = np.array([], dtype=int)

    # Parse the  file, line by line
    for line in f:
        if line == '\n': continue
        if "HIERARCHY" in line: continue
        if "MOTION" in line: continue

        rmatch = re.match(r"ROOT (\w+)", line)
        if rmatch:
            names.append(rmatch.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients = np.append(orients, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            active = (len(parents) - 1)
            continue

        if "{" in line: continue

        if "}" in line:
            if end_site:
                end_site = False
            else:
                active = parents[active]
            continue

        offmatch = re.match(r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line)
        if offmatch:
            if not end_site:
                offsets[active] = np.array([list(map(float, offmatch.groups()))])
            continue

        chanmatch = re.match(r"\s*CHANNELS\s+(\d+)", line)
        if chanmatch:
            channels = int(chanmatch.group(1))
            if order is None:
                channelis = 0 if channels == 3 else 3
                channelie = 3 if channels == 3 else 6
                # 计算rotation
                parts = line.split()[2 + channelis:2 + channelie]
                # channelmap找不到
                if any([p not in channelmap for p in parts]):
                    continue
                order = "".join([channelmap[p] for p in parts])
            continue

        jmatch = re.match("\s*JOINT\s+(\w+)", line)
        if jmatch:
            names.append(jmatch.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients = np.append(orients, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            active = (len(parents) - 1)
            continue

        if "End Site" in line:
            end_site = True
            continue

        fmatch = re.match("\s*Frames:\s+(\d+)", line)
        if fmatch:
            if start and end:
                fnum = (end - start) - 1
            else:
                fnum = int(fmatch.group(1))
            # np.newaxis增加新的维度
            positions = offsets[np.newaxis].repeat(fnum, axis=0)
            rotations = np.zeros((fnum, len(orients), 3))
            continue

        fmatch = re.match("\s*Frame Time:\s+([\d\.]+)", line)
        if fmatch:
            frametime = float(fmatch.group(1))
            continue

        if (start and end) and (i < start or i >= end - 1):
            i += 1
            continue

        dmatch = line.strip().split(' ')
        if dmatch:
            data_block = np.array(list(map(float, dmatch)))
            N = len(parents)
            fi = i - start if start else i
            if channels == 3:
                positions[fi, 0:1] = data_block[0:3]
                rotations[fi, :] = data_block[3:].reshape(N, 3)
            elif channels == 6:
                data_block = data_block.reshape(N, 6)
                positions[fi, :] = data_block[:, 0:3]
                rotations[fi, :] = data_block[:, 3:6]
            elif channels == 9:
                positions[fi, 0] = data_block[0:3]
                data_block = data_block[3:].reshape(N - 1, 9)
                rotations[fi, 1:] = data_block[:, 3:6]
                positions[fi, 1:] += data_block[:, 0:3] * data_block[:, 6:9]
            else:
                raise Exception("Too many channels! %i" % channels)

            i += 1

    f.close()

    rotations = utils.euler_to_quat(np.radians(rotations), order=order)
    rotations = utils.remove_quat_discontinuities(rotations)

    return Anim(rotations, positions, offsets, parents, names)


def get_lafan1_set(file, actors, window=50, offset=20):
    """
    Extract the same test set as in the article, given the location of the BVH files.

    :param bvh_path: Path to the dataset BVH files
    :param list: actor prefixes to use in set
    :param window: width  of the sliding windows (in timesteps)
    :param offset: offset between windows (in timesteps)
    :return: tuple:
        X: local positions
        Q: local quaternions
        parents: list of parent indices defining the bone hierarchy
        contacts_l: binary tensor of left-foot contacts of shape (Batchsize, Timesteps, 2)
        contacts_r: binary tensor of right-foot contacts of shape (Batchsize, Timesteps, 2)
    """
    npast = 10
    subjects = []
    seq_names = []
    X = []
    Q = []
    contacts_l = []
    contacts_r = []

    # phases = []
    # all_phases = None

    # Extract


    # for index, file in tqdm(enumerate(bvh_files)):

    if file.endswith('.bvh'):
        seq_name, subject = ntpath.basename(file[:-4]).split('_')

            # find phase file
            # phase_file = "Input" + seq_name + "_" + subject + ".txt"
            # phase = np.loadtxt(os.path.join(bvh_path, phase_file))
            # phase = np.insert(phase, -1, values=[0, 0, 0, 0], axis=0)
            # if index == 0:
            #     all_phases = phase
            # else:
            #     all_phases = np.vstack((all_phases, phase))

        if subject in actors:

            anim = read_bvh(file)

                # Sliding windows
            i = 0
            while i + window < anim.pos.shape[0]:
                q, x = utils.quat_fk(anim.quats[i: i + window], anim.pos[i: i + window], anim.parents)
                c_l, c_r = utils.extract_feet_contacts(x, [3, 4], [7, 8], velfactor=0.2)
                X.append(anim.pos[i: i + window])
                Q.append(anim.quats[i: i + window])



                 # Extract phase
                 # phases.append(phase[i: i + window])

                i += offset

    # phase_std = np.std(all_phases, axis=0)
    # phase_mn = np.mean(all_phases, axis=0)
    # phases = np.asarray(phases)

    X = np.asarray(X)
    Q = np.asarray(Q)


    # Sequences around XZ = 0
    xzs = np.mean(X[:, :, 0, ::2], axis=1, keepdims=True)
    X[:, :, 0, 0] = X[:, :, 0, 0] - xzs[..., 0]
    X[:, :, 0, 2] = X[:, :, 0, 2] - xzs[..., 1]

    # Unify facing on last seed frame
    X, Q = utils.rotate_at_frame(X, Q, anim.parents, n_past=npast)

    return X, Q, anim.parents, \
        # phases, phase_std, phase_mn


def get_train_stats(bvh_folder, train_set, window, offset):
    """
    Extract the same training set as in the paper in order to compute the normalizing statistics
    :return: Tuple of (local position mean vector, local position standard deviation vector, local joint offsets tensor)
    """
    # print('Building the train set...')
    xtrain, qtrain, parents = get_lafan1_set(bvh_folder, train_set,
                                                                     window=window,
                                                                     offset=offset)

    print('Computing stats...\n')
    # Joint offsets : are constant, so just take the first frame:
    offsets = xtrain[0:1, 0:1, :, :]  # Shape : (1, 1, J, 3)

    # Global representation:
    q_glbl, x_glbl = utils.quat_fk(qtrain, xtrain, parents)

    # Global positions stats:
    x_mean = np.mean(x_glbl.reshape([x_glbl.shape[0], x_glbl.shape[1], -1]).transpose([0, 2, 1]), axis=(0, 2),
                     keepdims=True)
    x_std = np.std(x_glbl.reshape([x_glbl.shape[0], x_glbl.shape[1], -1]).transpose([0, 2, 1]), axis=(0, 2),
                   keepdims=True)

    # velocity = torch.cat([x_glbl[:, :, 1:] - x_glbl[:, :, :-1],
    #                       torch.zeros_like(x_glbl[:, :, 0:1])], -1)
    # v_std = np.std(velocity.reshape([velocity.shape[0], velocity.shape[1], -1]).transpose([0, 2, 1]), axis=(0, 2),
    #                keepdims=True)

    return x_mean, x_std, offsets

    # return offsets, xtrain, qtrain, parents, contacts_l, contacts_r


if __name__ == "__main__":
    X, Q, parents, contacts_l, contacts_r = get_lafan1_set('D:\python\data\lafan1',
                                                           ["subject1", "subject2", "subject3", "subject4"], window=50,
                                                           offset=20)

    print(X.shape)
    print(Q.shape)
    print(parents.shape)
    print(contacts_l.shape)
    print(contacts_r.shape)