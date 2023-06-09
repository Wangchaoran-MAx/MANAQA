import sys, os
#sys.path.insert(0, os.path.dirname(__file__))
import pickle as pkl
import extract1, utils, benchmarks


"""
Unzips the data, extracts the LaFAN1 train statistics,
then the LaFAN1 test set and evaluates baselines on the test set.
"""

# output location for unzipped bvhs, stats, results, etc...
#out_path = os.path.join(os.path.dirname(__file__), 'output')
out_path='/hy-tmp/MANIQA/output/lafan_evaluate'
# the train/test set actors as in the paper
train_actors = ['subject1', 'subject2', 'subject3', 'subject4']
test_actors = ['subject5']




bvh_folder = '/hy-tmp/all_motion'
print('Retrieving statistics...')
stats_file = os.path.join(out_path, 'train_stats.pkl')
if not os.path.exists(stats_file):
    x_mean, x_std, offsets = extract1.get_train_stats(bvh_folder, train_actors, 50, 20)
    with open(stats_file, 'wb') as f:
        pkl.dump({
            'x_mean': x_mean,
            'x_std': x_std,
            'offsets': offsets,
        }, f, protocol=pkl.HIGHEST_PROTOCOL)
else:
    print('  Reusing stats file: ' + stats_file)
    with open(stats_file, 'rb') as f:
        stats = pkl.load(f)
    x_mean = stats['x_mean']
    x_std = stats['x_std']
    offsets = stats['offsets']


# Get test-set for windows of 65 frames, offset by 40 frames
print('\nBuilding the test set...')
X, Q, parents, contacts_l, contacts_r = extract1.get_lafan1_set(bvh_folder, test_actors, window=65, offset=40)
print('  Nb of sequences : {}\n'.format(X.shape[0]))

results = benchmarks.benchmark_interpolation(X, Q, x_mean, x_std, offsets, parents, out_path=out_path)

# save the results for validation if desired
with open(os.path.join(out_path, 'results.pkl'), 'wb') as f:
    pkl.dump(results, f, protocol=pkl.HIGHEST_PROTOCOL)


print('Done.')