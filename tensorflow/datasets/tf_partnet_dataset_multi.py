import os
import sys
import time
import pickle
import h5py
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data', 'PartNet')

if not os.path.exists(DATA_DIR):
    raise IOError(f"{DATA_DIR} not found!")

from utils.ply import read_ply, write_ply
from .custom_dataset import CustomDataset, grid_subsampling, tf_batch_subsampling, tf_batch_neighbors


class PartNetDataset(CustomDataset):
    def __init__(self, config, input_threads=8, noise_point_ratio=0.0):
        """Class to handle PartNet dataset for fine-grained part segmentation task.

        Args:
            config: config file
            input_threads: the number elements to process in parallel
            noise_point_ratio: the ratio of outlier points in a point cloud
        """
        super(PartNetDataset, self).__init__()

        self.config = config
        self.num_threads = input_threads
        self.noise_point_ratio = noise_point_ratio
        self.label_to_names = {0: 'Bed',
                               1: 'Bottle',
                               2: 'Chair',
                               3: 'Clock',
                               4: 'Dishwasher',
                               5: 'Display',
                               6: 'Door',
                               7: 'Earphone',
                               8: 'Faucet',
                               9: 'Knife',
                               10: 'Lamp',
                               11: 'Microwave',
                               12: 'Refrigerator',
                               13: 'StorageFurniture',
                               14: 'Table',
                               15: 'TrashCan',
                               16: 'Vase', }
        self.num_parts = [15, 9, 39, 11, 7, 4, 5, 10, 12, 10, 41, 6, 7, 24, 51, 11, 6]
        self.init_labels()

        # Path of the folder containing ply files
        self.path = os.path.join(DATA_DIR, 'sem_seg_h5')

        # Number of models
        self.num_train = 17119
        self.num_val = 2492
        self.num_test = 4895

        # Some configs
        self.num_gpus = config.num_gpus
        self.first_subsampling_dl = config.first_subsampling_dl
        self.in_features_dim = config.in_features_dim
        self.downsample_times = config.num_layers - 1
        self.first_subsampling_dl = config.first_subsampling_dl
        self.density_parameter = config.density_parameter
        self.batch_size = config.batch_size
        self.augment_scale_anisotropic = config.augment_scale_anisotropic
        self.augment_symmetries = config.augment_symmetries
        self.augment_rotation = config.augment_rotation
        self.augment_scale_min = config.augment_scale_min
        self.augment_scale_max = config.augment_scale_max
        self.augment_noise = config.augment_noise

        self.prepare_PartNet_ply()
        self.load_subsampled_clouds(self.first_subsampling_dl)
        self.batch_limit = self.calibrate_batches()
        print("batch_limit: ", self.batch_limit)
        self.neighborhood_limits = [23, 38, 42, 40, 36]
        self.neighborhood_limits = [int(l * self.density_parameter // 5) for l in self.neighborhood_limits]
        print("neighborhood_limits: ", self.neighborhood_limits)
        # Get generator and mapping function
        gen_function, gen_types, gen_shapes = self.get_batch_gen('training')
        gen_function_val, _, _ = self.get_batch_gen('validation')
        gen_function_test, _, _ = self.get_batch_gen('test')
        map_func_train = self.get_tf_mapping(augment=True)
        map_func_test = self.get_tf_mapping(augment=False)
        map_func_test_vote = self.get_tf_mapping(augment=True)

        # Training dataset
        self.train_data = tf.data.Dataset.from_generator(gen_function, gen_types, gen_shapes)
        self.train_data = self.train_data.map(map_func=map_func_train, num_parallel_calls=self.num_threads)
        self.train_data = self.train_data.prefetch(10 * self.num_gpus)
        # Val dataset
        self.val_data = tf.data.Dataset.from_generator(gen_function_val, gen_types, gen_shapes)
        self.val_data = self.val_data.map(map_func=map_func_test, num_parallel_calls=self.num_threads)
        self.val_data = self.val_data.prefetch(10 * self.num_gpus)
        # Val vote dataset
        self.val_vote_data = tf.data.Dataset.from_generator(gen_function_val, gen_types, gen_shapes)
        self.val_vote_data = self.val_vote_data.map(map_func=map_func_test_vote, num_parallel_calls=self.num_threads)
        self.val_vote_data = self.val_vote_data.prefetch(10 * self.num_gpus)
        # Test dataset
        self.test_data = tf.data.Dataset.from_generator(gen_function_test, gen_types, gen_shapes)
        self.test_data = self.test_data.map(map_func=map_func_test, num_parallel_calls=self.num_threads)
        self.test_data = self.test_data.prefetch(10 * self.num_gpus)
        # Test vote dataset
        self.test_vote_data = tf.data.Dataset.from_generator(gen_function_test, gen_types, gen_shapes)
        self.test_vote_data = self.test_vote_data.map(map_func=map_func_test_vote, num_parallel_calls=self.num_threads)
        self.test_vote_data = self.test_vote_data.prefetch(10 * self.num_gpus)

        # create a iterator of the correct shape and type
        iter = tf.data.Iterator.from_structure(self.train_data.output_types, self.train_data.output_shapes)
        self.flat_inputs = [None] * self.num_gpus
        for i in range(self.num_gpus):
            self.flat_inputs[i] = iter.get_next()
        # create the initialisation operations
        self.train_init_op = iter.make_initializer(self.train_data)
        self.val_init_op = iter.make_initializer(self.val_data)
        self.test_init_op = iter.make_initializer(self.test_data)
        self.val_vote_init_op = iter.make_initializer(self.val_vote_data)
        self.test_vote_init_op = iter.make_initializer(self.test_vote_data)

    def _load_seg(self, filelist):
        points = []
        labels_seg = []
        folder = os.path.dirname(filelist)
        for line in open(filelist):
            data = h5py.File(os.path.join(folder, line.strip()), mode='r')
            points.append(data['data'][...].astype(np.float32))
            labels_seg.append(data['label_seg'][...].astype(np.int32))

        return (np.concatenate(points, axis=0),
                np.concatenate(labels_seg, axis=0))

    def prepare_PartNet_ply(self):

        print('\nPreparing ply files')
        t0 = time.time()
        # Convert to ply
        # **************
        splits = ['train', 'val', 'test']
        for split in splits:
            ply_path = os.path.join(self.path, '{:s}_ply'.format(split))
            if not os.path.exists(ply_path):
                os.makedirs(ply_path)
            for class_name in self.label_names:
                split_filelist = os.path.join(self.path, '{}-{}'.format(class_name, 3), '{:s}_files.txt'.format(split))
                split_points, split_labels = self._load_seg(split_filelist)
                N = split_points.shape[0]
                for i in range(N):
                    ply_name = os.path.join(ply_path, '{:s}_{:04d}.ply'.format(class_name, i))
                    if os.path.exists(ply_name):
                        continue
                    points = split_points[i]
                    labels = split_labels[i]

                    # Center and rescale point for 1m radius
                    pmin = np.min(points, axis=0)
                    pmax = np.max(points, axis=0)
                    points -= (pmin + pmax) / 2
                    scale = np.max(np.linalg.norm(points, axis=1))
                    points *= 1.0 / scale

                    # Switch y and z dimensions
                    points = points[:, [0, 2, 1]]

                    # Save in ply format
                    write_ply(ply_name, (points, labels), ['x', 'y', 'z', 'label'])
                    # Display
                    print('preparing {:s} {:s} ply: {:.1f}%'.format(class_name, split, 100 * i / N))

        print('Done in {:.1f}s'.format(time.time() - t0))

    def load_subsampled_clouds(self, subsampling_parameter):
        """Presubsample point clouds and load into memory
        Args:
            subsampling_parameter: base grid size for input points grid sub-sampling

        Returns:

        """

        if 0 < subsampling_parameter <= 0.01:
            raise ValueError('subsampling_parameter too low (should be over 1 cm')

        # Initiate containers
        self.input_points = {'training': [], 'validation': [], 'test': []}
        self.input_labels = {'training': [], 'validation': [], 'test': []}
        self.input_point_labels = {'training': [], 'validation': [], 'test': []}

        ################
        # Training files
        ################
        t0 = time.time()
        # Load wanted points if possible
        print('\nLoading training points')
        filename = os.path.join(self.path, 'train_{:.3f}_record.pkl'.format(subsampling_parameter))
        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                self.input_labels['training'], \
                self.input_points['training'], \
                self.input_point_labels['training'] = pickle.load(file)
        else:
            # Collect training file names
            split_path = os.path.join(self.path, '{:s}_ply'.format('train'))
            names = [f[:-4] for f in os.listdir(split_path) if f[-4:] == '.ply']
            names = np.sort(names)
            # Collect point clouds
            for i, cloud_name in enumerate(names):
                data = read_ply(os.path.join(split_path, cloud_name + '.ply'))
                points = np.vstack((data['x'], data['y'], data['z'])).T
                point_labels = data['label']
                if subsampling_parameter > 0:
                    sub_points, sub_labels = grid_subsampling(points, labels=point_labels,
                                                              sampleDl=subsampling_parameter)
                    self.input_points['training'] += [sub_points]
                    self.input_point_labels['training'] += [sub_labels]
                else:
                    self.input_points['training'] += [points]
                    self.input_point_labels['training'] += [point_labels]
            # Get labels
            label_names = ['_'.join(n.split('_')[:-1]) for n in names]
            self.input_labels['training'] = np.array([self.name_to_label[name] for name in label_names])
            # Save for later use
            with open(filename, 'wb') as file:
                pickle.dump((self.input_labels['training'],
                             self.input_points['training'],
                             self.input_point_labels['training']), file)
        lengths = [p.shape[0] for p in self.input_points['training']]
        sizes = [l * 4 * 3 for l in lengths]
        print('{:.1f} MB loaded in {:.1f}s'.format(np.sum(sizes) * 1e-6, time.time() - t0))

        #############
        # Validation files
        ############
        t0 = time.time()
        # Load wanted points if possible
        print('\nLoading val points')
        filename = os.path.join(self.path, 'val_{:.3f}_record.pkl'.format(subsampling_parameter))
        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                self.input_labels['validation'], \
                self.input_points['validation'], \
                self.input_point_labels['validation'] = pickle.load(file)
            # add outlier points if needed
            if self.noise_point_ratio > 0:
                for points, point_labels in zip(self.input_points['validation'], self.input_point_labels['validation']):
                    num_points = points.shape[0]
                    num_noise_points = int(self.noise_point_ratio * num_points)
                    random_inds = np.random.permutation(num_points)[:num_noise_points]
                    points[random_inds] = np.random.uniform(-1, 1, (num_noise_points, 3))
                    point_labels[random_inds] = 0
        else:
            # Collect training file names
            split_path = os.path.join(self.path, '{:s}_ply'.format('val'))
            names = [f[:-4] for f in os.listdir(split_path) if f[-4:] == '.ply']
            names = np.sort(names)
            # Collect point clouds
            for i, cloud_name in enumerate(names):
                data = read_ply(os.path.join(split_path, cloud_name + '.ply'))
                points = np.vstack((data['x'], data['y'], data['z'])).T
                point_labels = data['label']
                if subsampling_parameter > 0:
                    sub_points, sub_labels = grid_subsampling(points, labels=point_labels,
                                                              sampleDl=subsampling_parameter)
                    self.input_points['validation'] += [sub_points]
                    self.input_point_labels['validation'] += [sub_labels]
                else:
                    self.input_points['validation'] += [points]
                    self.input_point_labels['validation'] += [point_labels]
            # Get labels
            label_names = ['_'.join(n.split('_')[:-1]) for n in names]
            self.input_labels['validation'] = np.array([self.name_to_label[name] for name in label_names])
            # Save for later use
            with open(filename, 'wb') as file:
                pickle.dump((self.input_labels['validation'],
                             self.input_points['validation'],
                             self.input_point_labels['validation']), file)
        lengths = [p.shape[0] for p in self.input_points['validation']]
        sizes = [l * 4 * 3 for l in lengths]
        print('{:.1f} MB loaded in {:.1f}s'.format(np.sum(sizes) * 1e-6, time.time() - t0))

        ############
        # Test files
        ############
        t0 = time.time()
        # Load wanted points if possible
        print('\nLoading test points')
        filename = os.path.join(self.path, 'test_{:.3f}_record.pkl'.format(subsampling_parameter))
        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                self.input_labels['test'], \
                self.input_points['test'], \
                self.input_point_labels['test'] = pickle.load(file)
            if self.noise_point_ratio > 0:
                for points, point_labels in zip(self.input_points['test'], self.input_point_labels['test']):
                    num_points = points.shape[0]
                    num_noise_points = int(self.noise_point_ratio * num_points)
                    random_inds = np.random.permutation(num_points)[:num_noise_points]
                    points[random_inds] = np.random.uniform(-1, 1, (num_noise_points, 3))
                    point_labels[random_inds] = 0
        else:
            # Collect test file names
            split_path = os.path.join(self.path, '{:s}_ply'.format('test'))
            names = [f[:-4] for f in os.listdir(split_path) if f[-4:] == '.ply']
            names = np.sort(names)
            # Collect point clouds
            for i, cloud_name in enumerate(names):
                data = read_ply(os.path.join(split_path, cloud_name + '.ply'))
                points = np.vstack((data['x'], data['y'], data['z'])).T
                point_labels = data['label']
                if subsampling_parameter > 0:
                    sub_points, sub_labels = grid_subsampling(points, labels=point_labels,
                                                              sampleDl=subsampling_parameter)
                    self.input_points['test'] += [sub_points]
                    self.input_point_labels['test'] += [sub_labels]
                else:
                    self.input_points['test'] += [points]
                    self.input_point_labels['test'] += [point_labels]
            # Get labels
            label_names = ['_'.join(n.split('_')[:-1]) for n in names]
            self.input_labels['test'] = np.array([self.name_to_label[name] for name in label_names])
            # Save for later use
            with open(filename, 'wb') as file:
                pickle.dump((self.input_labels['test'],
                             self.input_points['test'],
                             self.input_point_labels['test']), file)
        lengths = [p.shape[0] for p in self.input_points['test']]
        sizes = [l * 4 * 3 for l in lengths]
        print('{:.1f} MB loaded in {:.1f}s\n'.format(np.sum(sizes) * 1e-6, time.time() - t0))

    def get_batch_gen(self, split):
        def variable_batch_gen():

            # Initiate concatanation lists
            tp_list = []
            tl_list = []
            tpl_list = []
            ti_list = []
            batch_n = 0

            # Initiate parameters depending on the chosen split
            if split == 'training':
                gen_indices = np.random.permutation(self.num_train)
            elif split == 'validation':
                gen_indices1 = np.arange(self.num_val)
                # when using multi gpus for validation and testing, add padding for undesired dropping last batch.
                gen_indices2 = np.arange((self.num_gpus - 1) * 5)
                gen_indices = np.hstack([gen_indices1, gen_indices2])
            elif split == 'test':
                gen_indices1 = np.arange(self.num_test)
                gen_indices2 = np.arange((self.num_gpus - 1) * 5)
                gen_indices = np.hstack([gen_indices1, gen_indices2])
            else:
                raise ValueError('Split argument in data generator should be "training", "validation" or "test"')

            # Generator loop
            for i, rand_i in enumerate(gen_indices):
                # Get points
                new_points = self.input_points[split][rand_i].astype(np.float32)
                n = new_points.shape[0]
                # In case batch is full, yield it and reset it
                if batch_n + n > self.batch_limit:
                    yield (np.concatenate(tp_list, axis=0),
                           np.array(tl_list, dtype=np.int32),
                           np.concatenate(tpl_list, axis=0),
                           np.array(ti_list, dtype=np.int32),
                           np.array([tp.shape[0] for tp in tp_list]))
                    tp_list = []
                    tl_list = []
                    tpl_list = []
                    ti_list = []
                    batch_n = 0
                # Add data to current batch
                tp_list += [new_points]
                tl_list += [self.input_labels[split][rand_i]]
                tpl_list += [np.squeeze(self.input_point_labels[split][rand_i])]
                ti_list += [rand_i]

                # Update batch size
                batch_n += n

            yield (np.concatenate(tp_list, axis=0),
                   np.array(tl_list, dtype=np.int32),
                   np.concatenate(tpl_list, axis=0),
                   np.array(ti_list, dtype=np.int32),
                   np.array([tp.shape[0] for tp in tp_list]))

        # Generator types and shapes
        gen_types = (tf.float32, tf.int32, tf.int32, tf.int32, tf.int32)
        gen_shapes = ([None, 3], [None], [None], [None], [None])
        return variable_batch_gen, gen_types, gen_shapes

    def get_tf_mapping(self, augment=True):

        def tf_map_multi(stacked_points, object_labels, point_labels, obj_inds, stack_lengths):
            """
            From the input point cloud, this function compute all the point clouds at each layer, the neighbors
            indices, the pooling indices and other useful variables.
            :param stacked_points: Tensor with size [None, 3] where None is the total number of points
            :param labels: Tensor with size [None] where None is the number of batch
            :param stack_lengths: Tensor with size [None] where None is the number of batch
            """

            # Get batch indice for each point
            batch_inds = self.tf_get_batch_inds(stack_lengths)

            # Augment input points
            if augment:
                stacked_points, scales, rots = self.tf_augment_input(stacked_points,
                                                                     batch_inds)
            else:
                num_batches = batch_inds[-1] + 1
                scales = tf.ones(shape=(num_batches, 3))
                rots = tf.eye(3, batch_shape=(num_batches,))
            # First add a column of 1 as feature for the network to be able to learn 3D shapes
            stacked_features = tf.ones((tf.shape(stacked_points)[0], 1), dtype=tf.float32)

            # Then use positions or not
            if self.in_features_dim == 1:
                pass
            elif self.in_features_dim == 4:
                stacked_features = tf.concat((stacked_features, stacked_points), axis=1)
            elif self.in_features_dim == 7:
                stacked_features = tf.concat((stacked_features, stacked_points, tf.square(stacked_points)), axis=1)
            else:
                raise ValueError('Only accepted input dimensions are 1, 4 and 7 (without and with XYZ)')

            # Get the whole input list
            input_list = self.tf_segmentation_inputs(self.downsample_times,
                                                     self.first_subsampling_dl,
                                                     self.density_parameter,
                                                     stacked_points,
                                                     stacked_features,
                                                     point_labels,
                                                     stack_lengths,
                                                     batch_inds,
                                                     object_labels=object_labels)

            # Add scale and rotation for testing
            input_list += [scales, rots, obj_inds]

            return input_list

        return tf_map_multi

    def tf_segmentation_inputs(self,
                               downsample_times,
                               first_subsampling_dl,
                               density_parameter,
                               stacked_points,
                               stacked_features,
                               point_labels,
                               stacks_lengths,
                               batch_inds,
                               object_labels=None):
        # Batch weight at each point for loss (inverse of stacks_lengths for each point)
        min_len = tf.reduce_min(stacks_lengths, keep_dims=True)
        batch_weights = tf.cast(min_len, tf.float32) / tf.cast(stacks_lengths, tf.float32)
        stacked_weights = tf.gather(batch_weights, batch_inds)
        # Starting radius of convolutions
        dl = first_subsampling_dl
        dp = density_parameter
        r = dl * dp / 2.0
        # Lists of inputs
        num_layers = (downsample_times + 1)
        input_points = [None] * num_layers
        input_neighbors = [None] * num_layers
        input_pools = [None] * num_layers
        input_upsamples = [None] * num_layers
        input_batches_len = [None] * num_layers

        neighbors_inds = tf_batch_neighbors(stacked_points, stacked_points, stacks_lengths, stacks_lengths, r)
        pool_points, pool_stacks_lengths = tf_batch_subsampling(stacked_points, stacks_lengths, sampleDl=2 * dl)
        pool_inds = tf_batch_neighbors(pool_points, stacked_points, pool_stacks_lengths, stacks_lengths, r)
        up_inds = tf_batch_neighbors(stacked_points, pool_points, stacks_lengths, pool_stacks_lengths, 2 * r)
        neighbors_inds = self.big_neighborhood_filter(neighbors_inds, 0)
        pool_inds = self.big_neighborhood_filter(pool_inds, 0)
        up_inds = self.big_neighborhood_filter(up_inds, 0)
        input_points[0] = stacked_points
        input_neighbors[0] = neighbors_inds
        input_pools[0] = pool_inds
        input_upsamples[0] = tf.zeros((0, 1), dtype=tf.int32)
        input_upsamples[1] = up_inds
        input_batches_len[0] = stacks_lengths
        stacked_points = pool_points
        stacks_lengths = pool_stacks_lengths
        r *= 2
        dl *= 2

        for dt in range(1, downsample_times):
            neighbors_inds = tf_batch_neighbors(stacked_points, stacked_points, stacks_lengths, stacks_lengths, r)
            pool_points, pool_stacks_lengths = tf_batch_subsampling(stacked_points, stacks_lengths, sampleDl=2 * dl)
            pool_inds = tf_batch_neighbors(pool_points, stacked_points, pool_stacks_lengths, stacks_lengths, r)
            up_inds = tf_batch_neighbors(stacked_points, pool_points, stacks_lengths, pool_stacks_lengths, 2 * r)
            neighbors_inds = self.big_neighborhood_filter(neighbors_inds, dt)
            pool_inds = self.big_neighborhood_filter(pool_inds, dt)
            up_inds = self.big_neighborhood_filter(up_inds, dt)
            input_points[dt] = stacked_points
            input_neighbors[dt] = neighbors_inds
            input_pools[dt] = pool_inds
            input_upsamples[dt + 1] = up_inds
            input_batches_len[dt] = stacks_lengths
            stacked_points = pool_points
            stacks_lengths = pool_stacks_lengths
            r *= 2
            dl *= 2

        neighbors_inds = tf_batch_neighbors(stacked_points, stacked_points, stacks_lengths, stacks_lengths, r)
        neighbors_inds = self.big_neighborhood_filter(neighbors_inds, downsample_times)
        input_points[downsample_times] = stacked_points
        input_neighbors[downsample_times] = neighbors_inds
        input_pools[downsample_times] = tf.zeros((0, 1), dtype=tf.int32)
        input_batches_len[downsample_times] = stacks_lengths

        # Batch unstacking (with first layer indices for optionnal classif loss)
        stacked_batch_inds_0 = self.tf_stack_batch_inds(input_batches_len[0])
        # Batch unstacking (with last layer indices for optionnal classif loss)
        stacked_batch_inds_1 = self.tf_stack_batch_inds(input_batches_len[-1])
        # list of network inputs
        if object_labels is None:
            # list of network inputs
            li = input_points + input_neighbors + input_pools + input_upsamples
            li += [stacked_features, stacked_weights, stacked_batch_inds_0, stacked_batch_inds_1]
            li += [point_labels]
            return li
        else:
            # Object class ind for each point
            stacked_object_labels = tf.gather(object_labels, batch_inds)
            # list of network inputs
            li = input_points + input_neighbors + input_pools + input_upsamples
            li += [stacked_features, stacked_weights, stacked_batch_inds_0, stacked_batch_inds_1]
            li += [point_labels, stacked_object_labels]
            return li

    def calibrate_batches(self):

        # Get sizes at training and sort them
        sizes = np.sort([p.shape[0] for p in self.input_points['training']])

        # Higher bound for batch limit
        lim = sizes[-1] * self.batch_size

        # Biggest batch size with this limit
        sum_s = 0
        max_b = 0
        for i, s in enumerate(sizes):
            sum_s += s
            if sum_s > lim:
                max_b = i
                break

        # With a proportional corrector, find batch limit which gets the wanted batch_num
        estim_b = 0
        for i in range(10000):
            # Compute a random batch
            rand_shapes = np.random.choice(sizes, size=max_b, replace=False)
            b = np.sum(np.cumsum(rand_shapes) < lim)

            # Update estim_b (low pass filter istead of real mean
            estim_b += (b - estim_b) / min(i + 1, 100)

            # Correct batch limit
            lim += 10.0 * (self.batch_size - estim_b)

        return lim
