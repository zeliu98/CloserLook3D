import os
import sys
import time
import pickle
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data', 'ModelNet40')

if not os.path.exists(DATA_DIR):
    raise IOError(f"{DATA_DIR} not found!")

from .custom_dataset import CustomDataset, grid_subsampling, tf_batch_subsampling, tf_batch_neighbors


class ModelNetDataset(CustomDataset):
    def __init__(self, config, input_threads=8):
        """Class to handle ModelNet40 dataset for shape classification.

        Args:
            config: config file
            input_threads: the number elements to process in parallel
        """
        super(ModelNetDataset, self).__init__()
        self.num_threads = input_threads
        self.path = DATA_DIR
        self.data_folder = 'modelnet40_normal_resampled'
        self.label_to_names = {0: 'airplane',
                               1: 'bathtub',
                               2: 'bed',
                               3: 'bench',
                               4: 'bookshelf',
                               5: 'bottle',
                               6: 'bowl',
                               7: 'car',
                               8: 'chair',
                               9: 'cone',
                               10: 'cup',
                               11: 'curtain',
                               12: 'desk',
                               13: 'door',
                               14: 'dresser',
                               15: 'flower_pot',
                               16: 'glass_box',
                               17: 'guitar',
                               18: 'keyboard',
                               19: 'lamp',
                               20: 'laptop',
                               21: 'mantel',
                               22: 'monitor',
                               23: 'night_stand',
                               24: 'person',
                               25: 'piano',
                               26: 'plant',
                               27: 'radio',
                               28: 'range_hood',
                               29: 'sink',
                               30: 'sofa',
                               31: 'stairs',
                               32: 'stool',
                               33: 'table',
                               34: 'tent',
                               35: 'toilet',
                               36: 'tv_stand',
                               37: 'vase',
                               38: 'wardrobe',
                               39: 'xbox'}
        self.init_labels()
        # Number of models
        self.num_train = 9843
        self.num_test = 2468
        self.num_val = 2468
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

        self.load_subsampled_clouds(self.first_subsampling_dl)
        self.batch_limit = self.calibrate_batches()
        print("batch_limit: ", self.batch_limit)
        self.neighborhood_limits = [20, 31, 38, 36, 34]
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

        # create a iterator of the correct shape and type
        iter = tf.data.Iterator.from_structure(self.train_data.output_types, self.train_data.output_shapes)
        self.flat_inputs = [None] * self.num_gpus
        for i in range(self.num_gpus):
            self.flat_inputs[i] = iter.get_next()
        # create the initialisation operations
        self.train_init_op = iter.make_initializer(self.train_data)
        self.val_init_op = iter.make_initializer(self.val_data)
        self.val_vote_init_op = iter.make_initializer(self.val_vote_data)

    def load_subsampled_clouds(self, subsampling_parameter):
        """
        Presubsample point clouds and load into memory
        """

        if 0 < subsampling_parameter <= 0.01:
            raise ValueError('subsampling_parameter too low (should be over 1 cm')

        # Initiate containers
        self.input_points = {'training': [], 'validation': [], 'test': []}
        self.input_normals = {'training': [], 'validation': [], 'test': []}
        self.input_labels = {'training': [], 'validation': [], 'test': []}

        ################
        # Training files
        ################
        t0 = time.time()
        # Load wanted points if possible
        print('\nLoading training points')
        filename = os.path.join(self.path, 'train_{:.3f}_record.pkl'.format(subsampling_parameter))
        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                self.input_points['training'], \
                self.input_normals['training'], \
                self.input_labels['training'] = pickle.load(file)
        else:
            # Collect training file names
            names = np.loadtxt(os.path.join(self.path, self.data_folder, 'modelnet40_train.txt'), dtype=np.str)
            # Collect point clouds
            for i, cloud_name in enumerate(names):

                # Read points
                class_folder = '_'.join(cloud_name.split('_')[:-1])
                txt_file = os.path.join(self.path, self.data_folder, class_folder, cloud_name) + '.txt'
                data = np.loadtxt(txt_file, delimiter=',', dtype=np.float32)

                # Subsample them
                if subsampling_parameter > 0:
                    points, normals = grid_subsampling(data[:, :3],
                                                       features=data[:, 3:],
                                                       sampleDl=subsampling_parameter)
                else:
                    points = data[:, :3]
                    normals = data[:, 3:]

                # Add to list
                self.input_points['training'] += [points]
                self.input_normals['training'] += [normals]
            # Get labels
            label_names = ['_'.join(name.split('_')[:-1]) for name in names]
            self.input_labels['training'] = np.array([self.name_to_label[name] for name in label_names])
            # Save for later use
            with open(filename, 'wb') as file:
                pickle.dump((self.input_points['training'],
                             self.input_normals['training'],
                             self.input_labels['training']), file)

        lengths = [p.shape[0] for p in self.input_points['training']]
        sizes = [l * 4 * 6 for l in lengths]
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
                self.input_points['validation'], \
                self.input_normals['validation'], \
                self.input_labels['validation'] = pickle.load(file)
        else:
            # Collect test file names
            names = np.loadtxt(os.path.join(self.path, self.data_folder, 'modelnet40_test.txt'), dtype=np.str)
            # Collect point clouds
            for i, cloud_name in enumerate(names):

                # Read points
                class_folder = '_'.join(cloud_name.split('_')[:-1])
                txt_file = os.path.join(self.path, self.data_folder, class_folder, cloud_name) + '.txt'
                data = np.loadtxt(txt_file, delimiter=',', dtype=np.float32)

                # Subsample them
                if subsampling_parameter > 0:
                    points, normals = grid_subsampling(data[:, :3],
                                                       features=data[:, 3:],
                                                       sampleDl=subsampling_parameter)
                else:
                    points = data[:, :3]
                    normals = data[:, 3:]

                # Add to list
                self.input_points['validation'] += [points]
                self.input_normals['validation'] += [normals]
            # Get labels
            label_names = ['_'.join(name.split('_')[:-1]) for name in names]
            self.input_labels['validation'] = np.array([self.name_to_label[name] for name in label_names])
            # Save for later use
            with open(filename, 'wb') as file:
                pickle.dump((self.input_points['validation'],
                             self.input_normals['validation'],
                             self.input_labels['validation']), file)

        lengths = [p.shape[0] for p in self.input_points['validation']]
        sizes = [l * 4 * 6 for l in lengths]
        print('{:.1f} MB loaded in {:.1f}s\n'.format(np.sum(sizes) * 1e-6, time.time() - t0))

        # Test = validation
        self.input_points['test'] = self.input_points['validation']
        self.input_normals['test'] = self.input_normals['validation']
        self.input_labels['test'] = self.input_labels['validation']

        return

    def get_batch_gen(self, split, balanced=False):

        def random_balanced_gen():
            # Initiate concatenation lists
            tp_list = []
            tn_list = []
            tl_list = []
            ti_list = []
            batch_n = 0
            # Initiate parameters depending on the chosen split
            if split == 'training':
                if balanced:
                    pick_n = int(np.ceil(self.num_train / self.num_classes))
                    gen_indices = []
                    for l in self.label_values:
                        label_inds = np.where(np.equal(self.input_labels[split], l))[0]
                        rand_inds = np.random.choice(label_inds, size=pick_n, replace=True)
                        gen_indices += [rand_inds]
                    gen_indices = np.random.permutation(np.hstack(gen_indices))
                else:
                    gen_indices = np.random.permutation(self.num_train)
            elif split == 'validation':
                if self.num_gpus == 4:
                    gen_indices = np.arange(self.num_test + 64)
                    gen_indices[self.num_test:] = 0
                else:
                    gen_indices = np.arange(self.num_test)
            elif split == 'test':
                if self.num_gpus == 4:
                    gen_indices = np.arange(self.num_test + 64)
                    gen_indices[self.num_test:] = 0
                else:
                    gen_indices = np.arange(self.num_test)
            else:
                raise ValueError('Wrong split argument in data generator: ' + split)

            # Generator loop
            for p_i in gen_indices:
                # Get points
                new_points = self.input_points[split][p_i].astype(np.float32)
                new_normals = self.input_normals[split][p_i].astype(np.float32)
                n = new_points.shape[0]

                # Collect labels
                input_label = self.label_to_idx[self.input_labels[split][p_i]]

                # In case batch is full, yield it and reset it
                if batch_n + n > self.batch_limit and batch_n > 0:
                    yield (np.concatenate(tp_list, axis=0),
                           np.concatenate(tn_list, axis=0),
                           np.array(tl_list, dtype=np.int32),
                           np.array(ti_list, dtype=np.int32),
                           np.array([tp.shape[0] for tp in tp_list]))
                    tp_list = []
                    tn_list = []
                    tl_list = []
                    ti_list = []
                    batch_n = 0

                # Add data to current batch
                tp_list += [new_points]
                tn_list += [new_normals]
                tl_list += [input_label]
                ti_list += [p_i]

                # Update batch size
                batch_n += n

            yield (np.concatenate(tp_list, axis=0),
                   np.concatenate(tn_list, axis=0),
                   np.array(tl_list, dtype=np.int32),
                   np.array(ti_list, dtype=np.int32),
                   np.array([tp.shape[0] for tp in tp_list]))

        ##################
        # Return generator
        ##################
        # Generator types and shapes
        gen_types = (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32)
        gen_shapes = ([None, 3], [None, 3], [None], [None], [None])

        return random_balanced_gen, gen_types, gen_shapes

    def get_tf_mapping(self, augment=True):

        def tf_map(stacked_points, stacked_normals, labels, obj_inds, stack_lengths):
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
            elif self.in_features_dim == 3:
                stacked_features = tf.concat((stacked_features, stacked_points), axis=1)
            elif self.in_features_dim == 4:
                stacked_features = tf.concat((stacked_features, stacked_normals), axis=1)
            elif self.in_features_dim == 5:
                angles = tf.asin(tf.abs(stacked_normals)) * (2 / np.pi)
                stacked_features = tf.concat((stacked_features, angles), axis=1)
            elif self.in_features_dim == 7:
                stacked_features = tf.concat((stacked_features, stacked_points, stacked_normals), axis=1)
            else:
                raise ValueError('Only accepted input dimensions are 1, 4 and 7 (without and with XYZ)')

            # Get the whole input list
            input_list = self.tf_classification_inputs(self.downsample_times,
                                                       self.first_subsampling_dl,
                                                       self.density_parameter,
                                                       stacked_points,
                                                       stacked_features,
                                                       labels,
                                                       stack_lengths,
                                                       batch_inds)

            # Add scale and rotation for testing
            input_list += [scales, rots, obj_inds]

            return input_list

        return tf_map

    def tf_classification_inputs(self,
                                 downsample_times,
                                 first_subsampling_dl,
                                 density_parameter,
                                 stacked_points,
                                 stacked_features,
                                 labels,
                                 stacks_lengths,
                                 batch_inds):
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
        input_batches_len = [None] * num_layers

        neighbors_inds = tf_batch_neighbors(stacked_points, stacked_points, stacks_lengths, stacks_lengths, r)
        pool_points, pool_stacks_lengths = tf_batch_subsampling(stacked_points, stacks_lengths, sampleDl=2 * dl)
        pool_inds = tf_batch_neighbors(pool_points, stacked_points, pool_stacks_lengths, stacks_lengths, r)
        neighbors_inds = self.big_neighborhood_filter(neighbors_inds, 0)
        pool_inds = self.big_neighborhood_filter(pool_inds, 0)
        input_points[0] = stacked_points
        input_neighbors[0] = neighbors_inds
        input_pools[0] = pool_inds
        input_batches_len[0] = stacks_lengths
        stacked_points = pool_points
        stacks_lengths = pool_stacks_lengths
        r *= 2
        dl *= 2

        for dt in range(1, downsample_times):
            neighbors_inds = tf_batch_neighbors(stacked_points, stacked_points, stacks_lengths, stacks_lengths, r)
            pool_points, pool_stacks_lengths = tf_batch_subsampling(stacked_points, stacks_lengths, sampleDl=2 * dl)
            pool_inds = tf_batch_neighbors(pool_points, stacked_points, pool_stacks_lengths, stacks_lengths, r)
            neighbors_inds = self.big_neighborhood_filter(neighbors_inds, dt)
            pool_inds = self.big_neighborhood_filter(pool_inds, dt)
            input_points[dt] = stacked_points
            input_neighbors[dt] = neighbors_inds
            input_pools[dt] = pool_inds
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
        li = input_points + input_neighbors + input_pools
        li += [stacked_features, stacked_weights, stacked_batch_inds_0, stacked_batch_inds_1]
        li += [labels]
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
