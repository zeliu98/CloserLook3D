import os
import sys
import numpy as np
import time
import pickle
from sklearn.neighbors import KDTree
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data', 'S3DIS')

if not os.path.exists(DATA_DIR):
    raise IOError(f"{DATA_DIR} not found!")

from utils.ply import read_ply, write_ply
from .custom_dataset import CustomDataset, grid_subsampling, tf_batch_subsampling, tf_batch_neighbors


class S3DISDataset(CustomDataset):

    def __init__(self, config, input_threads=8):
        """Class to handle S3DIS dataset for scene segmentation task.

        Args:
            config: config file
            input_threads: the number elements to process in parallel
        """
        super(S3DISDataset, self).__init__()
        self.config = config
        self.num_threads = input_threads

        # Dict from labels to names
        self.label_to_names = {0: 'ceiling',
                               1: 'floor',
                               2: 'wall',
                               3: 'beam',
                               4: 'column',
                               5: 'window',
                               6: 'door',
                               7: 'chair',
                               8: 'table',
                               9: 'bookcase',
                               10: 'sofa',
                               11: 'board',
                               12: 'clutter'}

        # Initiate a bunch of variables concerning class labels
        self.init_labels()
        config.num_classes = self.num_classes
        # Number of input threads
        self.num_threads = input_threads

        # Path of the folder containing ply files
        self.path = os.path.join(DATA_DIR, 'Stanford3dDataset_v1.2')
        # Path of the training files
        self.train_path = 'original_ply'
        # List of files to process
        ply_path = os.path.join(self.path, self.train_path)
        # Proportion of validation scenes
        self.cloud_names = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_5', 'Area_6']
        self.all_splits = [0, 1, 2, 3, 4, 5]
        self.validation_split = 4
        # List of training files
        self.train_files = [os.path.join(ply_path, f + '.ply') for f in self.cloud_names]

        # Some configs
        self.num_gpus = config.num_gpus
        self.first_subsampling_dl = config.first_subsampling_dl
        self.in_features_dim = config.in_features_dim
        self.num_layers = config.num_layers
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
        self.augment_color = config.augment_color
        self.epoch_steps = config.epoch_steps
        self.validation_size = config.validation_size
        self.in_radius = config.in_radius

        # prepare ply file
        self.prepare_S3DIS_ply()

        # input subsampling
        self.load_subsampled_clouds(self.first_subsampling_dl)

        self.batch_limit = self.calibrate_batches()
        print("batch_limit: ", self.batch_limit)
        self.neighborhood_limits = [26, 31, 38, 41, 39]
        self.neighborhood_limits = [int(l * self.density_parameter // 5) for l in self.neighborhood_limits]
        print("neighborhood_limits: ", self.neighborhood_limits)

        # Get generator and mapping function
        gen_function, gen_types, gen_shapes = self.get_batch_gen('training')
        gen_function_val, _, _ = self.get_batch_gen('validation')
        map_func = self.get_tf_mapping()

        self.train_data = tf.data.Dataset.from_generator(gen_function, gen_types, gen_shapes)
        self.train_data = self.train_data.map(map_func=map_func, num_parallel_calls=self.num_threads)
        self.train_data = self.train_data.prefetch(10)

        self.val_data = tf.data.Dataset.from_generator(gen_function_val, gen_types, gen_shapes)
        self.val_data = self.val_data.map(map_func=map_func, num_parallel_calls=self.num_threads)
        self.val_data = self.val_data.prefetch(10)

        # create a iterator of the correct shape and type
        # create a iterator of the correct shape and type
        iter = tf.data.Iterator.from_structure(self.train_data.output_types, self.train_data.output_shapes)
        self.flat_inputs = [None] * self.num_gpus
        for i in range(self.num_gpus):
            self.flat_inputs[i] = iter.get_next()
        # create the initialisation operations
        self.train_init_op = iter.make_initializer(self.train_data)
        self.val_init_op = iter.make_initializer(self.val_data)

    def prepare_S3DIS_ply(self):

        print('\nPreparing ply files')
        t0 = time.time()

        # Folder for the ply files
        ply_path = os.path.join(self.path, self.train_path)
        if not os.path.exists(ply_path):
            os.makedirs(ply_path)

        for cloud_name in self.cloud_names:

            # Pass if the cloud has already been computed
            cloud_file = os.path.join(ply_path, cloud_name + '.ply')
            if os.path.exists(cloud_file):
                continue

            # Get rooms of the current cloud
            cloud_folder = os.path.join(self.path, cloud_name)
            room_folders = [os.path.join(cloud_folder, room) for room in os.listdir(cloud_folder) if
                            os.path.isdir(os.path.join(cloud_folder, room))]

            # Initiate containers
            cloud_points = np.empty((0, 3), dtype=np.float32)
            cloud_colors = np.empty((0, 3), dtype=np.uint8)
            cloud_classes = np.empty((0, 1), dtype=np.int32)

            # Loop over rooms
            for i, room_folder in enumerate(room_folders):

                print(
                    'Cloud %s - Room %d/%d : %s' % (cloud_name, i + 1, len(room_folders), room_folder.split('\\')[-1]))

                for object_name in os.listdir(os.path.join(room_folder, 'Annotations')):

                    if object_name[-4:] == '.txt':

                        # Text file containing point of the object
                        object_file = os.path.join(room_folder, 'Annotations', object_name)

                        # Object class and ID
                        tmp = object_name[:-4].split('_')[0]
                        if tmp in self.name_to_label:
                            object_class = self.name_to_label[tmp]
                        elif tmp in ['stairs']:
                            object_class = self.name_to_label['clutter']
                        else:
                            raise ValueError('Unknown object name: ' + str(tmp))

                        # Read object points and colors
                        with open(object_file, 'r') as f:
                            object_data = np.array([[float(x) for x in line.split()] for line in f])

                        # Stack all data
                        cloud_points = np.vstack((cloud_points, object_data[:, 0:3].astype(np.float32)))
                        cloud_colors = np.vstack((cloud_colors, object_data[:, 3:6].astype(np.uint8)))
                        object_classes = np.full((object_data.shape[0], 1), object_class, dtype=np.int32)
                        cloud_classes = np.vstack((cloud_classes, object_classes))

            # Save as ply
            write_ply(cloud_file,
                      (cloud_points, cloud_colors, cloud_classes),
                      ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

        print('Done in {:.1f}s'.format(time.time() - t0))

    def load_subsampled_clouds(self, subsampling_parameter):
        """
        Presubsample point clouds and load into memory (Load KDTree for neighbors searches)
        """

        if 0 < subsampling_parameter <= 0.01:
            raise ValueError('subsampling_parameter too low (should be over 1 cm')

        # Create path for files
        tree_path = os.path.join(self.path, 'input_{:.3f}'.format(subsampling_parameter))
        if not os.path.exists(tree_path):
            os.makedirs(tree_path)

        # Initiate containers
        self.input_trees = {'training': [], 'validation': []}
        self.input_colors = {'training': [], 'validation': []}
        self.input_labels = {'training': [], 'validation': []}

        for i, file_path in enumerate(self.train_files):

            # Restart timer
            t0 = time.time()

            # get cloud name and split
            cloud_name = file_path.split('/')[-1][:-4]
            if self.all_splits[i] == self.validation_split:
                cloud_split = 'validation'
            else:
                cloud_split = 'training'

            # Name of the input files
            KDTree_file = os.path.join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = os.path.join(tree_path, '{:s}.ply'.format(cloud_name))

            # Check if inputs have already been computed
            if os.path.isfile(KDTree_file):
                print('\nFound KDTree for cloud {:s}, subsampled at {:.3f}'.format(cloud_name, subsampling_parameter))

                # read ply with data
                data = read_ply(sub_ply_file)
                sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
                sub_labels = data['class']

                # Read pkl with search tree
                with open(KDTree_file, 'rb') as f:
                    search_tree = pickle.load(f)

            else:
                print(
                    '\nPreparing KDTree for cloud {:s}, subsampled at {:.3f}'.format(cloud_name, subsampling_parameter))

                # Read ply file
                data = read_ply(file_path)
                points = np.vstack((data['x'], data['y'], data['z'])).T
                colors = np.vstack((data['red'], data['green'], data['blue'])).T
                labels = data['class']

                # Subsample cloud
                sub_points, sub_colors, sub_labels = grid_subsampling(points,
                                                                      features=colors,
                                                                      labels=labels,
                                                                      sampleDl=subsampling_parameter)

                # Rescale float color and squeeze label
                sub_colors = sub_colors / 255
                sub_labels = np.squeeze(sub_labels)

                # Get chosen neighborhoods
                search_tree = KDTree(sub_points, leaf_size=50)

                # Save KDTree
                with open(KDTree_file, 'wb') as f:
                    pickle.dump(search_tree, f)

                # Save ply
                write_ply(sub_ply_file,
                          [sub_points, sub_colors, sub_labels],
                          ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

            # Fill data containers
            self.input_trees[cloud_split] += [search_tree]
            self.input_colors[cloud_split] += [sub_colors]
            self.input_labels[cloud_split] += [sub_labels]

            size = sub_colors.shape[0] * 4 * 7
            print('{:.1f} MB loaded in {:.1f}s'.format(size * 1e-6, time.time() - t0))

        print('\nPreparing reprojection indices for testing')

        # Get number of clouds
        self.num_training = len(self.input_trees['training'])
        self.num_validation = len(self.input_trees['validation'])

        # Get validation and test reprojection indices
        self.validation_proj = []
        self.validation_labels = []
        i_val = 0
        for i, file_path in enumerate(self.train_files):

            # Restart timer
            t0 = time.time()

            # Get info on this cloud
            cloud_name = file_path.split('/')[-1][:-4]

            # Validation projection and labels
            if self.all_splits[i] == self.validation_split:
                proj_file = os.path.join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                if os.path.isfile(proj_file):
                    with open(proj_file, 'rb') as f:
                        proj_inds, labels = pickle.load(f)
                else:
                    data = read_ply(file_path)
                    points = np.vstack((data['x'], data['y'], data['z'])).T
                    labels = data['class']

                    # Compute projection inds
                    proj_inds = np.squeeze(self.input_trees['validation'][i_val].query(points, return_distance=False))
                    proj_inds = proj_inds.astype(np.int32)

                    # Save
                    with open(proj_file, 'wb') as f:
                        pickle.dump([proj_inds, labels], f)

                self.validation_proj += [proj_inds]
                self.validation_labels += [labels]
                i_val += 1
                print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))

        return

    def get_batch_gen(self, split):
        """
        A function defining the batch generator for each split. Should return the generator, the generated types and
        generated shapes
        :param split: string in "training", "validation" or "test"
        :return: gen_func, gen_types, gen_shapes
        """

        ############
        # Parameters
        ############

        # Initiate parameters depending on the chosen split
        if split == 'training':
            # First compute the number of point we want to pick in each cloud and for each class
            epoch_n = self.epoch_steps * self.batch_size
        elif split == 'validation':
            # First compute the number of point we want to pick in each cloud and for each class
            epoch_n = self.validation_size * self.batch_size
        elif split == 'test':
            # First compute the number of point we want to pick in each cloud and for each class
            epoch_n = self.validation_size * self.batch_size
        else:
            raise ValueError('Split argument in data generator should be "training", "validation" or "test"')

        # Initiate potentials for regular generation
        if not hasattr(self, 'potentials'):
            self.potentials = {}
            self.min_potentials = {}

        # Reset potentials
        self.potentials[split] = []
        self.min_potentials[split] = []
        data_split = split
        for i, tree in enumerate(self.input_trees[data_split]):
            self.potentials[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_potentials[split] += [float(np.min(self.potentials[split][-1]))]

        ##########################
        # Def generators
        ##########################
        def spatially_regular_gen():

            # Initiate concatanation lists
            p_list = []
            c_list = []
            pl_list = []
            pi_list = []
            ci_list = []

            batch_n = 0

            # Generator loop
            for i in range(epoch_n):

                # Choose a random cloud
                cloud_ind = int(np.argmin(self.min_potentials[split]))

                # Choose point ind as minimum of potentials
                point_ind = np.argmin(self.potentials[split][cloud_ind])

                # Get points from tree structure
                points = np.array(self.input_trees[data_split][cloud_ind].data, copy=False)

                # Center point of input region
                center_point = points[point_ind, :].reshape(1, -1)

                # Add noise to the center point
                noise = np.random.normal(scale=self.in_radius / 10, size=center_point.shape)
                pick_point = center_point + noise.astype(center_point.dtype)

                # Indices of points in input region
                input_inds = self.input_trees[data_split][cloud_ind].query_radius(pick_point,
                                                                                  r=self.in_radius)[0]

                # Number collected
                n = input_inds.shape[0]

                # Update potentials (Tuckey weights)
                dists = np.sum(np.square((points[input_inds] - pick_point).astype(np.float32)), axis=1)
                tukeys = np.square(1 - dists / np.square(self.in_radius))
                tukeys[dists > np.square(self.in_radius)] = 0
                self.potentials[split][cloud_ind][input_inds] += tukeys
                self.min_potentials[split][cloud_ind] = float(np.min(self.potentials[split][cloud_ind]))

                # Safe check for very dense areas
                if n > self.batch_limit:
                    input_inds = np.random.choice(input_inds, size=int(self.batch_limit) - 1, replace=False)
                    n = input_inds.shape[0]

                # Collect points and colors
                input_points = (points[input_inds] - pick_point).astype(np.float32)
                input_colors = self.input_colors[data_split][cloud_ind][input_inds]
                if split == 'test':
                    input_labels = np.zeros(input_points.shape[0])
                else:
                    input_labels = self.input_labels[data_split][cloud_ind][input_inds]
                    input_labels = np.array([self.label_to_idx[l] for l in input_labels])

                # In case batch is full, yield it and reset it
                if batch_n + n > self.batch_limit and batch_n > 0:
                    yield (np.concatenate(p_list, axis=0),
                           np.concatenate(c_list, axis=0),
                           np.concatenate(pl_list, axis=0),
                           np.array([tp.shape[0] for tp in p_list]),
                           np.concatenate(pi_list, axis=0),
                           np.array(ci_list, dtype=np.int32))

                    p_list = []
                    c_list = []
                    pl_list = []
                    pi_list = []
                    ci_list = []
                    batch_n = 0

                # Add data to current batch
                if n > 0:
                    p_list += [input_points]
                    c_list += [np.hstack((input_colors, input_points + pick_point))]
                    pl_list += [input_labels]
                    pi_list += [input_inds]
                    ci_list += [cloud_ind]

                # Update batch size
                batch_n += n

            if batch_n > 0:
                yield (np.concatenate(p_list, axis=0),
                       np.concatenate(c_list, axis=0),
                       np.concatenate(pl_list, axis=0),
                       np.array([tp.shape[0] for tp in p_list]),
                       np.concatenate(pi_list, axis=0),
                       np.array(ci_list, dtype=np.int32))

        # Define the generator that should be used for this split
        if split == 'training':
            gen_func = spatially_regular_gen
        elif split == 'validation':
            gen_func = spatially_regular_gen
        elif split == 'test':
            gen_func = spatially_regular_gen
        else:
            raise ValueError('Split argument in data generator should be "training", "validation" or "test"')

        # Define generated types and shapes
        gen_types = (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32, tf.int32)
        gen_shapes = ([None, 3], [None, 6], [None], [None], [None], [None])

        return gen_func, gen_types, gen_shapes

    def get_tf_mapping(self):

        # Returned mapping function
        def tf_map(stacked_points, stacked_colors, point_labels, stacks_lengths, point_inds, cloud_inds):
            """
            [None, 3], [None, 3], [None], [None]
            """

            # Get batch indice for each point
            batch_inds = self.tf_get_batch_inds(stacks_lengths)

            # Augment input points
            stacked_points, scales, rots = self.tf_augment_input(stacked_points,
                                                                 batch_inds)

            # First add a column of 1 as feature for the network to be able to learn 3D shapes
            stacked_features = tf.ones((tf.shape(stacked_points)[0], 1), dtype=tf.float32)

            # Get coordinates and colors
            stacked_original_coordinates = stacked_colors[:, 3:]
            stacked_colors = stacked_colors[:, :3]

            # Augmentation : randomly drop colors
            if self.in_features_dim in [4, 5]:
                num_batches = batch_inds[-1] + 1
                s = tf.cast(tf.less(tf.random_uniform((num_batches,)), self.augment_color), tf.float32)
                stacked_s = tf.gather(s, batch_inds)
                stacked_colors = stacked_colors * tf.expand_dims(stacked_s, axis=1)

            # Then use positions or not
            if self.in_features_dim == 1:
                pass
            elif self.in_features_dim == 2:
                stacked_features = tf.concat((stacked_features, stacked_original_coordinates[:, 2:]), axis=1)
            elif self.in_features_dim == 3:
                stacked_features = stacked_colors
            elif self.in_features_dim == 4:
                stacked_features = tf.concat((stacked_features, stacked_colors), axis=1)
            elif self.in_features_dim == 5:
                stacked_features = tf.concat((stacked_features, stacked_colors, stacked_original_coordinates[:, 2:]),
                                             axis=1)
            elif self.in_features_dim == 7:
                stacked_features = tf.concat((stacked_features, stacked_colors, stacked_points), axis=1)
            elif self.in_features_dim == 8:
                stacked_features = tf.concat(
                    (stacked_features, stacked_colors, stacked_points, stacked_original_coordinates[:, 2:]), axis=1)
            else:
                raise ValueError('Only accepted input dimensions are 1, 3, 4 and 7 (without and with rgb/xyz)')

            # Get the whole input list
            input_list = self.tf_segmentation_inputs(self.downsample_times,
                                                     self.first_subsampling_dl,
                                                     self.density_parameter,
                                                     stacked_points,
                                                     stacked_features,
                                                     point_labels,
                                                     stacks_lengths,
                                                     batch_inds)

            # Add scale and rotation for testing
            input_list += [scales, rots]
            input_list += [point_inds, cloud_inds]

            return input_list

        return tf_map

    def load_evaluation_points(self, file_path):
        """
        Load points (from test or validation split) on which the metrics should be evaluated
        """

        # Get original points
        data = read_ply(file_path)
        return np.vstack((data['x'], data['y'], data['z'])).T

    def calibrate_batches(self):
        if len(self.input_trees['training']) > 0:
            split = 'training'
        else:
            split = 'test'
        N = (10000 // len(self.input_trees[split])) + 1
        sizes = []
        # Take a bunch of example neighborhoods in all clouds
        for i, tree in enumerate(self.input_trees[split]):
            # Randomly pick points
            points = np.array(tree.data, copy=False)
            rand_inds = np.random.choice(points.shape[0], size=N, replace=False)
            rand_points = points[rand_inds]
            noise = np.random.normal(scale=self.in_radius / 4, size=rand_points.shape)
            rand_points += noise.astype(rand_points.dtype)
            neighbors = tree.query_radius(points[rand_inds], r=self.in_radius)
            # Only save neighbors lengths
            sizes += [len(neighb) for neighb in neighbors]
        sizes = np.sort(sizes)
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

    def calibrate_neighbors(self, keep_ratio=0.8, samples_threshold=10000):

        # Create a tensorflow input pipeline
        # **********************************
        if len(self.input_trees['training']) > 0:
            split = 'training'
        else:
            split = 'test'

        # Get mapping function
        gen_function, gen_types, gen_shapes = self.get_batch_gen(split)
        map_func = self.get_tf_mapping()

        # Create batched dataset from generator
        train_data = tf.data.Dataset.from_generator(gen_function,
                                                    gen_types,
                                                    gen_shapes)

        train_data = train_data.map(map_func=map_func, num_parallel_calls=self.num_threads)

        # Prefetch data
        train_data = train_data.prefetch(10)

        # create a iterator of the correct shape and type
        iter = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
        flat_inputs = iter.get_next()

        # create the initialisation operations
        train_init_op = iter.make_initializer(train_data)

        hist_n = int(np.ceil(4 / 3 * np.pi * (self.density_parameter + 1) ** 3))

        # Create a local session for the calibration.
        cProto = tf.ConfigProto()
        cProto.gpu_options.allow_growth = True
        with tf.Session(config=cProto) as sess:

            # Init variables
            sess.run(tf.global_variables_initializer())

            # Initialise iterator with train data
            sess.run(train_init_op)

            # Get histogram of neighborhood sizes in 1 epoch max
            # **************************************************

            neighb_hists = np.zeros((self.num_layers, hist_n), dtype=np.int32)
            t0 = time.time()
            mean_dt = np.zeros(2)
            last_display = t0
            epoch = 0
            training_step = 0
            while epoch < 1 and np.min(np.sum(neighb_hists, axis=1)) < samples_threshold:
                try:

                    # Get next inputs
                    t = [time.time()]
                    ops = flat_inputs[self.num_layers:2 * self.num_layers]
                    neighbors = sess.run(ops)
                    t += [time.time()]

                    # Update histogram
                    counts = [np.sum(neighb_mat < neighb_mat.shape[0], axis=1) for neighb_mat in neighbors]
                    hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
                    neighb_hists += np.vstack(hists)
                    t += [time.time()]

                    # Average timing
                    mean_dt = 0.01 * mean_dt + 0.99 * (np.array(t[1:]) - np.array(t[:-1]))

                    # Console display
                    if (t[-1] - last_display) > 1.0:
                        last_display = t[-1]
                        message = 'Calib Neighbors {:08d} : timings {:4.2f} {:4.2f}'
                        print(message.format(training_step,
                                             1000 * mean_dt[0],
                                             1000 * mean_dt[1]))

                    training_step += 1

                except tf.errors.OutOfRangeError:
                    print('End of train dataset')
                    epoch += 1

            cumsum = np.cumsum(neighb_hists.T, axis=0)
            percentiles = np.sum(cumsum < (keep_ratio * cumsum[hist_n - 1, :]), axis=0)

            self.neighborhood_limits = percentiles
            print('neighborhood_limits : {}'.format(self.neighborhood_limits))

        return

    def tf_segmentation_inputs(self,
                               downsample_times,
                               first_subsampling_dl,
                               density_parameter,
                               stacked_points,
                               stacked_features,
                               point_labels,
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
        # list of network inputs
        li = input_points + input_neighbors + input_pools + input_upsamples
        li += [stacked_features, stacked_weights, stacked_batch_inds_0, stacked_batch_inds_1]
        li += [point_labels]
        return li


if __name__ == '__main__':
    from utils.config import config

    config.num_gpus = 1
    config.first_subsampling_dl = 0.04
    config.epoch_steps = 500
    config.validation_size = 50
    config.in_radius = 2.0
    config.augment_scale_anisotropic = True
    config.augment_symmetries = [True, False, False]
    config.augment_rotation = 'vertical'
    config.augment_scale_min = 0.8
    config.augment_scale_max = 1.2
    config.augment_noise = 0.001
    config.augment_color = 0.8

    dataset = S3DISDataset(config, 16)

    sess = tf.Session()
    sess.run(dataset.train_init_op)
    tic = time.time()
    total_n = 0.0
    total_i = 0.0
    while True:
        try:
            np_flat_inputs = sess.run(dataset.flat_inputs[0])
        except tf.errors.OutOfRangeError:
            break
    print(f"avg: {total_n / total_i}")
