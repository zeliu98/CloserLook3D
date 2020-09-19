import torch
import torch.utils.data as data
import numpy as np
import os
import sys
import pickle
from sklearn.neighbors import KDTree
from .data_utils import grid_subsampling

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)


def pc_normalize(pc):
    # Center and rescale point for 1m radius
    pmin = np.min(pc, axis=0)
    pmax = np.max(pc, axis=0)
    pc -= (pmin + pmax) / 2
    scale = np.max(np.linalg.norm(pc, axis=1))
    pc *= 1.0 / scale

    return pc


def get_scene_seg_features(input_features_dim, pc, color, height):
    if input_features_dim == 1:
        features = height
    elif input_features_dim == 3:
        features = color
    elif input_features_dim == 4:
        features = torch.cat([color, height], -1)
    elif input_features_dim == 5:
        ones = torch.ones_like(height)
        features = torch.cat([ones, color, height], -1)
    elif input_features_dim == 6:
        features = torch.cat([color, pc], -1)
    elif input_features_dim == 7:
        features = torch.cat([color, height, pc], -1)
    else:
        raise NotImplementedError("error")
    return features.transpose(0, 1).contiguous()


class S3DISSeg(data.Dataset):
    def __init__(self, input_features_dim, subsampling_parameter,
                 in_radius, num_points, num_steps, num_epochs,
                 color_drop=0, data_root=None, transforms=None, split='train'):
        """S3DIS dataset for scene segmentation task.

        Args:
            input_features_dim: input features dimensions, used to choose input feature type
            subsampling_parameter: grid length for pre-subsampling point clouds.
            in_radius: radius of each input spheres.
            num_points: max number of points for the input spheres.
            num_steps: number of spheres for one training epoch.
            num_epochs: total epochs.
            color_drop: probability ratio for random color dropping.
            data_root: root path for data.
            transforms: data transformations.
            split: dataset split name.
        """
        super().__init__()
        self.epoch = 0
        self.input_features_dim = input_features_dim
        self.transforms = transforms
        self.subsampling_parameter = subsampling_parameter
        self.color_drop = color_drop
        self.in_radius = in_radius
        self.num_points = num_points
        self.num_steps = num_steps
        self.num_epochs = num_epochs
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
        self.name_to_label = {v: k for k, v in self.label_to_names.items()}
        self.train_clouds = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_6']
        self.val_clouds = ['Area_5']
        if split == 'train':
            self.cloud_names = self.train_clouds
        elif split == 'val':
            self.cloud_names = self.val_clouds
        else:
            self.cloud_names = self.val_clouds + self.train_clouds

        self.color_mean = np.array([0.5136457, 0.49523646, 0.44921124])
        self.color_std = np.array([0.18308958, 0.18415008, 0.19252081])

        if data_root is None:
            self.data_root = os.path.join(ROOT_DIR, 'data')
        else:
            self.data_root = data_root
        if not os.path.exists(self.data_root):
            os.makedirs(self.data_root)
        self.folder = 'S3DIS'
        self.data_dir = os.path.join(self.data_root, self.folder, 'Stanford3dDataset_v1.2', 'processed')
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        # prepare data
        filename = os.path.join(self.data_dir, f'{split}_{subsampling_parameter:.3f}_data.pkl')
        if not os.path.exists(filename):
            cloud_points_list, cloud_points_color_list, cloud_points_label_list = [], [], []
            sub_cloud_points_list, sub_cloud_points_label_list, sub_cloud_points_color_list = [], [], []
            sub_cloud_tree_list = []

            for cloud_idx, cloud_name in enumerate(self.cloud_names):
                # Pass if the cloud has already been computed
                cloud_file = os.path.join(self.data_dir, cloud_name + '.pkl')
                if os.path.exists(cloud_file):
                    with open(cloud_file, 'rb') as f:
                        cloud_points, cloud_colors, cloud_classes = pickle.load(f)
                else:
                    # Get rooms of the current cloud
                    cloud_folder = os.path.join(self.data_root, self.folder, 'Stanford3dDataset_v1.2', cloud_name)
                    room_folders = [os.path.join(cloud_folder, room) for room in os.listdir(cloud_folder) if
                                    os.path.isdir(os.path.join(cloud_folder, room))]
                    # Initiate containers
                    cloud_points = np.empty((0, 3), dtype=np.float32)
                    cloud_colors = np.empty((0, 3), dtype=np.float32)
                    cloud_classes = np.empty((0, 1), dtype=np.int32)
                    # Loop over rooms
                    for i, room_folder in enumerate(room_folders):
                        print(
                            'Cloud %s - Room %d/%d : %s' % (
                                cloud_name, i + 1, len(room_folders), room_folder.split('\\')[-1]))

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
                    with open(cloud_file, 'wb') as f:
                        pickle.dump((cloud_points, cloud_colors, cloud_classes), f)
                cloud_points_list.append(cloud_points)
                cloud_points_color_list.append(cloud_colors)
                cloud_points_label_list.append(cloud_classes)

                sub_cloud_file = os.path.join(self.data_dir, cloud_name + f'_{subsampling_parameter:.3f}_sub.pkl')
                if os.path.exists(sub_cloud_file):
                    with open(sub_cloud_file, 'rb') as f:
                        sub_points, sub_colors, sub_labels, search_tree = pickle.load(f)
                else:
                    if subsampling_parameter > 0:
                        sub_points, sub_colors, sub_labels = grid_subsampling(cloud_points,
                                                                              features=cloud_colors,
                                                                              labels=cloud_classes,
                                                                              sampleDl=subsampling_parameter)
                        sub_colors /= 255.0
                        sub_labels = np.squeeze(sub_labels)
                    else:
                        sub_points = cloud_points
                        sub_colors = cloud_colors / 255.0
                        sub_labels = cloud_classes

                    # Get chosen neighborhoods
                    search_tree = KDTree(sub_points, leaf_size=50)

                    with open(sub_cloud_file, 'wb') as f:
                        pickle.dump((sub_points, sub_colors, sub_labels, search_tree), f)

                sub_cloud_points_list.append(sub_points)
                sub_cloud_points_color_list.append(sub_colors)
                sub_cloud_points_label_list.append(sub_labels)
                sub_cloud_tree_list.append(search_tree)

            self.clouds_points = cloud_points_list
            self.clouds_points_colors = cloud_points_color_list
            self.clouds_points_labels = cloud_points_label_list

            self.sub_clouds_points = sub_cloud_points_list
            self.sub_clouds_points_colors = sub_cloud_points_color_list
            self.sub_clouds_points_labels = sub_cloud_points_label_list
            self.sub_cloud_trees = sub_cloud_tree_list

            with open(filename, 'wb') as f:
                pickle.dump((self.clouds_points, self.clouds_points_colors, self.clouds_points_labels,
                             self.sub_clouds_points, self.sub_clouds_points_colors, self.sub_clouds_points_labels,
                             self.sub_cloud_trees), f)
                print(f"{filename} saved successfully")
        else:
            with open(filename, 'rb') as f:
                (self.clouds_points, self.clouds_points_colors, self.clouds_points_labels,
                 self.sub_clouds_points, self.sub_clouds_points_colors, self.sub_clouds_points_labels,
                 self.sub_cloud_trees) = pickle.load(f)
                print(f"{filename} loaded successfully")

        # prepare iteration indices
        filename = os.path.join(self.data_dir,
                                f'{split}_{subsampling_parameter:.3f}_{self.num_epochs}_{self.num_steps}_iterinds.pkl')
        if not os.path.exists(filename):
            potentials = []
            min_potentials = []
            for cloud_i, tree in enumerate(self.sub_cloud_trees):
                print(f"{split}/{cloud_i} has {tree.data.shape[0]} points")
                cur_potential = np.random.rand(tree.data.shape[0]) * 1e-3
                potentials.append(cur_potential)
                min_potentials.append(float(np.min(cur_potential)))
            self.cloud_inds = []
            self.point_inds = []
            self.noise = []
            for ep in range(self.num_epochs):
                for st in range(self.num_steps):
                    cloud_ind = int(np.argmin(min_potentials))
                    point_ind = np.argmin(potentials[cloud_ind])
                    print(f"[{ep}/{st}]: {cloud_ind}/{point_ind}")
                    self.cloud_inds.append(cloud_ind)
                    self.point_inds.append(point_ind)
                    points = np.array(self.sub_cloud_trees[cloud_ind].data, copy=False)
                    center_point = points[point_ind, :].reshape(1, -1)
                    noise = np.random.normal(scale=self.in_radius / 10, size=center_point.shape)
                    self.noise.append(noise)
                    pick_point = center_point + noise.astype(center_point.dtype)
                    # Indices of points in input region
                    query_inds = self.sub_cloud_trees[cloud_ind].query_radius(pick_point,
                                                                              r=self.in_radius,
                                                                              return_distance=True,
                                                                              sort_results=True)[0][0]
                    cur_num_points = query_inds.shape[0]
                    if self.num_points < cur_num_points:
                        query_inds = query_inds[:self.num_points]
                    # Update potentials (Tuckey weights)
                    dists = np.sum(np.square((points[query_inds] - pick_point).astype(np.float32)), axis=1)
                    tukeys = np.square(1 - dists / np.square(self.in_radius))
                    tukeys[dists > np.square(self.in_radius)] = 0
                    potentials[cloud_ind][query_inds] += tukeys
                    min_potentials[cloud_ind] = float(np.min(potentials[cloud_ind]))
                    # print(f"====>potentials: {potentials}")
                    print(f"====>min_potentials: {min_potentials}")
            with open(filename, 'wb') as f:
                pickle.dump((self.cloud_inds, self.point_inds, self.noise), f)
                print(f"{filename} saved successfully")
        else:
            with open(filename, 'rb') as f:
                self.cloud_inds, self.point_inds, self.noise = pickle.load(f)
                print(f"{filename} loaded successfully")

        # prepare validation projection inds
        filename = os.path.join(self.data_dir, f'{split}_{subsampling_parameter:.3f}_proj.pkl')
        if not os.path.exists(filename):
            proj_ind_list = []
            for points, search_tree in zip(self.clouds_points, self.sub_cloud_trees):
                proj_inds = np.squeeze(search_tree.query(points, return_distance=False))
                proj_inds = proj_inds.astype(np.int32)
                proj_ind_list.append(proj_inds)
            self.projections = proj_ind_list
            with open(filename, 'wb') as f:
                pickle.dump(self.projections, f)
                print(f"{filename} saved successfully")
        else:
            with open(filename, 'rb') as f:
                self.projections = pickle.load(f)
                print(f"{filename} loaded successfully")

    def __getitem__(self, idx):
        """
        Returns:
            current_points: (N, 3), a point cloud.
            mask: (N, ), 0/1 mask to distinguish padding points.
            features: (input_features_dim, N), input points features.
            current_points_labels: (N), point label.
            current_cloud_index: (1), cloud index.
            input_inds: (N), the index of input points in point cloud.
        """
        cloud_ind = self.cloud_inds[idx + self.epoch * self.num_steps]
        point_ind = self.point_inds[idx + self.epoch * self.num_steps]
        noise = self.noise[idx + self.epoch * self.num_steps]
        points = np.array(self.sub_cloud_trees[cloud_ind].data, copy=False)
        center_point = points[point_ind, :].reshape(1, -1)
        pick_point = center_point + noise.astype(center_point.dtype)
        # Indices of points in input region
        query_inds = self.sub_cloud_trees[cloud_ind].query_radius(pick_point,
                                                                  r=self.in_radius,
                                                                  return_distance=True,
                                                                  sort_results=True)[0][0]
        # Number collected
        cur_num_points = query_inds.shape[0]
        if self.num_points < cur_num_points:
            # choice = np.random.choice(cur_num_points, self.num_points)
            # input_inds = query_inds[choice]
            shuffle_choice = np.random.permutation(np.arange(self.num_points))
            input_inds = query_inds[:self.num_points][shuffle_choice]
            mask = torch.ones(self.num_points).type(torch.int32)
        else:
            shuffle_choice = np.random.permutation(np.arange(cur_num_points))
            query_inds = query_inds[shuffle_choice]
            padding_choice = np.random.choice(cur_num_points, self.num_points - cur_num_points)
            input_inds = np.hstack([query_inds, query_inds[padding_choice]])
            mask = torch.zeros(self.num_points).type(torch.int32)
            mask[:cur_num_points] = 1

        original_points = points[input_inds]
        current_points = (original_points - pick_point).astype(np.float32)
        current_points_height = original_points[:, 2:]
        current_points_height = torch.from_numpy(current_points_height).type(torch.float32)

        current_colors = self.sub_clouds_points_colors[cloud_ind][input_inds]
        current_colors = (current_colors - self.color_mean) / self.color_std
        current_colors = torch.from_numpy(current_colors).type(torch.float32)

        current_colors_drop = (torch.rand(1) > self.color_drop).type(torch.float32)
        current_colors = (current_colors * current_colors_drop).type(torch.float32)
        current_points_labels = torch.from_numpy(self.sub_clouds_points_labels[cloud_ind][input_inds]).type(torch.int64)
        current_cloud_index = torch.from_numpy(np.array(cloud_ind)).type(torch.int64)

        if self.transforms is not None:
            current_points = self.transforms(current_points)

        features = get_scene_seg_features(self.input_features_dim, current_points, current_colors,
                                          current_points_height)

        output_list = [current_points, mask, features,
                       current_points_labels, current_cloud_index, input_inds]
        return output_list

    def __len__(self):
        return self.num_steps


if __name__ == "__main__":
    import data_utils as d_utils
    from torchvision import transforms

    transforms = transforms.Compose(
        [
            d_utils.PointcloudToTensor(),
        ]
    )
    dset = S3DISSeg(4, in_radius=2.0, subsampling_parameter=0.04, num_points=15000,
                    num_steps=2000, num_epochs=600, split='train', transforms=transforms)
    dset_v = S3DISSeg(4, in_radius=2.0, subsampling_parameter=0.04, num_points=15000,
                      num_steps=2000, num_epochs=20, split='val', transforms=transforms)

    print(dset[0][0])
    print(dset[0][1])
    print(dset[0][2])
    print(dset[0][3])
    print(dset[0][4])
    print(dset[0][5])
    print(len(dset))
