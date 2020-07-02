import torch
import torch.utils.data as data
import numpy as np
import os
import sys
import subprocess
import shlex
import pickle
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


def get_cls_features(input_features_dim, pc, normal=None):
    if input_features_dim == 3:
        features = pc
    elif input_features_dim == 4:
        features = torch.ones(size=(pc.shape[0], 1), dtype=torch.float32)
        features = torch.cat([features, pc], -1)
    elif input_features_dim == 6:
        features = torch.cat([pc, normal])
    elif input_features_dim == 7:
        features = torch.ones(size=(pc.shape[0], 1), dtype=torch.float32)
        features = torch.cat([features, pc, normal], -1)
    else:
        raise NotImplementedError("error")
    return features.transpose(0, 1).contiguous()


class ModelNet40Cls(data.Dataset):
    def __init__(self, input_features_dim, num_points,
                 data_root=None, transforms=None, split='train',
                 subsampling_parameter=0.02, download=False):
        """ModelNet40 dataset for shape classification.

        Args:
            input_features_dim: input features dimensions, used to choose input feature type
            num_points: max number of points for the input point cloud.
            data_root: root path for data.
            transforms: data transformations.
            split: dataset split name.
            subsampling_parameter: grid length for pre-subsampling point clouds.
            download: whether downloading when dataset don't exists.
        """
        super().__init__()
        self.num_points = num_points
        self.input_features_dim = input_features_dim
        self.transforms = transforms
        self.use_normal = (input_features_dim >= 6)
        self.subsampling_parameter = subsampling_parameter
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
        self.name_to_label = {v: k for k, v in self.label_to_names.items()}

        if data_root is None:
            self.data_root = os.path.join(ROOT_DIR, 'data')
        else:
            self.data_root = data_root
        if not os.path.exists(self.data_root):
            os.makedirs(self.data_root)
        self.folder = 'ModelNet40'
        self.data_dir = os.path.join(self.data_root, self.folder, 'modelnet40_normal_resampled')
        self.url = "https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip"

        if download and not os.path.exists(self.data_dir):
            zipfile = os.path.join(self.data_root, os.path.basename(self.url))
            subprocess.check_call(
                shlex.split("curl {} -o {}".format(self.url, zipfile))
            )

            subprocess.check_call(
                shlex.split("unzip {} -d {}".format(zipfile, os.path.join(self.data_root, self.folder)))
            )

            subprocess.check_call(shlex.split("rm {}".format(zipfile)))

        # Collect test file names
        if split == 'train':
            names = np.loadtxt(os.path.join(self.data_dir, 'modelnet40_train.txt'), dtype=np.str)
        elif split == 'test':
            names = np.loadtxt(os.path.join(self.data_dir, 'modelnet40_test.txt'), dtype=np.str)
        else:
            raise KeyError(f"ModelNet40 has't split: {split}")

        filename = os.path.join(self.data_root, self.folder, '{}_{:.3f}_data.pkl'.format(split, subsampling_parameter))
        if not os.path.exists(filename):
            print(f"Preparing ModelNet40 data with subsampling_parameter={subsampling_parameter}")
            point_list, normal_list, label_list = [], [], []
            # Collect point clouds
            for i, cloud_name in enumerate(names):
                # Read points
                class_folder = '_'.join(cloud_name.split('_')[:-1])
                txt_file = os.path.join(self.data_dir, class_folder, cloud_name) + '.txt'
                data = np.loadtxt(txt_file, delimiter=',', dtype=np.float32)
                pc = data[:, :3]
                pc = pc_normalize(pc)
                normal = data[:, 3:]
                label = np.array(self.name_to_label[class_folder])
                # Subsample
                if subsampling_parameter > 0:
                    pc, normal = grid_subsampling(pc, features=normal, sampleDl=subsampling_parameter)

                point_list.append(pc)
                normal_list.append(normal)
                label_list.append(label)
            self.points = point_list
            self.normals = normal_list
            self.labels = label_list

            with open(filename, 'wb') as f:
                pickle.dump((self.points, self.normals, self.labels), f)
                print(f"{filename} saved successfully")
        else:
            with open(filename, 'rb') as f:
                self.points, self.normals, self.labels = pickle.load(f)
                print(f"{filename} loaded successfully")

        print(f"{split} dataset has {len(self.points)} data with  {num_points} points")

    def __getitem__(self, idx):
        """
        Returns:
            pc: (N, 3), a point cloud.
            mask: (N, ), 0/1 mask to distinguish padding points.
            features: (input_features_dim, N), input points features.
            label: (1), class label.
        """
        current_points = self.points[idx]
        cur_num_points = current_points.shape[0]
        if cur_num_points >= self.num_points:
            choice = np.random.choice(cur_num_points, self.num_points)
            current_points = current_points[choice, :]
            if self.use_normal:
                current_normals = self.normals[idx]
                current_normals = current_normals[choice, :]
            mask = torch.ones(self.num_points).type(torch.int32)
        else:
            padding_num = self.num_points - cur_num_points
            shuffle_choice = np.random.permutation(np.arange(cur_num_points))
            padding_choice = np.random.choice(cur_num_points, padding_num)
            choice = np.hstack([shuffle_choice, padding_choice])
            current_points = current_points[choice, :]
            if self.use_normal:
                current_normals = self.normals[idx]
                current_normals = current_normals[choice, :]
            mask = torch.cat([torch.ones(cur_num_points), torch.zeros(padding_num)]).type(torch.int32)

        label = torch.from_numpy(self.labels[idx]).type(torch.int64)

        if self.use_normal:
            current_points = np.hstack([current_points, current_normals])

        if self.transforms is not None:
            current_points = self.transforms(current_points)
        pc = current_points[:, :3]
        normal = current_points[:, 3:]
        features = get_cls_features(self.input_features_dim, pc, normal)

        return pc, mask, features, label

    def __len__(self):
        return len(self.points)


if __name__ == "__main__":
    import data_utils as d_utils
    from torchvision import transforms

    transforms = transforms.Compose(
        [
            d_utils.PointcloudToTensor(),
        ]
    )
    dset = ModelNet40Cls(3, 10000, split='train', transforms=transforms)
    dset_test = ModelNet40Cls(3, 10000, split='test', transforms=transforms)

    print(dset[0][0])
    print(dset[0][1])
    print(dset[0][2])
    print(dset[0][3])
    print(len(dset))
