import torch
import torch.utils.data as data
import numpy as np
import os
import sys
import h5py
import pickle

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


def get_part_seg_features(input_features_dim, pc):
    if input_features_dim == 3:
        features = pc
    elif input_features_dim == 4:
        features = torch.ones(size=(pc.shape[0], 1), dtype=torch.float32)
        features = torch.cat([features, pc], -1)
    else:
        raise NotImplementedError("error")
    return features.transpose(0, 1).contiguous()


class PartNetSeg(data.Dataset):
    def __init__(self, input_features_dim, data_root=None, transforms=None, split='train'):
        """PartNet dataset for fine-grained part segmentation task.

        Args:
            input_features_dim: input features dimensions, used to choose input feature type
            data_root: root path for data.
            transforms: data transformations.
            split: dataset split name.
        """
        super().__init__()
        self.input_features_dim = input_features_dim
        self.transforms = transforms
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
                               16: 'Vase'}
        self.name_to_label = {v: k for k, v in self.label_to_names.items()}
        self.num_parts = [15, 9, 39, 11, 7, 4, 5, 10, 12, 10, 41, 6, 7, 24, 51, 11, 6]
        self.shuffle_points = True if (split == 'train') else False

        if data_root is None:
            self.data_root = os.path.join(ROOT_DIR, 'data')
        else:
            self.data_root = data_root
        if not os.path.exists(self.data_root):
            os.makedirs(self.data_root)
        self.folder = 'PartNet'
        self.data_dir = os.path.join(self.data_root, self.folder, 'sem_seg_h5')

        filename = os.path.join(self.data_root, self.folder, '{}_data.pkl'.format(split))
        if not os.path.exists(filename):
            print(f"Preparing PartNet data")
            point_list, points_label_list, label_list = [], [], []
            for class_id, class_name in self.label_to_names.items():
                split_filelist = os.path.join(self.data_dir, '{}-{}'.format(class_name, 3),
                                              '{:s}_files.txt'.format(split))
                split_points, split_labels = self._load_seg(split_filelist)
                N = split_points.shape[0]
                for i in range(N):
                    pc = split_points[i]
                    pc = pc_normalize(pc)
                    pc = pc[:, [0, 2, 1]]
                    pcl = split_labels[i]
                    point_list.append(pc)
                    points_label_list.append(pcl)
                    label_list.append(np.array(class_id))

            self.points = point_list
            self.points_labels = points_label_list
            self.labels = label_list

            with open(filename, 'wb') as f:
                pickle.dump((self.points, self.points_labels, self.labels), f)
                print(f"{filename} saved successfully")
        else:
            with open(filename, 'rb') as f:
                self.points, self.points_labels, self.labels = pickle.load(f)
                print(f"{filename} loaded successfully")

        self.num_points = self.points[0].shape[0]
        print(f"{split} dataset has {len(self.points)} data with  {self.num_points} points")

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

    def __getitem__(self, idx):
        """
        Returns:
            current_points: (N, 3), a point cloud.
            mask: (N, ), 0/1 mask to distinguish padding points.
            features: (input_features_dim, N), input points features.
            current_points_labels: (N), point label.
            shape_labels: (1), shape label.
        """
        current_points = self.points[idx]
        current_points_labels = self.points_labels[idx]
        if self.shuffle_points:
            shuffle_choice = np.random.permutation(np.arange(self.num_points))
            current_points = current_points[shuffle_choice, :]
            current_points_labels = current_points_labels[shuffle_choice]
        current_points_labels = torch.from_numpy(current_points_labels).type(torch.int64)
        shape_labels = torch.from_numpy(self.labels[idx]).type(torch.int64)
        mask = torch.ones(self.num_points).type(torch.int32)

        if self.transforms is not None:
            current_points = self.transforms(current_points)

        features = get_part_seg_features(self.input_features_dim, current_points)

        return current_points, mask, features, current_points_labels, shape_labels

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
    dset = PartNetSeg(3, split='train', transforms=transforms)
    dset_v = PartNetSeg(3, split='val', transforms=transforms)
    dset_te = PartNetSeg(3, split='test', transforms=transforms)

    print(dset[0][0])
    print(dset[0][1])
    print(dset[0][2])
    print(dset[0][3])
    print(len(dset))
