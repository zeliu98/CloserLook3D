import os
import torch
import json
import shlex
import pickle
import subprocess
import numpy as np
import torch.utils.data as data

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


class ShapeNetPartSeg(data.Dataset):
    def __init__(self, num_points, data_root=None, transforms=None, split='train', download=True):
        self.transforms = transforms
        self.num_points = num_points
        self.split = split
        self.label_to_names = {0: 'Airplane',
                               1: 'Bag',
                               2: 'Cap',
                               3: 'Car',
                               4: 'Chair',
                               5: 'Earphone',
                               6: 'Guitar',
                               7: 'Knife',
                               8: 'Lamp',
                               9: 'Laptop',
                               10: 'Motorbike',
                               11: 'Mug',
                               12: 'Pistol',
                               13: 'Rocket',
                               14: 'Skateboard',
                               15: 'Table'}

        self.name_to_label = {v: k for k, v in self.label_to_names.items()}
        self.num_parts = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]

        self.data_root = data_root
        if data_root is None:
            self.data_root = os.path.join(ROOT_DIR, 'data')
        else:
            self.data_root = data_root
        if not os.path.exists(self.data_root):
            os.makedirs(self.data_root)
        self.folder = "ShapeNetPart"
        self.data_dir = os.path.join(self.data_root, self.folder, 'shapenetcore_partanno_segmentation_benchmark_v0')
        self.url = "https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0.zip"

        if download and not os.path.exists(self.data_dir):
            zipfile = os.path.join(self.data_root, os.path.basename(self.url))
            subprocess.check_call(
                shlex.split("curl {} -o {}".format(self.url, zipfile))
            )

            subprocess.check_call(
                shlex.split("unzip {} -d {}".format(zipfile, self.data_root))
            )

            subprocess.check_call(shlex.split("rm {}".format(zipfile)))

        self.category_and_synsetoffset = [['Airplane', '02691156'],
                                          ['Bag', '02773838'],
                                          ['Cap', '02954340'],
                                          ['Car', '02958343'],
                                          ['Chair', '03001627'],
                                          ['Earphone', '03261776'],
                                          ['Guitar', '03467517'],
                                          ['Knife', '03624134'],
                                          ['Lamp', '03636649'],
                                          ['Laptop', '03642806'],
                                          ['Motorbike', '03790512'],
                                          ['Mug', '03797390'],
                                          ['Pistol', '03948459'],
                                          ['Rocket', '04099429'],
                                          ['Skateboard', '04225987'],
                                          ['Table', '04379243']]
        synsetoffset_to_category = {s: n for n, s in self.category_and_synsetoffset}

        # Train split
        split_file = os.path.join(self.data_dir, 'train_test_split', 'shuffled_train_file_list.json')
        with open(split_file, 'r') as f:
            train_files = json.load(f)
        train_files = [name[11:] for name in train_files]

        # Val split
        split_file = os.path.join(self.data_dir, 'train_test_split', 'shuffled_val_file_list.json')
        with open(split_file, 'r') as f:
            val_files = json.load(f)
        val_files = [name[11:] for name in val_files]

        # Test split
        split_file = os.path.join(self.data_dir, 'train_test_split', 'shuffled_test_file_list.json')
        with open(split_file, 'r') as f:
            test_files = json.load(f)
        test_files = [name[11:] for name in test_files]

        split_files = {'train': train_files,
                       'trainval': train_files + val_files,
                       'val': val_files,
                       'test': test_files
                       }
        files = split_files[split]
        filename = os.path.join(self.data_root, self.folder, '{}_data.pkl'.format(split))
        if not os.path.exists(filename):
            point_list = []
            points_label_list = []
            label_list = []
            for i, file in enumerate(files):
                # Get class
                synset = file.split('/')[0]
                class_name = synsetoffset_to_category[synset]
                cls = self.name_to_label[class_name]
                cls = np.array(cls)
                # Get filename
                file_name = file.split('/')[1]
                # Load points and labels
                point_set = np.loadtxt(os.path.join(self.data_dir, synset, 'points', file_name + '.pts')).astype(
                    np.float32)
                point_set = pc_normalize(point_set)
                seg = np.loadtxt(os.path.join(self.data_dir, synset, 'points_label', file_name + '.seg')).astype(
                    np.int64) - 1
                point_list.append(point_set)
                points_label_list.append(seg)
                label_list.append(cls)
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

        print(f"split:{split} had {len(self.points)} data")

    def __getitem__(self, idx):
        current_points = self.points[idx]
        current_points_labels = self.points_labels[idx]
        cur_num_points = current_points.shape[0]
        if cur_num_points >= self.num_points:
            choice = np.random.choice(cur_num_points, self.num_points)
            current_points = current_points[choice, :]
            current_points_labels = current_points_labels[choice]
            mask = torch.ones(self.num_points).type(torch.int32)
        else:
            padding_num = self.num_points - cur_num_points
            shuffle_choice = np.random.permutation(np.arange(cur_num_points))
            padding_choice = np.random.choice(cur_num_points, padding_num)
            choice = np.hstack([shuffle_choice, padding_choice])
            current_points = current_points[choice, :]
            current_points_labels = current_points_labels[choice]
            mask = torch.cat([torch.ones(cur_num_points), torch.zeros(padding_num)]).type(torch.int32)

        label = torch.from_numpy(self.labels[idx]).type(torch.int64)
        current_points_labels = torch.from_numpy(current_points_labels).type(torch.int64)
        if self.transforms is not None:
            current_points = self.transforms(current_points)

        return current_points, mask, current_points_labels, label

    def __len__(self):
        return len(self.points)
