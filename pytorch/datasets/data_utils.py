import os
import sys
import torch
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'ops'))
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling


def grid_subsampling(points, features=None, labels=None, sampleDl=0.1, verbose=0):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features
    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param sampleDl: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: subsampled points, with features and/or labels depending of the input
    """

    if (features is None) and (labels is None):
        return cpp_subsampling.compute(points, sampleDl=sampleDl, verbose=verbose)
    elif (labels is None):
        return cpp_subsampling.compute(points, features=features, sampleDl=sampleDl, verbose=verbose)
    elif (features is None):
        return cpp_subsampling.compute(points, classes=labels, sampleDl=sampleDl, verbose=verbose)
    else:
        return cpp_subsampling.compute(points, features=features, classes=labels, sampleDl=sampleDl, verbose=verbose)


def angle_axis(angle, axis):
    # type: (float, np.ndarray) -> float
    r"""Returns a 4x4 rotation matrix that performs a rotation around axis by angle

    Parameters
    ----------
    angle : float
        Angle to rotate by
    axis: np.ndarray
        Axis to rotate about

    Returns
    -------
    torch.Tensor
        3x3 rotation matrix
    """
    u = axis / np.linalg.norm(axis)
    cosval, sinval = np.cos(angle), np.sin(angle)

    # yapf: disable
    cross_prod_mat = np.array([[0.0, -u[2], u[1]],
                               [u[2], 0.0, -u[0]],
                               [-u[1], u[0], 0.0]])

    R = torch.from_numpy(
        cosval * np.eye(3)
        + sinval * cross_prod_mat
        + (1.0 - cosval) * np.outer(u, u)
    )
    # yapf: enable
    return R.float()


class PointcloudScale(object):
    def __init__(self, scale_low=0.8, scale_high=1.25):
        self.scale_low, self.scale_high = scale_low, scale_high

    def __call__(self, points):
        scaler = np.random.uniform(self.scale_low, self.scale_high, size=[3])
        scaler = torch.from_numpy(scaler).float()
        points[:, 0:3] *= scaler
        return points


class PointcloudRotate(object):
    def __init__(self, axis=np.array([0.0, 1.0, 0.0])):
        self.axis = axis

    def __call__(self, points):
        rotation_angle = np.random.uniform() * 2 * np.pi
        rotation_matrix = angle_axis(rotation_angle, self.axis)

        normals = points.size(1) > 3
        if not normals:
            return torch.matmul(points, rotation_matrix.t())
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]
            points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())
            points[:, 3:] = torch.matmul(pc_normals, rotation_matrix.t())

            return points


class PointcloudRotatePerturbation(object):
    def __init__(self, angle_sigma=0.06, angle_clip=0.18):
        self.angle_sigma, self.angle_clip = angle_sigma, angle_clip

    def _get_angles(self):
        angles = np.clip(
            self.angle_sigma * np.random.randn(3), -self.angle_clip, self.angle_clip
        )

        return angles

    def __call__(self, points):
        angles_ = self._get_angles()
        Rx = angle_axis(angles_[0], np.array([1.0, 0.0, 0.0]))
        Ry = angle_axis(angles_[1], np.array([0.0, 1.0, 0.0]))
        Rz = angle_axis(angles_[2], np.array([0.0, 0.0, 1.0]))

        rotation_matrix = torch.matmul(torch.matmul(Rz, Ry), Rx)

        normals = points.size(1) > 3
        if not normals:
            return torch.matmul(points, rotation_matrix.t())
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]
            points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())
            points[:, 3:] = torch.matmul(pc_normals, rotation_matrix.t())

            return points


class PointcloudRandomRotate(object):
    def __init__(self, x_range=np.pi, y_range=np.pi, z_range=np.pi):
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range

    def _get_angles(self):
        x_angle = np.random.uniform(-self.x_range, self.x_range)
        y_angle = np.random.uniform(-self.y_range, self.y_range)
        z_angle = np.random.uniform(-self.z_range, self.z_range)

        return np.array([x_angle, y_angle, z_angle])

    def __call__(self, points):
        angles_ = self._get_angles()
        Rx = angle_axis(angles_[0], np.array([1.0, 0.0, 0.0]))
        Ry = angle_axis(angles_[1], np.array([0.0, 1.0, 0.0]))
        Rz = angle_axis(angles_[2], np.array([0.0, 0.0, 1.0]))

        rotation_matrix = torch.matmul(torch.matmul(Rz, Ry), Rx)

        normals = points.size(1) > 3
        if not normals:
            return torch.matmul(points, rotation_matrix.t())
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]
            points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())
            points[:, 3:] = torch.matmul(pc_normals, rotation_matrix.t())

            return points


class PointcloudJitter(object):
    def __init__(self, std=0.01, clip=0.05):
        self.std, self.clip = std, clip

    def __call__(self, points):
        jittered_data = (
            points.new(points.size(0), 3)
                .normal_(mean=0.0, std=self.std)
                .clamp_(-self.clip, self.clip)
        )
        points[:, 0:3] += jittered_data
        return points


class PointcloudTranslate(object):
    def __init__(self, translate_range=0.1):
        self.translate_range = translate_range

    def __call__(self, points):
        translation = np.random.uniform(-self.translate_range, self.translate_range, size=[3])
        translation = torch.from_numpy(translation)
        points[:, 0:3] += translation
        return points


class PointcloudToTensor(object):
    def __call__(self, points):
        return torch.from_numpy(points).float()


class PointcloudRandomInputDropout(object):
    def __init__(self, max_dropout_ratio=0.875):
        assert max_dropout_ratio >= 0 and max_dropout_ratio < 1
        self.max_dropout_ratio = max_dropout_ratio

    def __call__(self, points):
        pc = points.numpy()

        dropout_ratio = np.random.random() * self.max_dropout_ratio  # 0~0.875
        drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            pc[drop_idx] = pc[0]  # set to the first point

        return torch.from_numpy(pc).float()


class PointcloudScaleAndTranslate(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2., translate_range=0.2):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_range = translate_range

    def __call__(self, pc):
        xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
        xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[3])
        pc[:, 0:3] = torch.mul(pc[:, 0:3], torch.from_numpy(xyz1).float()) + torch.from_numpy(
            xyz2).float()

        return pc


class PointcloudScaleAndJitter(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2., std=0.01, clip=0.05, augment_symmetries=[0, 0, 0]):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.std = std
        self.clip = clip
        self.augment_symmetries = augment_symmetries

    def __call__(self, pc):
        xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
        symmetries = np.round(np.random.uniform(low=0, high=1, size=[3])) * 2 - 1
        symmetries = symmetries * np.array(self.augment_symmetries) + (1 - np.array(self.augment_symmetries))
        xyz1 *= symmetries
        xyz2 = np.clip(np.random.normal(scale=self.std, size=[pc.shape[0], 3]), a_min=-self.clip, a_max=self.clip)
        pc[:, 0:3] = torch.mul(pc[:, 0:3], torch.from_numpy(xyz1).float()) + torch.from_numpy(
            xyz2).float()

        return pc


class BatchPointcloudScaleAndTranslate(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2., translate_range=0.2):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_range = translate_range

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[3])

            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().to(pc.device)) + torch.from_numpy(
                xyz2).float().to(pc.device)

        return pc


class BatchPointcloudScaleAndJitter(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2., std=0.01, clip=0.05, augment_symmetries=[0, 0, 0]):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.std, self.clip = std, clip
        self.augment_symmetries = augment_symmetries

    def __call__(self, pc):
        bsize = pc.size()[0]
        npoint = pc.size()[1]
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            symmetries = np.round(np.random.uniform(low=0, high=1, size=[3])) * 2 - 1
            symmetries = symmetries * np.array(self.augment_symmetries) + (1 - np.array(self.augment_symmetries))
            xyz1 *= symmetries
            xyz2 = np.clip(np.random.normal(scale=self.std, size=[npoint, 3]), a_max=self.clip, a_min=-self.clip)

            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().to(pc.device)) + torch.from_numpy(
                xyz2).float().to(pc.device)

        return pc


class BatchPointcloudRandomRotate(object):
    def __init__(self, x_range=np.pi, y_range=np.pi, z_range=np.pi):
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range

    def _get_angles(self):
        x_angle = np.random.uniform(-self.x_range, self.x_range)
        y_angle = np.random.uniform(-self.y_range, self.y_range)
        z_angle = np.random.uniform(-self.z_range, self.z_range)

        return np.array([x_angle, y_angle, z_angle])

    def __call__(self, pc):
        bsize = pc.size()[0]
        normals = pc.size()[2] > 3
        for i in range(bsize):
            angles_ = self._get_angles()
            Rx = angle_axis(angles_[0], np.array([1.0, 0.0, 0.0]))
            Ry = angle_axis(angles_[1], np.array([0.0, 1.0, 0.0]))
            Rz = angle_axis(angles_[2], np.array([0.0, 0.0, 1.0]))

            rotation_matrix = torch.matmul(torch.matmul(Rz, Ry), Rx).to(pc.device)

            if not normals:
                pc[i, :, 0:3] = torch.matmul(pc[i, :, 0:3], rotation_matrix.t())
            else:
                pc[i, :, 0:3] = torch.matmul(pc[i, :, 0:3], rotation_matrix.t())
                pc[i, :, 3:] = torch.matmul(pc[i, :, 3:], rotation_matrix.t())
        return pc
