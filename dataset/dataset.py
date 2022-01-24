from torch.utils.data import Dataset, Subset
import torch
import os
import utils
import cv2
import random
import numpy as np


class FlyingThings3D(Dataset):
    # ROOT = '/media/jack/data/Dataset/pytorch/flyingthings3d'
    ROOT = r'F:\Dataset\pytorch\flyingthings3d'

    # height, width = 540, 960

    def __init__(self, max_disparity, crop_size=None, type='train', image='cleanpass', crop_seed=None, down_sampling=1,
                 disparity=['left'], small=False):
        if small:
            self.ROOT += '_s'

        assert os.path.exists(self.ROOT), 'Dataset path is not exist'
        assert isinstance(down_sampling, int)
        self.down_sampling = down_sampling
        self.disparity = disparity
        self.data_max_disparity = []

        if type == 'train':
            for d in self.disparity:
                self.data_max_disparity.append(utils.load(os.path.join(self.ROOT, f'{d}_max_disparity.np'))[0])
            self.root = os.path.join(self.ROOT, 'TRAIN')

            if small:
                self.size = 7460
            else:
                self.size = 22390
        elif type == 'test':
            for d in self.disparity:
                self.data_max_disparity.append(utils.load(os.path.join(self.ROOT, f'{d}_max_disparity.np'))[1])
            self.root = os.path.join(self.ROOT, 'TEST')

            if small:
                self.size = 1440
            else:
                self.size = 4370

        else:
            raise Exception(f'Unknown type: "{type}"')

        self.mask = np.ones(self.size, dtype=np.uint8)
        for d in self.data_max_disparity:
            self.mask = self.mask & (d < max_disparity - 1)
        self.size = np.sum(self.mask)
        self.mask = torch.from_numpy(self.mask)

        if image not in ['cleanpass', 'finalpass']:
            raise Exception(f'Unknown image: "{image}"')

        self.image = image
        self.crop_size = crop_size
        self.crop_seed = crop_seed
        self._make_mask_index()

    def __getitem__(self, index):
        index = self.mask_index[index]
        X = utils.load(os.path.join(self.root, f'{self.image}/{index:05d}.np'))
        X = torch.from_numpy(X)

        if self.crop_size is not None:
            cropper = utils.RandomCropper(X.shape[1:3], self.crop_size, seed=self.crop_seed)
            X = cropper.crop(X)
        X = X.float() / 255

        Y_list = []
        for d in self.disparity:
            Y = utils.load(os.path.join(self.root, f'{d}_disparity/{index:05d}.np'))
            Y = torch.from_numpy(Y)
            if self.crop_size is not None:
                Y = cropper.crop(Y)
            Y_list.append(Y.unsqueeze(0))
        Y = torch.cat(Y_list, dim=0)

        if self.down_sampling != 1:
            X = X[:, ::self.down_sampling, ::self.down_sampling]
            Y = Y[:, ::self.down_sampling, ::self.down_sampling]
            Y /= self.down_sampling

        return X.cuda(), Y.cuda()

    def __len__(self):
        return self.size

    def _make_mask_index(self):
        self.mask_index = np.zeros(self.size, dtype=np.int)

        i = 0
        m = 0
        while i < len(self.mask):
            if self.mask[i]:
                self.mask_index[m] = i
                m += 1
            i += 1


class KITTI_2015(Dataset):
    ROOT = r'F:\Dataset\KITTI 2015'

    # KITTI 2015 original height and width (375, 1242, 3), dtype uint8
    # height and width: (370, 1224) is the smallest size

    # HEIGHT, WIDTH = 384, 1248
    # HEIGHT, WIDTH = 352, 1216  # GTX 2080 Ti
    # HEIGHT, WIDTH = 256, 1248  # GTX 1660 Ti

    def __init__(self, type='train', use_crop_size=False, crop_size=None, crop_seed=None, untexture_rate=0.1,
                 use_resize=False, resize=(None, None)):
        assert os.path.exists(self.ROOT), 'Dataset path is not exist'
        assert use_crop_size + use_resize <= 1, 'Using one of the crop size and the resize'

        self.type = type
        if type == 'train':
            self.root = os.path.join(self.ROOT, 'training')
        elif type == 'test':
            self.root = os.path.join(self.ROOT, 'testing')
        else:
            raise Exception('Unknown type "{}"'.format(type))

        self.use_crop_size = use_crop_size
        self.crop_size = crop_size
        self.crop_seed = crop_seed
        self.use_resize = use_resize
        self.resize_height, self.resize_width = resize
        self.untexture_rate = untexture_rate

    def __getitem__(self, index):
        if self.type == 'train':
            untexture_learning = random.randint(1, 100) <= int(self.untexture_rate * 100)
            if untexture_learning:
                bgr = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                X1 = np.full((375, 1242, 3), bgr, dtype=np.uint8)
                X2 = np.full((375, 1242, 3), bgr, dtype=np.uint8)
                Y = np.full((375, 1242), 0.001, dtype=np.float32)

                X = np.concatenate([X1, X2], axis=2)
                X = X.swapaxes(0, 2).swapaxes(1, 2)
                X, Y = torch.from_numpy(X).float(), torch.from_numpy(Y)
                Y = Y.unsqueeze(0)

                if self.use_crop_size:
                    cropper = utils.RandomCropper(X1.shape[0:2], self.crop_size, seed=self.crop_seed)
                    X, Y = cropper.crop(X), cropper.crop(Y)
                X, Y = X.float() / 255, Y.float()
                return X.cuda(), Y.cuda()
            else:
                if self.use_resize:
                    X1 = cv2.imread(os.path.join(self.root, 'image_2/{:06d}_10.png'.format(index)))
                    X2 = cv2.imread(os.path.join(self.root, 'image_3/{:06d}_10.png'.format(index)))
                    Y = cv2.imread(
                        os.path.join(self.root, 'disp_occ_0/{:06d}_10.png'.format(index)))  # (376, 1241, 3) uint8
                    self.original_height, self.original_width = X1.shape[:2]

                    X1 = cv2.resize(X1, (self.resize_width, self.resize_height))
                    X2 = cv2.resize(X2, (self.resize_width, self.resize_height))

                    X1 = utils.rgb2bgr(X1)
                    X2 = utils.rgb2bgr(X2)

                    X = np.concatenate([X1, X2], axis=2)  # batch, height, width, channel
                    X = X.swapaxes(0, 2).swapaxes(1, 2)  # channel*2, height, width

                    Y = Y[:, :, 0]
                    X, Y = torch.from_numpy(X).float() / 255, torch.from_numpy(Y)
                    Y = Y.unsqueeze(0)

                elif self.use_crop_size:
                    X1 = cv2.imread(os.path.join(self.root, 'image_2/{:06d}_10.png'.format(index)))
                    X2 = cv2.imread(os.path.join(self.root, 'image_3/{:06d}_10.png'.format(index)))
                    Y = cv2.imread(
                        os.path.join(self.root, 'disp_occ_0/{:06d}_10.png'.format(index)))  # (376, 1241, 3) uint8

                    X1 = utils.rgb2bgr(X1)
                    X2 = utils.rgb2bgr(X2)

                    X = np.concatenate([X1, X2], axis=2)
                    X = X.swapaxes(0, 2).swapaxes(1, 2)

                    Y = Y[:, :, 0]
                    X, Y = torch.from_numpy(X).float() / 255, torch.from_numpy(Y)
                    Y = Y.unsqueeze(0)

                    cropper = utils.RandomCropper(X1.shape[0:2], self.crop_size, seed=self.crop_seed)
                    X, Y = cropper.crop(X), cropper.crop(Y)

                return X.cuda(), Y.cuda()

        elif self.type == 'test':
            if self.use_resize:
                X1 = cv2.imread(os.path.join(self.root, 'image_2/{:06d}_10.png'.format(index)))
                X2 = cv2.imread(os.path.join(self.root, 'image_3/{:06d}_10.png'.format(index)))
                self.original_height, self.original_width = X1.shape[:2]

                X1 = cv2.resize(X1, (self.resize_width, self.resize_height))
                X2 = cv2.resize(X2, (self.resize_width, self.resize_height))

                X1 = utils.rgb2bgr(X1)
                X2 = utils.rgb2bgr(X2)

                X = np.concatenate([X1, X2], axis=2)  # batch, height, width, channel
                X = X.swapaxes(0, 2).swapaxes(1, 2)  # channel*2, height, width

                Y = torch.ones((1, X.size(1), X.size(2)), dtype=torch.float)

            elif self.use_crop_size:
                X1 = cv2.imread(os.path.join(self.root, 'image_2/{:06d}_10.png'.format(index)))
                X2 = cv2.imread(os.path.join(self.root, 'image_3/{:06d}_10.png'.format(index)))

                X1 = utils.rgb2bgr(X1)
                X2 = utils.rgb2bgr(X2)

                X = np.concatenate([X1, X2], axis=2)
                X = X.swapaxes(0, 2).swapaxes(1, 2)

                cropper = utils.RandomCropper(X1.shape[0:2], self.crop_size, seed=self.crop_seed)
                X = cropper.crop(X)

                Y = torch.ones((1, X.size(1), X.size(2)), dtype=torch.float)

            return X.cuda(), Y.cuda()

    def __len__(self):
        if self.type == 'train':
            return 200
        if self.type == 'test':
            return 20


class KITTI_2015_benchmark(Dataset):
    ROOT = r'F:\Dataset\KITTI 2015'

    # KITTI 2015 original height and width (375, 1242, 3), dtype uint8
    # height and width: (370, 1224) is the smallest size

    def __init__(self, use_resize=False, resize=(None, None)):
        assert os.path.exists(self.ROOT), 'Dataset path is not exist'
        self.root = os.path.join(self.ROOT, 'testing')
        self.use_resize = use_resize
        self.resize_height, self.resize_width = resize

    def __getitem__(self, index):
        X1 = cv2.imread(os.path.join(self.root, 'image_2/{:06d}_10.png'.format(index)))
        X2 = cv2.imread(os.path.join(self.root, 'image_3/{:06d}_10.png'.format(index)))
        self.original_height, self.original_width = X1.shape[:2]

        if self.use_resize:
            X1 = cv2.resize(X1, (self.resize_width, self.resize_height))
            X2 = cv2.resize(X2, (self.resize_width, self.resize_height))

        X1 = utils.rgb2bgr(X1)
        X2 = utils.rgb2bgr(X2)

        X = np.concatenate([X1, X2], axis=2)  # batch, height, width, channel
        X = X.swapaxes(0, 2).swapaxes(1, 2)  # channel*2, height, width
        X = torch.from_numpy(X) / 255.0

        Y = torch.ones((1, self.original_height, self.original_width), dtype=torch.float)
        return X.cuda(), Y.cuda()

    def __len__(self):
        return 200


class KITTI_2015_Augmentation(Dataset):
    ROOT = r'F:\Dataset\KITTI 2015 Data Augmentation'

    # KITTI 2015 original height and width (375, 1242, 3), dtype uint8
    # width range = [1224, 1242]
    # height range = [370, 376]

    def __init__(self, type='train', crop_size=None, crop_seed=None, seed=0):
        assert os.path.exists(self.ROOT), 'Dataset path is not exist'
        self.type = type
        self.files = os.listdir(os.path.join(self.ROOT, 'image_2'))
        np.random.seed(seed)
        indexes = np.arange(len(self.files))
        np.random.shuffle(indexes)
        self.train_indexes = indexes[:3840]
        self.test_indexes = indexes[3840:]
        self.crop_size = crop_size
        self.crop_seed = crop_seed

    def __getitem__(self, index):
        if self.type == 'train':
            X1 = cv2.imread(os.path.join(self.ROOT, f'image_2/{self.files[self.train_indexes[index]]}'))
            X2 = cv2.imread(os.path.join(self.ROOT, f'image_3/{self.files[self.train_indexes[index]]}'))
            Y = cv2.imread(os.path.join(self.ROOT, f'disp_occ_0/{self.files[self.train_indexes[index]]}'))

            X1 = utils.rgb2bgr(X1)
            X2 = utils.rgb2bgr(X2)

            X = np.concatenate([X1, X2], axis=2)
            X = X.swapaxes(0, 2).swapaxes(1, 2)

            Y = Y[:, :, 0]
            X, Y = torch.from_numpy(X).float(), torch.from_numpy(Y)
            Y = Y.unsqueeze(0)
            if self.crop_size is not None:
                cropper = utils.RandomCropper(X1.shape[0:2], self.crop_size, seed=self.crop_seed)
                X, Y = cropper.crop(X), cropper.crop(Y)
            X, Y = X.float() / 255, Y.float()

            return X.cuda(), Y.cuda()

        elif self.type == 'test':
            X1 = cv2.imread(os.path.join(self.ROOT, f'image_2/{self.files[self.test_indexes[index]]}'))
            X2 = cv2.imread(os.path.join(self.ROOT, f'image_3/{self.files[self.test_indexes[index]]}'))
            Y = cv2.imread(os.path.join(self.ROOT, f'disp_occ_0/{self.files[self.test_indexes[index]]}'))

            X1 = utils.rgb2bgr(X1)
            X2 = utils.rgb2bgr(X2)

            X = np.concatenate([X1, X2], axis=2)
            X = X.swapaxes(0, 2).swapaxes(1, 2)

            Y = Y[:, :, 0]
            X, Y = torch.from_numpy(X).float(), torch.from_numpy(Y)
            Y = Y.unsqueeze(0)

            if self.crop_size is not None:
                cropper = utils.RandomCropper(X1.shape[0:2], self.crop_size, seed=self.crop_seed)
                X, Y = cropper.crop(X), cropper.crop(Y)
            X, Y = X.float() / 255, Y.float()

            return X.cuda(), Y.cuda()

    def __len__(self):
        if self.type == 'train':
            return 3840
        if self.type == 'test':
            return 960


class AerialImagery(Dataset):
    ROOT = '/media/jack/data/Dataset/aerial imagery'
    # image_size = (800, 1280)
    image_size = (384, 1280)

    def __init__(self):
        assert os.path.exists(self.ROOT), 'Dataset path is not exist'

        self.rc = [(3020, 3015), (3200, 4500), (2950, 5760)][-2:-1]
        self.disp = 400

    def __getitem__(self, index):
        r, c = self.rc[index]
        os.makedirs(os.path.join(self.ROOT, 'cache'), exist_ok=True)
        cach_path = os.path.join(self.ROOT, 'cache', f'{r:d}_{c:d}_{self.image_size[0]}x{self.image_size[1]}.np')

        if os.path.exists(cach_path):
            print(f'using cache: {cach_path}')
            X = utils.load(cach_path)

        else:
            left_image = cv2.imread(os.path.join(self.ROOT, '0-rectL.tif'))
            right_image = cv2.imread(os.path.join(self.ROOT, '0-rectR.tif'))

            left_image = cv2.rotate(left_image, cv2.ROTATE_90_CLOCKWISE)
            right_image = cv2.rotate(right_image, cv2.ROTATE_90_CLOCKWISE)

            left_image = utils.rgb2bgr(left_image)
            right_image = utils.rgb2bgr(right_image)

            height, width = self.image_size
            left_image = left_image[r:r + height, c:c + width]
            right_image = right_image[r:r + height, c - self.disp:c - self.disp + width]

            X = np.concatenate([left_image, right_image], axis=2)
            X = X.swapaxes(0, 2).swapaxes(1, 2)
            X = torch.from_numpy(X)
            utils.save(X, cach_path)

        Y = torch.ones((1, X.size(1), X.size(2)), dtype=torch.float)

        return X.cuda() / 255.0, Y.cuda()

    def __len__(self):
        return len(self.rc)


def random_subset(dataset, size, seed=None):
    assert size <= len(dataset), 'subset size cannot larger than dataset'
    np.random.seed(seed)
    indexes = np.arange(len(dataset))
    np.random.shuffle(indexes)
    indexes = indexes[:size]
    return Subset(dataset, indexes)


def random_split(dataset, train_ratio=0.8, seed=None):
    assert 0 <= train_ratio <= 1
    train_size = int(train_ratio * len(dataset))
    np.random.seed(seed)
    indexes = np.arange(len(dataset))
    np.random.shuffle(indexes)
    train_indexes = indexes[:train_size]
    test_indexes = indexes[train_size:]
    return Subset(dataset, train_indexes), Subset(dataset, test_indexes)


def sub_sampling(X, Y, ratio):
    X = X[:, ::ratio, ::ratio]
    Y = Y[::ratio, ::ratio] / ratio
    return X, Y
