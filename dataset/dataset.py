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
    def __init__(self, max_disparity, type='train', image='cleanpass', use_crop_size=False, crop_size=None,
                 crop_seed=None, use_resize=False, resize=(None, None),
                 use_padding_crop_size=False, padding_crop_size=(None, None)):

        assert os.path.exists(self.ROOT), 'Dataset path is not exist'
        self.data_max_disparity = []
        self.image = image
        self.use_crop_size = use_crop_size
        self.crop_size = crop_size
        self.crop_seed = crop_seed
        self.use_resize = use_resize
        self.resize = resize
        self.use_padding_crop_size = use_padding_crop_size
        self.padding_crop_size = padding_crop_size
        self.pass_info = {}

        if type == 'train':
            self.data_max_disparity.append(utils.load(os.path.join(self.ROOT, f'left_max_disparity.np'))[0])
            self.root = os.path.join(self.ROOT, 'TRAIN')
            self.size = 22390

        elif type == 'test':
            self.data_max_disparity.append(utils.load(os.path.join(self.ROOT, f'left_max_disparity.np'))[1])
            self.root = os.path.join(self.ROOT, 'TEST')
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

        self._make_mask_index()

    def __getitem__(self, index):
        if self.use_crop_size:
            index = self.mask_index[index]
            X = utils.load(os.path.join(self.root, f'{self.image}/{index:05d}.np'))  # channel, height, width
            X = torch.from_numpy(X)

            cropper = utils.RandomCropper(X.shape[1:3], self.crop_size, seed=self.crop_seed)
            X = cropper.crop(X)
            X = X.float() / 255

            Y_list = []
            Y = utils.load(os.path.join(self.root, f'left_disparity/{index:05d}.np'))
            Y = torch.from_numpy(Y)
            Y = cropper.crop(Y)
            Y_list.append(Y.unsqueeze(0))
            Y = torch.cat(Y_list, dim=0)

        elif self.use_resize:
            index = self.mask_index[index]
            X = utils.load(os.path.join(self.root, f'{self.image}/{index:05d}.np'))  # channel, height, width
            self.pass_info['original_height'], self.pass_info['original_width'] = X.shape[1:]

            X1 = X[:3, :, :].swapaxes(0, 2).swapaxes(0, 1)
            X2 = X[3:, :, :].swapaxes(0, 2).swapaxes(0, 1)

            X1 = cv2.resize(X1, (self.resize[1], self.resize[0]))
            X2 = cv2.resize(X2, (self.resize[1], self.resize[0]))

            X = np.concatenate([X1, X2], axis=2)
            X = X.swapaxes(0, 1).swapaxes(0, 2)
            X = torch.from_numpy(X).float() / 255.0

            Y_list = []
            Y = utils.load(os.path.join(self.root, f'left_disparity/{index:05d}.np'))
            Y = torch.from_numpy(Y)
            Y_list.append(Y.unsqueeze(0))
            Y = torch.cat(Y_list, dim=0)

        elif self.use_padding_crop_size:
            index = self.mask_index[index]
            X = utils.load(os.path.join(self.root, f'{self.image}/{index:05d}.np'))  # channel, height, width

            self.pass_info['original_height'], self.pass_info['original_width'] = X.shape[1:]
            assert self.pass_info['original_height'] <= self.padding_crop_size[0]
            assert self.pass_info['original_width'] <= self.padding_crop_size[1]

            X_pad = np.zeros((6, *self.padding_crop_size), dtype=np.uint8)
            X_pad[:X.shape[0], :X.shape[1], :] = X[...]
            X = X_pad
            X = torch.from_numpy(X).float() / 255.0

            Y_list = []
            Y = utils.load(os.path.join(self.root, f'left_disparity/{index:05d}.np'))
            Y = torch.from_numpy(Y)
            Y_list.append(Y.unsqueeze(0))
            Y = torch.cat(Y_list, dim=0)

        return X, Y, self.pass_info

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

    def __str__(self):
        return 'FlyingThings3D'


class KITTI_2015(Dataset):

    # KITTI 2015 original height and width (375, 1242, 3), dtype uint8
    # height and width: (370, 1224) is the smallest size

    # HEIGHT, WIDTH = 384, 1248
    # HEIGHT, WIDTH = 352, 1216  # GTX 2080 Ti
    # HEIGHT, WIDTH = 256, 1248  # GTX 1660 Ti
    def __init__(self, type='train', use_crop_size=False, crop_size=None, crop_seed=None,
                 use_resize=False, resize=(None, None), use_padding_crop_size=False, padding_crop_size=(None, None),
                 untexture_rate=0.1):

        assert os.path.exists(self.get_root_directory()), 'Dataset path is not exist'
        assert use_crop_size + use_resize <= 1, 'Using one of the crop size and the resize'

        self.type = type
        if type == 'train':
            self.root = os.path.join(self.get_root_directory(), 'training')
        elif type == 'test':
            self.root = os.path.join(self.get_root_directory(), 'testing')
        else:
            raise Exception('Unknown type "{}"'.format(type))

        self.use_crop_size = use_crop_size
        self.crop_size = crop_size
        self.crop_seed = crop_seed
        self.use_resize = use_resize
        self.resize = resize
        self.use_padding_crop_size = use_padding_crop_size
        self.padding_crop_size = padding_crop_size
        self.untexture_rate = untexture_rate
        self.pass_info = {}

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

            else:
                if self.use_resize:
                    X1 = cv2.imread(os.path.join(self.root, 'image_2/{:06d}_10.png'.format(index)))
                    X2 = cv2.imread(os.path.join(self.root, 'image_3/{:06d}_10.png'.format(index)))
                    Y = cv2.imread(os.path.join(self.root, 'disp_occ_0/{:06d}_10.png'.format(index)))  # (376, 1241, 3) uint8
                    self.pass_info['original_height'], self.pass_info['original_width'] = X1.shape[:2]

                    X1 = cv2.resize(X1, (self.resize[1], self.resize[0]))
                    X2 = cv2.resize(X2, (self.resize[1], self.resize[0]))

                    X1 = utils.rgb2bgr(X1)
                    X2 = utils.rgb2bgr(X2)

                    X = np.concatenate([X1, X2], axis=2)  # batch, height, width, channel
                    X = X.swapaxes(0, 2).swapaxes(1, 2)  # channel*2, height, width

                    Y = Y[:, :, 0]
                    X, Y = torch.from_numpy(X).float() / 255, torch.from_numpy(Y).float().unsqueeze(0)

                elif self.use_crop_size:
                    X1 = cv2.imread(os.path.join(self.root, 'image_2/{:06d}_10.png'.format(index)))
                    X2 = cv2.imread(os.path.join(self.root, 'image_3/{:06d}_10.png'.format(index)))
                    Y = cv2.imread(os.path.join(self.root, 'disp_occ_0/{:06d}_10.png'.format(index)))  # (376, 1241, 3) uint8

                    X1 = utils.rgb2bgr(X1)
                    X2 = utils.rgb2bgr(X2)

                    X = np.concatenate([X1, X2], axis=2)
                    X = X.swapaxes(0, 2).swapaxes(1, 2)

                    Y = Y[:, :, 0]
                    X, Y = torch.from_numpy(X).float() / 255, torch.from_numpy(Y).float().unsqueeze(0)

                    cropper = utils.RandomCropper(X1.shape[0:2], self.crop_size, seed=self.crop_seed)
                    X, Y = cropper.crop(X), cropper.crop(Y)

                elif self.use_padding_crop_size:

                    X1 = cv2.imread(os.path.join(self.root, 'image_2/{:06d}_10.png'.format(index)))
                    X2 = cv2.imread(os.path.join(self.root, 'image_3/{:06d}_10.png'.format(index)))
                    Y = cv2.imread(os.path.join(self.root, 'disp_occ_0/{:06d}_10.png'.format(index)))  # (376, 1241, 3) uint8

                    self.pass_info['original_height'], self.pass_info['original_width'] = X1.shape[:2]
                    assert self.pass_info['original_height'] <= self.padding_crop_size[0]
                    assert self.pass_info['original_width'] <= self.padding_crop_size[1]

                    X1 = utils.rgb2bgr(X1)
                    X2 = utils.rgb2bgr(X2)

                    X1_pad = np.zeros((*self.padding_crop_size, 3), dtype=np.uint8)
                    X2_pad = np.zeros((*self.padding_crop_size, 3), dtype=np.uint8)

                    X1_pad[:X1.shape[0], :X1.shape[1], :] = X1[...]
                    X2_pad[:X1.shape[0], :X1.shape[1], :] = X2[...]

                    X1 = X1_pad
                    X2 = X2_pad

                    X = np.concatenate([X1, X2], axis=2)  # batch, height, width, channel
                    X = X.swapaxes(0, 2).swapaxes(1, 2)  # channel*2, height, width

                    Y = Y[:, :, 0]
                    X, Y = torch.from_numpy(X).float() / 255, torch.from_numpy(Y).float().unsqueeze(0)

        elif self.type == 'test':
            if self.use_resize:
                X1 = cv2.imread(os.path.join(self.root, 'image_2/{:06d}_10.png'.format(index)))
                X2 = cv2.imread(os.path.join(self.root, 'image_3/{:06d}_10.png'.format(index)))
                self.pass_info['original_height'], self.pass_info['original_width'] = X1.shape[:2]

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

            elif self.use_padding_crop_size:
                X1 = cv2.imread(os.path.join(self.root, 'image_2/{:06d}_10.png'.format(index)))
                X2 = cv2.imread(os.path.join(self.root, 'image_3/{:06d}_10.png'.format(index)))
                self.pass_info['original_height'], self.pass_info['original_width'] = X1.shape[:2]
                assert self.pass_info['original_height'] <= self.padding_crop_size[0]
                assert self.pass_info['original_width'] <= self.padding_crop_size[1]

                X1 = utils.rgb2bgr(X1)
                X2 = utils.rgb2bgr(X2)

                X1_pad = np.zeros((*self.padding_crop_size, 3), dtype=np.uint8)
                X2_pad = np.zeros((*self.padding_crop_size, 3), dtype=np.uint8)

                X1_pad[:X1.shape[0], :X1.shape[1], :] = X1[...]
                X2_pad[:X1.shape[0], :X1.shape[1], :] = X2[...]

                X1 = X1_pad
                X2 = X2_pad

                X = np.concatenate([X1, X2], axis=2)  # batch, height, width, channel
                X = X.swapaxes(0, 2).swapaxes(1, 2)  # channel*2, height, width
                X = torch.from_numpy(X).float() / 255

                Y = torch.ones((1, X.size(1), X.size(2)), dtype=torch.float)

        return X, Y, self.pass_info

    def get_root_directory(self):
        return f'F:\Dataset\KITTI 2015'

    def get_left_image_folder(self):
        return 'image_2'

    def get_right_image_folder(self):
        return 'image_3'

    def get_disp_image_folder(self):
        return 'disp_occ_0'

    def __len__(self):
        if self.type == 'train':
            return 200
        if self.type == 'test':
            return 20

    def __str__(self):
        return 'KITTI_2015'


class KITTI_2015_benchmark(Dataset):
    ROOT = r'F:\Dataset\KITTI 2015'

    # KITTI 2015 original height and width (375, 1242, 3), dtype uint8
    # height and width: (370, 1224) is the smallest size

    def __init__(self, use_padding_crop_size=False, padding_crop_size=(None, None)):
        assert os.path.exists(self.ROOT), 'Dataset path is not exist'
        self.root = os.path.join(self.ROOT, 'testing')
        self.use_padding_crop_size = use_padding_crop_size
        self.padding_crop_size = padding_crop_size
        self.pass_info = {}

    def __getitem__(self, index):
        X1 = cv2.imread(os.path.join(self.root, 'image_2/{:06d}_10.png'.format(index)))
        X2 = cv2.imread(os.path.join(self.root, 'image_3/{:06d}_10.png'.format(index)))
        self.pass_info['original_height'], self.pass_info['original_width'] = X1.shape[:2]
        assert self.pass_info['original_height'] <= self.padding_crop_size[0]
        assert self.pass_info['original_width'] <= self.padding_crop_size[1]

        X1 = utils.rgb2bgr(X1)
        X2 = utils.rgb2bgr(X2)

        X1_pad = np.zeros((*self.padding_crop_size, 3), dtype=np.uint8)
        X2_pad = np.zeros((*self.padding_crop_size, 3), dtype=np.uint8)

        X1_pad[:X1.shape[0], :X1.shape[1], :] = X1[...]
        X2_pad[:X1.shape[0], :X1.shape[1], :] = X2[...]

        X1 = X1_pad
        X2 = X2_pad

        X = np.concatenate([X1, X2], axis=2)  # height, width, channel
        X = X.swapaxes(0, 2).swapaxes(1, 2)  # channel*2, height, width
        X = torch.from_numpy(X) / 255.0

        Y = torch.ones((1, self.pass_info['original_height'], self.pass_info['original_width']), dtype=torch.float)

        return X, Y, self.pass_info

    def __len__(self):
        return 200

    def __str__(self):
        return 'KITTI_2015_benchmark'


class KITTI_Augmentation(Dataset):
    # KITTI 2015 original height and width (375, 1242, 3), dtype uint8
    # width range = [1224, 1242]
    # height range = [370, 376]

    def __init__(self, type='train', shuffle_seed=0, use_crop_size=False, crop_size=None, crop_seed=None,
                 use_resize=False, resize=(None, None), use_padding_crop_size=False, padding_crop_size=(None, None)):
        assert os.path.exists(self.get_root_directory()), 'Dataset path is not exist'
        assert use_crop_size + use_resize + use_padding_crop_size == 1, 'Using one of methods to produce disparity'
        self.type = type

        if type == 'train':
            self.files = os.listdir(os.path.join(self.get_root_directory(), 'training', self.get_left_image_folder()))
            self.train_indexes = np.arange(self.get_train_size())
            np.random.seed(shuffle_seed)
            np.random.shuffle(self.train_indexes)

        elif type == 'test':
            self.files = os.listdir(os.path.join(self.get_root_directory(), 'testing', self.get_left_image_folder()))
            self.test_indexes = np.arange(self.get_test_size())
            np.random.seed(shuffle_seed)
            np.random.shuffle(self.test_indexes)

        self.use_crop_size = use_crop_size
        self.use_resize = use_resize
        self.use_padding_crop_size = use_padding_crop_size
        self.resize = resize
        self.crop_size = crop_size
        self.padding_crop_size = padding_crop_size
        self.crop_seed = crop_seed
        self.pass_info = {}

    def __getitem__(self, index):
        if self.type == 'train':
            if self.use_resize:
                X1 = cv2.imread(os.path.join(self.get_root_directory(),
                                             f'training/{self.get_left_image_folder()}/{self.files[self.train_indexes[index]]}'))
                X2 = cv2.imread(os.path.join(self.get_root_directory(),
                                             f'training/{self.get_right_image_folder()}/{self.files[self.train_indexes[index]]}'))
                Y = cv2.imread(os.path.join(self.get_root_directory(),
                                            f'training/{self.get_disp_image_folder()}/{self.files[self.train_indexes[index]]}'))
                self.pass_info['original_height'], self.pass_info['original_width'] = X1.shape[:2]

                X1 = cv2.resize(X1, (self.resize[1], self.resize[0]))
                X2 = cv2.resize(X2, (self.resize[1], self.resize[0]))

                X1 = utils.rgb2bgr(X1)
                X2 = utils.rgb2bgr(X2)

                X = np.concatenate([X1, X2], axis=2)  # height, width, channel
                X = X.swapaxes(0, 2).swapaxes(1, 2)  # channel*2, height, width

                Y = Y[:, :, 0]
                X, Y = torch.from_numpy(X).float() / 255, torch.from_numpy(Y)
                Y = Y.unsqueeze(0)

            elif self.use_crop_size:
                X1 = cv2.imread(os.path.join(self.get_root_directory(),
                                             f'training/{self.get_left_image_folder()}/{self.files[self.train_indexes[index]]}'))
                X2 = cv2.imread(os.path.join(self.get_root_directory(),
                                             f'training/{self.get_right_image_folder()}/{self.files[self.train_indexes[index]]}'))
                Y = cv2.imread(os.path.join(self.get_root_directory(),
                                            f'training/{self.get_disp_image_folder()}/{self.files[self.train_indexes[index]]}'))

                X1 = utils.rgb2bgr(X1)
                X2 = utils.rgb2bgr(X2)

                X = np.concatenate([X1, X2], axis=2)  # height, width, channel
                X = X.swapaxes(0, 2).swapaxes(1, 2)  # channel*2, height, width

                Y = Y[:, :, 0]
                X, Y = torch.from_numpy(X).float(), torch.from_numpy(Y)
                Y = Y.unsqueeze(0)

                cropper = utils.RandomCropper(X1.shape[0:2], self.crop_size, seed=self.crop_seed)
                X, Y = cropper.crop(X), cropper.crop(Y)
                X, Y = X.float() / 255, Y.float()

            elif self.use_padding_crop_size:
                X1 = cv2.imread(os.path.join(self.get_root_directory(),
                                             f'training/{self.get_left_image_folder()}/{self.files[self.train_indexes[index]]}'))
                X2 = cv2.imread(os.path.join(self.get_root_directory(),
                                             f'training/{self.get_right_image_folder()}/{self.files[self.train_indexes[index]]}'))
                Y = cv2.imread(os.path.join(self.get_root_directory(),
                                            f'training/{self.get_disp_image_folder()}/{self.files[self.train_indexes[index]]}'))

                self.pass_info['original_height'], self.pass_info['original_width'] = X1.shape[:2]
                assert self.pass_info['original_height'] <= self.padding_crop_size[0]
                assert self.pass_info['original_width'] <= self.padding_crop_size[1]

                X1 = utils.rgb2bgr(X1)
                X2 = utils.rgb2bgr(X2)

                X1_pad = np.zeros((*self.padding_crop_size, 3), dtype=np.uint8)
                X2_pad = np.zeros((*self.padding_crop_size, 3), dtype=np.uint8)

                X1_pad[:X1.shape[0], :X1.shape[1], :] = X1[...]
                X2_pad[:X1.shape[0], :X1.shape[1], :] = X2[...]

                X1 = X1_pad
                X2 = X2_pad

                X = np.concatenate([X1, X2], axis=2)  # height, width, channel
                X = X.swapaxes(0, 2).swapaxes(1, 2)  # channel*2, height, width

                Y = Y[:, :, 0]
                X, Y = torch.from_numpy(X).float(), torch.from_numpy(Y)
                Y = Y.unsqueeze(0)
                X, Y = X.float() / 255, Y.float()

        elif self.type == 'test':
            if self.use_resize:
                X1 = cv2.imread(os.path.join(self.get_root_directory(),
                                             f'testing/{self.get_left_image_folder()}/{self.files[self.test_indexes[index]]}'))
                X2 = cv2.imread(os.path.join(self.get_root_directory(),
                                             f'testing/{self.get_right_image_folder()}/{self.files[self.test_indexes[index]]}'))
                Y = cv2.imread(os.path.join(self.get_root_directory(),
                                            f'testing/{self.get_disp_image_folder()}/{self.files[self.test_indexes[index]]}'))
                self.pass_info['original_height'], self.pass_info['original_width'] = X1.shape[:2]

                X1 = cv2.resize(X1, (self.resize_width, self.resize_height))
                X2 = cv2.resize(X2, (self.resize_width, self.resize_height))

                X1 = utils.rgb2bgr(X1)
                X2 = utils.rgb2bgr(X2)

                X = np.concatenate([X1, X2], axis=2)  # height, width, channel
                X = X.swapaxes(0, 2).swapaxes(1, 2)  # channel*2, height, width

                Y = Y[:, :, 0]
                X, Y = torch.from_numpy(X).float() / 255, torch.from_numpy(Y)
                Y = Y.unsqueeze(0)

            elif self.use_crop_size:
                X1 = cv2.imread(os.path.join(self.get_root_directory(),
                                             f'testing/{self.get_left_image_folder()}/{self.files[self.test_indexes[index]]}'))
                X2 = cv2.imread(os.path.join(self.get_root_directory(),
                                             f'testing/{self.get_right_image_folder()}/{self.files[self.test_indexes[index]]}'))
                Y = cv2.imread(os.path.join(self.get_root_directory(),
                                            f'testing/{self.get_disp_image_folder()}/{self.files[self.test_indexes[index]]}'))

                X1 = utils.rgb2bgr(X1)
                X2 = utils.rgb2bgr(X2)

                X = np.concatenate([X1, X2], axis=2)  # height, width, channel
                X = X.swapaxes(0, 2).swapaxes(1, 2)  # channel*2, height, width

                Y = Y[:, :, 0]
                X, Y = torch.from_numpy(X).float(), torch.from_numpy(Y)
                Y = Y.unsqueeze(0)

                cropper = utils.RandomCropper(X1.shape[0:2], self.crop_size, seed=self.crop_seed)
                X, Y = cropper.crop(X), cropper.crop(Y)
                X, Y = X.float() / 255, Y.float()

            elif self.use_padding_crop_size:
                X1 = cv2.imread(os.path.join(self.get_root_directory(),
                                             f'testing/{self.get_left_image_folder()}/{self.files[self.test_indexes[index]]}'))
                X2 = cv2.imread(os.path.join(self.get_root_directory(),
                                             f'testing/{self.get_right_image_folder()}/{self.files[self.test_indexes[index]]}'))
                Y = cv2.imread(os.path.join(self.get_root_directory(),
                                            f'testing/{self.get_disp_image_folder()}/{self.files[self.test_indexes[index]]}'))

                self.pass_info['original_height'], self.pass_info['original_width'] = X1.shape[:2]
                assert self.pass_info['original_height'] < self.padding_crop_size[0]
                assert self.pass_info['original_height'] < self.padding_crop_size[1]

                X1 = utils.rgb2bgr(X1)
                X2 = utils.rgb2bgr(X2)

                X1_pad = np.zeros((*self.padding_crop_size, 3), dtype=np.uint8)
                X2_pad = np.zeros((*self.padding_crop_size, 3), dtype=np.uint8)

                X1_pad[:X1.shape[0], :X1.shape[1], :] = X1[...]
                X2_pad[:X1.shape[0], :X1.shape[1], :] = X2[...]

                X1 = X1_pad
                X2 = X2_pad

                X = np.concatenate([X1, X2], axis=2)  # height, width, channel
                X = X.swapaxes(0, 2).swapaxes(1, 2)  # channel*2, height, width

                Y = Y[:, :, 0]
                X, Y = torch.from_numpy(X).float(), torch.from_numpy(Y)
                Y = Y.unsqueeze(0)
                X, Y = X.float() / 255, Y.float()

        return X, Y, self.pass_info

    def get_root_directory(self):
        return ''

    def get_left_image_folder(self):
        return ''

    def get_right_image_folder(self):
        return ''

    def get_disp_image_folder(self):
        return ''

    def get_train_size(self):
        pass

    def get_test_size(self):
        pass

    def __len__(self):
        if self.type == 'train':
            return self.get_train_size()
        if self.type == 'test':
            return self.get_test_size()

    def __str__(self):
        pass


class KITTI_2015_Augmentation(KITTI_Augmentation):

    def __init__(self, type='train', shuffle_seed=0, use_crop_size=False, crop_size=None, crop_seed=None,
                 use_resize=False, resize=(None, None), use_padding_crop_size=False, padding_crop_size=(None, None)):

        super().__init__(type=type, use_crop_size=use_crop_size, crop_size=crop_size, crop_seed=crop_seed,
                         shuffle_seed=shuffle_seed, use_resize=use_resize, resize=resize,
                         use_padding_crop_size=use_padding_crop_size,
                         padding_crop_size=padding_crop_size)

    def get_root_directory(self):
        return f'F:\Dataset\KITTI 2015 Data Augmentation'

    def get_left_image_folder(self):
        return 'image_2'

    def get_right_image_folder(self):
        return 'image_3'

    def get_disp_image_folder(self):
        return 'disp_occ_0'

    def get_train_size(self):
        return 3840

    def get_test_size(self):
        return 960

    def __str__(self):
        return 'KITTI_2015_Augmentation'


class KITTI_2012_Augmentation(KITTI_Augmentation):
    ROOT = r'F:\Dataset\KITTI 2015 Data Augmentation'

    def __init__(self, type='train', shuffle_seed=0, use_crop_size=False, crop_size=None, crop_seed=None,
                 use_resize=False, resize=(None, None), use_padding_crop_size=False, padding_crop_size=(None, None)):

        super().__init__(type=type, use_crop_size=use_crop_size, crop_size=crop_size, crop_seed=crop_seed,
                         shuffle_seed=shuffle_seed, use_resize=use_resize, resize=resize,
                         use_padding_crop_size=use_padding_crop_size,
                         padding_crop_size=padding_crop_size)

    def get_root_directory(self):
        return r'F:\Dataset\KITTI 2012 Data Augmentation'

    def get_left_image_folder(self):
        return 'colored_0'

    def get_right_image_folder(self):
        return 'colored_1'

    def get_disp_image_folder(self):
        return 'disp_occ'

    def get_train_size(self):
        return 3720

    def get_test_size(self):
        return 936

    def __str__(self):
        return 'KITTI_2012_Augmentation'


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

        return X / 255.0, Y

    def __len__(self):
        return len(self.rc)

    def __str__(self):
        return 'AerialImagery'


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
