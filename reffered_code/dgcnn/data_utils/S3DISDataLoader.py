import os
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


class S3DISDataset(Dataset):
    def __init__(self, split='train', data_root='trainval_fullarea', num_point=4096, test_area=5, block_size=1.0, sample_rate=1.0, transform=None):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        rooms = sorted(os.listdir(data_root))
        rooms = [room for room in rooms if 'Area_' in room]
        if split == 'train':
            rooms_split = [room for room in rooms if not 'Area_{}'.format(test_area) in room]
        else:
            rooms_split = [room for room in rooms if 'Area_{}'.format(test_area) in room]

        self.room_points, self.room_labels = [], []
        self.room_coord_min, self.room_coord_max = [], []
        num_point_all = []
        labelweights = np.zeros(13)

        for room_name in tqdm(rooms_split, total=len(rooms_split)):
            room_path = os.path.join(data_root, room_name)
            room_data = np.load(room_path)  # xyzrgbl, N*7
            points, labels = room_data[:, 0:6], room_data[:, 6]  # xyzrgb, N*6; l, N
            tmp, _ = np.histogram(labels, range(14))  # tmp: each number of labels
            labelweights += tmp                       # labelweights: each number of labels
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_points.append(points), self.room_labels.append(labels)
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
            num_point_all.append(labels.size)
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)   # calculate all the average ?
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
       # print(self.labelweights)
        sample_prob = num_point_all / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point)
        room_idxs = []
        for index in range(len(rooms_split)):
            room_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
        self.room_idxs = np.array(room_idxs)
        print("Totally {} samples in {} set.".format(len(self.room_idxs), split))

    def __getitem__(self, idx):
        room_idx = self.room_idxs[idx]
        points = self.room_points[room_idx]   # N * 6
        labels = self.room_labels[room_idx]   # N
        N_points = points.shape[0]

        while (True):
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
            if point_idxs.size > 1024:
                break

        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

        # normalize
        selected_points = points[selected_point_idxs, :]  # num_point * 6
        current_points = np.zeros((self.num_point, 9))  # num_point * 9
        current_points[:, 6] = selected_points[:, 0] / self.room_coord_max[room_idx][0]
        current_points[:, 7] = selected_points[:, 1] / self.room_coord_max[room_idx][1]
        current_points[:, 8] = selected_points[:, 2] / self.room_coord_max[room_idx][2]
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        selected_points[:, 3:6] /= 255.0
        current_points[:, 0:6] = selected_points
        current_labels = labels[selected_point_idxs]
        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)
        return current_points, current_labels

    def __len__(self):
        return len(self.room_idxs)


class MotorDataset(Dataset):
    def __init__(self, split='train', data_root='trainval_fullarea', num_point=4096, bolt_weight = 1, test_area='Validation', block_size=1.0, sample_rate=1.0, transform=None):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        motors = sorted(os.listdir(data_root))
        motors = [motor for motor in motors if 'Type' in motor]
        if split == 'train':
            motors_split = [motor for motor in motors if not '{}'.format(test_area) in motor]
        else:
            motors_split = [motor for motor in motors if '{}'.format(test_area) in motor]

        self.motor_points, self.motor_labels = [], []
        self.motor_coord_min, self.motor_coord_max = [], []
        num_point_all = []
        labelweights = np.zeros(6)

        for motor_name in tqdm(motors_split, total=len(motors_split)):
            motor_path = os.path.join(data_root, motor_name)
            motor_data = np.load(motor_path)  # xyzrgbl, N*7
            points, labels = motor_data[:, 0:6], motor_data[:, 6]  # xyzrgb, N*6; l, N
            tmp, _ = np.histogram(labels, range(7))
            labelweights += tmp
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.motor_points.append(points), self.motor_labels.append(labels)
            self.motor_coord_min.append(coord_min), self.motor_coord_max.append(coord_max)
            num_point_all.append(labels.size)
        labelweights = labelweights.astype(np.float32)
        labelweights[-1] /= bolt_weight
        labelweights = labelweights / np.sum(labelweights)
        labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        self.labelweights = labelweights / np.sum(labelweights) ########### add change 07/03
        print(self.labelweights)
        sample_prob = num_point_all / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point)
        motor_idxs = []
        for index in range(len(motors_split)):
            motor_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
        self.motor_idxs = np.array(motor_idxs)
        print("Totally {} samples in {} set.".format(len(self.motor_idxs), split))

    def __getitem__(self, idx):
        motor_idx = self.motor_idxs[idx]
        points = self.motor_points[motor_idx]   # N * 6
        labels = self.motor_labels[motor_idx]   # N
        N_points = points.shape[0]

        while (True):
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
            if point_idxs.size > 1024:
                break

        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

        # normalize
        selected_points = points[selected_point_idxs, :]  # num_point * 6
        current_points = np.zeros((self.num_point, 9))  # num_point * 9
        current_points[:, 6] = selected_points[:, 0] / self.motor_coord_max[motor_idx][0]
        current_points[:, 7] = selected_points[:, 1] / self.motor_coord_max[motor_idx][1]
        current_points[:, 8] = selected_points[:, 2] / self.motor_coord_max[motor_idx][2]
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        selected_points[:, 3:6] /= 255.0
        current_points[:, 0:6] = selected_points
        current_labels = labels[selected_point_idxs]
        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)
        return current_points, current_labels

    def __len__(self):
        return len(self.motor_idxs)

class ScannetDatasetWholeScene():
    # prepare to give prediction on each points
    def __init__(self, root, block_points=4096, split='test', test_area=5, stride=0.5, block_size=1.0, padding=0.001):
        self.block_points = block_points
        self.block_size = block_size
        self.padding = padding
        self.root = root
        self.split = split
        self.stride = stride
        self.scene_points_num = []
        assert split in ['train', 'test']
        if self.split == 'train':
            self.file_list = [d for d in os.listdir(root) if d.find('Area_%d' % test_area) is -1]
        else:
            self.file_list = [d for d in os.listdir(root) if d.find('Area'%d % test_area) is not -1]
        self.scene_points_list = []
        self.semantic_labels_list = []
        self.room_coord_min, self.room_coord_max = [], []
        for file in self.file_list:
            data = np.load(root + file)
            points = data[:, :3]
            self.scene_points_list.append(data[:, :6])
            self.semantic_labels_list.append(data[:, 6])
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
        assert len(self.scene_points_list) == len(self.semantic_labels_list)

        labelweights = np.zeros(13)
        for seg in self.semantic_labels_list:
            tmp, _ = np.histogram(seg, range(14))
            self.scene_points_num.append(seg.shape[0])
            labelweights += tmp
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)

    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]
        points = point_set_ini[:,:6]
        labels = self.semantic_labels_list[index]
        coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
        grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - self.block_size) / self.stride) + 1)
        grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - self.block_size) / self.stride) + 1)
        data_room, label_room, sample_weight, index_room = np.array([]), np.array([]), np.array([]),  np.array([])
        for index_y in range(0, grid_y):
            for index_x in range(0, grid_x):
                s_x = coord_min[0] + index_x * self.stride
                e_x = min(s_x + self.block_size, coord_max[0])
                s_x = e_x - self.block_size
                s_y = coord_min[1] + index_y * self.stride
                e_y = min(s_y + self.block_size, coord_max[1])
                s_y = e_y - self.block_size
                point_idxs = np.where(
                    (points[:, 0] >= s_x - self.padding) & (points[:, 0] <= e_x + self.padding) & (points[:, 1] >= s_y - self.padding) & (
                                points[:, 1] <= e_y + self.padding))[0]
                if point_idxs.size == 0:
                    continue
                num_batch = int(np.ceil(point_idxs.size / self.block_points))
                point_size = int(num_batch * self.block_points)
                replace = False if (point_size - point_idxs.size <= point_idxs.size) else True
                point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
                point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
                np.random.shuffle(point_idxs)
                data_batch = points[point_idxs, :]
                normlized_xyz = np.zeros((point_size, 3))
                normlized_xyz[:, 0] = data_batch[:, 0] / coord_max[0]
                normlized_xyz[:, 1] = data_batch[:, 1] / coord_max[1]
                normlized_xyz[:, 2] = data_batch[:, 2] / coord_max[2]
                data_batch[:, 0] = data_batch[:, 0] - (s_x + self.block_size / 2.0)
                data_batch[:, 1] = data_batch[:, 1] - (s_y + self.block_size / 2.0)
                data_batch[:, 3:6] /= 255.0
                data_batch = np.concatenate((data_batch, normlized_xyz), axis=1)
                label_batch = labels[point_idxs].astype(int)
                batch_weight = self.labelweights[label_batch]

                data_room = np.vstack([data_room, data_batch]) if data_room.size else data_batch
                label_room = np.hstack([label_room, label_batch]) if label_room.size else label_batch
                sample_weight = np.hstack([sample_weight, batch_weight]) if label_room.size else batch_weight
                index_room = np.hstack([index_room, point_idxs]) if index_room.size else point_idxs
        data_room = data_room.reshape((-1, self.block_points, data_room.shape[1]))
        label_room = label_room.reshape((-1, self.block_points))
        sample_weight = sample_weight.reshape((-1, self.block_points))
        index_room = index_room.reshape((-1, self.block_points))
        return data_room, label_room, sample_weight, index_room

    def __len__(self):
        return len(self.scene_points_list)


class ScannetDatasetGridMotor():
    # prepare to give prediction on each points
    def __init__(self, root, block_points=4096, split='test', test_area='Validation', stride=1.0, block_size=1.0, padding=0.001):
        self.block_points = block_points
        self.block_size = block_size
        self.padding = padding
        self.root = root
        self.split = split
        self.test_area = test_area
        self.stride = stride
        self.scene_points_num = []
        assert split in ['train', 'test']
        if self.split == 'train':
            self.file_list = [d for d in os.listdir(root) if d.find('%s' % test_area) is -1]
        else:
            self.file_list = [d for d in os.listdir(root) if d.find('%s' % test_area) is not -1]
        self.scene_points_list = []
        self.semantic_labels_list = []
        self.motor_coord_min, self.motor_coord_max = [], []
        for file in self.file_list:
            data = np.load(root + file)
            points = data[:, :3]
            self.scene_points_list.append(data[:, :6])
            self.semantic_labels_list.append(data[:, 6])

            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.motor_coord_min.append(coord_min), self.motor_coord_max.append(coord_max)

        assert len(self.scene_points_list) == len(self.semantic_labels_list)

        labelweights = np.zeros(6)
        for seg in self.semantic_labels_list:
            tmp, _ = np.histogram(seg, range(7))
            self.scene_points_num.append(seg.shape[0])
            labelweights += tmp
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)

    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]
        points = point_set_ini[:,:6]   # total_num_points*6

        coor_min_test = np.array([-255.2752533,  -110.75037384,  519.2489624])
        coor_max_test = np.array([ 45.58950043, 125.44611359, 797.55413818])
        coor_min_val = np.array([-0.22853388, -0.6627815,  -3.93524766])
        coor_max_val = np.array([ 0.76870452,  0.74199943, -2.73275002])
        scale = (coor_max_val-coor_min_val) / (coor_max_test - coor_min_test)
       # trans = coor_max_val - scale * coor_max_test
        points[:, 0:3] = points[:,0:3] * 0.004

        labels = self.semantic_labels_list[index]
        coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
        '''
        under Test: coor_min = [-255.2752533  -110.75037384  519.2489624 ]
                    coor_max = [ 45.58950043 125.44611359 797.55413818]
        under Validation: coor_min = [-0.22853388 -0.6627815  -3.93524766]
                          coor_max = [ 0.76870452  0.74199943 -2.73275002]
        '''
       # print('1111111111111', coord_min)
        grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - self.block_size) / self.stride) + 1) # under Vali: 1   under Test: 601
        grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - self.block_size) / self.stride) + 1)  # under Vali: 2   under Test grid_y = 472 
        data_motor, label_motor, sample_weight, index_motor = np.array([]), np.array([]), np.array([]),  np.array([])
        for index_y in range(0, grid_y):
            for index_x in range(0, grid_x):
                s_x = coord_min[0] + index_x * self.stride
                e_x = min(s_x + self.block_size, coord_max[0])
                s_x = e_x - self.block_size
                s_y = coord_min[1] + index_y * self.stride
                e_y = min(s_y + self.block_size, coord_max[1])
                s_y = e_y - self.block_size
                point_idxs = np.where(
                    (points[:, 0] >= s_x - self.padding) & (points[:, 0] <= e_x + self.padding) & (points[:, 1] >= s_y - self.padding) & (
                                points[:, 1] <= e_y + self.padding))[0]
                if point_idxs.size == 0:
                    continue
                num_batch = int(np.ceil(point_idxs.size / self.block_points))
                point_size = int(num_batch * self.block_points)    # point_size = 4096
                replace = False if (point_size - point_idxs.size <= point_idxs.size) else True
                point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
                point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
                np.random.shuffle(point_idxs)  # fix the range
               # print('1111111111111', point_idxs.shape)
                data_batch = points[point_idxs, :]
                normlized_xyz = np.zeros((point_size, 3))
                normlized_xyz[:, 0] = data_batch[:, 0] / coord_max[0]
                normlized_xyz[:, 1] = data_batch[:, 1] / coord_max[1]
                normlized_xyz[:, 2] = data_batch[:, 2] / coord_max[2]
                data_batch[:, 0] = data_batch[:, 0] - (s_x + self.block_size / 2.0) 
                data_batch[:, 1] = data_batch[:, 1] - (s_y + self.block_size / 2.0)
                data_batch[:, 3:6] /= 255.0

                data_batch = np.concatenate((data_batch, normlized_xyz), axis=1)  # col = 9
                label_batch = labels[point_idxs].astype(int)
                batch_weight = self.labelweights[label_batch]

                data_motor = np.vstack([data_motor, data_batch]) if data_motor.size else data_batch  # under Test: data_motor[0]=[4.58282471e-01 -5.00000000e-01  5.65709167e+02  6.00000000e+00  1.00000000e+01  1.00000000e+01 -2.23334254e+00 -8.82852172e-01  7.09305037e-01]

                label_motor = np.hstack([label_motor, label_batch]) if label_motor.size else label_batch
                sample_weight = np.hstack([sample_weight, batch_weight]) if label_motor.size else batch_weight
                index_motor = np.hstack([index_motor, point_idxs]) if index_motor.size else point_idxs  # under Test: shape: += 4096
           # print('11111111111', index_motor.shape)                                                 # under Validation: First 73721, second 14220, end
        data_motor = data_motor.reshape((-1, self.block_points, data_motor.shape[1]))  # 215*N*9
        index_motor = index_motor.reshape((-1, self.block_points))
        label_motor = label_motor.reshape((-1, self.block_points))      # 215*N
        sample_weight = sample_weight.reshape((-1, self.block_points))
        return data_motor, label_motor, sample_weight, index_motor

    def __len__(self):
        return len(self.scene_points_list)


class ScannetDatasetwholeMotor():
    # prepare to give prediction on each points
    def __init__(self, root, block_points=4096, split='test', test_area='Validation', stride=100.0, block_size=50.0, padding=0.001):
        self.block_points = block_points
        self.block_size = block_size
        self.padding = padding
        self.root = root
        self.split = split
        self.test_area = test_area
        self.stride = stride
        self.scene_points_num = []
        assert split in ['train', 'test']
        if self.split == 'train':
            self.file_list = [d for d in os.listdir(root) if d.find('%s' % test_area) is -1]
        else:
            self.file_list = [d for d in os.listdir(root) if d.find('%s' % test_area) is not -1]
        self.scene_points_list = []
        self.semantic_labels_list = []
        self.motor_coord_min, self.motor_coord_max = [], []
        for file in self.file_list:
            data = np.load(root + file)
            points = data[:, :3]
            self.scene_points_list.append(data[:, :6])
            self.semantic_labels_list.append(data[:, 6])

            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.motor_coord_min.append(coord_min), self.motor_coord_max.append(coord_max)

        assert len(self.scene_points_list) == len(self.semantic_labels_list)

        labelweights = np.zeros(6)
        for seg in self.semantic_labels_list:
            tmp, _ = np.histogram(seg, range(7))
            self.scene_points_num.append(seg.shape[0])
            labelweights += tmp
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)

    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]
        points = point_set_ini[:,:6]
        labels = self.semantic_labels_list[index]
        coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
        '''
        under Test: coor_min = [-255.2752533  -110.75037384  519.2489624 ]
                    coor_max = [ 45.58950043 125.44611359 797.55413818]
        under Validation: coor_min = [-0.22853388 -0.6627815  -3.93524766]
                          coor_max = [ 0.76870452  0.74199943 -2.73275002]
        '''
       # print('1111111111111', coord_min)
        # grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - self.block_size) / self.stride) + 1) # under Vali: 1   under Test: 601
        # grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - self.block_size) / self.stride) + 1)  # under Vali: 2   under Test grid_y = 472 
        # data_motor, label_motor, sample_weight, index_motor = np.array([]), np.array([]), np.array([]),  np.array([])
        # for index_y in range(0, grid_y):
        #     for index_x in range(0, grid_x):
        #         s_x = coord_min[0] + index_x * self.stride
        #         e_x = min(s_x + self.block_size, coord_max[0])
        #         s_x = e_x - self.block_size
        #         s_y = coord_min[1] + index_y * self.stride
        #         e_y = min(s_y + self.block_size, coord_max[1])
        #         s_y = e_y - self.block_size
        #         point_idxs = np.where(
        #             (points[:, 0] >= s_x - self.padding) & (points[:, 0] <= e_x + self.padding) & (points[:, 1] >= s_y - self.padding) & (
        #                         points[:, 1] <= e_y + self.padding))[0]
        #         if point_idxs.size == 0:
        #             continue
        #         num_batch = int(np.ceil(point_idxs.size / self.block_points))
        #         point_size = int(num_batch * self.block_points)    # point_size = 4096
        #         replace = False if (point_size - point_idxs.size <= point_idxs.size) else True
        #         point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
        #         point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
        #         np.random.shuffle(point_idxs)
        #         data_batch = points[point_idxs, :]
        #         normlized_xyz = np.zeros((point_size, 3))
        #         normlized_xyz[:, 0] = data_batch[:, 0] / coord_max[0]
        #         normlized_xyz[:, 1] = data_batch[:, 1] / coord_max[1]
        #         normlized_xyz[:, 2] = data_batch[:, 2] / coord_max[2]
        #         data_batch[:, 0] = data_batch[:, 0] - (s_x + self.block_size / 2.0)
        #         data_batch[:, 1] = data_batch[:, 1] - (s_y + self.block_size / 2.0)
        #         data_batch[:, 3:6] /= 255.0

        #         data_batch = np.concatenate((data_batch, normlized_xyz), axis=1)  # col = 9
        #         label_batch = labels[point_idxs].astype(int)
        #         batch_weight = self.labelweights[label_batch]

        #         data_motor = np.vstack([data_motor, data_batch]) if data_motor.size else data_batch  # under Test: data_mptpr[0]=[4.58282471e-01 -5.00000000e-01  5.65709167e+02  6.00000000e+00  1.00000000e+01  1.00000000e+01 -2.23334254e+00 -8.82852172e-01  7.09305037e-01]

        #         label_motor = np.hstack([label_motor, label_batch]) if label_motor.size else label_batch
        #         sample_weight = np.hstack([sample_weight, batch_weight]) if label_motor.size else batch_weight
        #         index_motor = np.hstack([index_motor, point_idxs]) if index_motor.size else point_idxs  # under Test: shape: += 4096
           # print('11111111111', index_motor.shape)                                                 # under Validation: First 73721, second 14220, end

        # data_motor = data_motor.reshape((-1, self.block_points, data_motor.shape[1]))
        # index_motor = index_motor.reshape((-1, self.block_points))
        # label_motor = label_motor.reshape((-1, self.block_points))
        # sample_weight = sample_weight.reshape((-1, self.block_points))
        return points, labels

    def __len__(self):
        return len(self.scene_points_list)


if __name__ == '__main__':
    data_root = '/data/yxu/PointNonLocal/data/stanford_indoor3d/'
    num_point, test_area, block_size, sample_rate = 4096, 'Validation', 1.0, 0.01

    point_data = S3DISDataset(split='train', data_root=data_root, num_point=num_point, test_area=test_area, block_size=block_size, sample_rate=sample_rate, transform=None)
    print('point data size:', point_data.__len__())
    print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
    print('point label 0 shape:', point_data.__getitem__(0)[1].shape)
    import torch, time, random
    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)
    train_loader = torch.utils.data.DataLoader(point_data, batch_size=16, shuffle=True, num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    for idx in range(4):
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i+1, len(train_loader), time.time() - end))
            end = time.time()