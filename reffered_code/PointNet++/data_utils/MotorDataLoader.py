import os
import numpy as np
from numpy.random import choice
from tqdm import tqdm
from torch.utils.data import Dataset

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
   # print('center of this point cloud is:', centroid)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def down_Sample(array_1, array_2, pointsToKeep): 
    row_total = array_1.shape[0]
    row_sequence= np.random.choice(row_total, pointsToKeep, replace=False)
    return array_1[row_sequence,:], array_2[row_sequence]


class MotorDataset(Dataset):  # for train
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

        # add downsample process #
       # points,labels = down_Sample(points, labels, int(np.random.uniform(0.4, 0.8)*len(points)))

        N_points = points.shape[0]

        # normalize
       # normalized_points = np.zeros((N_points, 3))
        normalized_points = pc_normalize(points[:, :3])
       
       # current_points = np.concatenate((normalized_points, points[:, 3:]), axis=1)
        current_labels = labels

        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)
        
        # resample
        choice = np.random.choice(N_points, self.num_point, replace=True)
        current_points = normalized_points[choice, :]    #   4096*6 / *3
        current_labels = current_labels[choice]
       # print('1111111111current point size after normalized: ', current_points.shape)

        return current_points, current_labels

    def __len__(self):
        return len(self.motor_idxs)



class ScannetDatasetwholeMotor():  # for test
    # prepare to give prediction on each points
    def __init__(self, root, block_points=4096, split='test', test_area='Validation', block_size=50.0, padding=0.001):
        self.block_points = block_points
        self.block_size = block_size
        self.padding = padding
        self.root = root
        self.split = split
        self.test_area = test_area
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
            self.scene_points_list.append(data[:, :6])    # num_files*num_points*6
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
       # print('size of init_points[]', point_set_ini.shape)
        points = point_set_ini[:,:6] 
       # print('size of points[]', points.shape)
        labels_o = self.semantic_labels_list[index]
       # coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
        N_points = points.shape[0]
        '''
        under Test: coor_min = [-255.2752533  -110.75037384  519.2489624 ]
                    coor_max = [ 45.58950043 125.44611359 797.55413818]
        under Validation: coor_min = [-0.22853388 -0.6627815  -3.93524766]
                          coor_max = [ 0.76870452  0.74199943 -2.73275002]
        '''
        normalized_motor = pc_normalize(points[:, :3])
       # normalized_motor = np.concatenate((normalized_motor, points[:, 3:]), axis=1)  # num_point * 6
        normalized_motor = np.hstack((normalized_motor, labels_o.reshape(len(labels_o), 1))) # num_point * 4 / 7
        np.random.shuffle(normalized_motor)
        labels = normalized_motor[:, -1]
        normalized_motor = normalized_motor[:,:3]


        # normalize  with RGB
       # current_points = np.concatenate((normalized_motor, points[:, 3:]), axis=1)  # num_point * 6


        ### pad 0 into last block, which not enough to 4096 ###
        num_block= divmod(N_points, self.block_points)  # num and res
        # data_motor = np.zeros((110, 4096, 9))
        # data_label = np.zeros((110, 4096))
        data_motor = normalized_motor[0 : num_block[0]*self.block_points, :].reshape((num_block[0], self.block_points, normalized_motor.shape[1]))  # num_block*N*3
        data_label = labels[0:num_block[0]*self.block_points].reshape(-1, self.block_points)
        if num_block[1]:
            block_res = np.zeros((self.block_points, 3))
            label_res = np.zeros(self.block_points)
            block_res[0:num_block[1], :] = normalized_motor[N_points-num_block[1]:, :]
            label_res[0:num_block[1]] = labels[N_points-num_block[1]:]
            block_res = np.array([block_res])
            data_motor = np.vstack([data_motor, block_res])
            data_label = np.vstack([data_label, label_res])
       # labels = labels.reshape((-1, self.block_points))  # num_block*N
    
       # print('current data size after normalized: ', data_motor[0].shape)
        return data_motor, data_label


    def __len__(self):
        return len(self.scene_points_list)


class ScannetDatasetBatchMotor():  # for test
    # prepare to give prediction on each points
    def __init__(self, root, block_points=4096, split='test', test_area='Validation', block_size=50.0, padding=0.001):
        self.mode = 'gama'
        self.block_points = block_points
        self.block_size = block_size
        self.padding = padding
        self.root = root
        self.split = split
        self.test_area = test_area
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
            self.scene_points_list.append(data[:, :6])    # num_files*num_points*6
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
       # print('size of init_points[]', point_set_ini.shape)
        points = point_set_ini[:,:6] 
       # print('size of points[]', points.shape)
        labels_o = self.semantic_labels_list[index]
       # coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
        N_points = points.shape[0]
        '''
        under Test: coor_min = [-255.2752533  -110.75037384  519.2489624 ]
                    coor_max = [ 45.58950043 125.44611359 797.55413818]
        under Validation: coor_min = [-0.22853388 -0.6627815  -3.93524766]
                          coor_max = [ 0.76870452  0.74199943 -2.73275002]
        '''
        normalized_motor = pc_normalize(points[:, :3])
       # normalized_motor = np.concatenate((normalized_motor, points[:, 3:]), axis=1)  # num_point * 6
        normalized_motor = np.hstack((normalized_motor, labels_o.reshape(len(labels_o), 1))) # num_point * 4 / 7
        np.random.shuffle(normalized_motor)
        labels = normalized_motor[:, -1]
        normalized_motor = normalized_motor[:,:3]

        ### pad 0 into last block, which not enough to 4096 ###
        num_block= divmod(N_points, self.block_points)  # num and res
        # data_motor = np.zeros((110, 4096, 3))
        # data_label = np.zeros((110, 4096))
        data_motor = normalized_motor[0 : num_block[0]*self.block_points, :].reshape((num_block[0], self.block_points, normalized_motor.shape[1]))  # num_block*N*3
        data_label = labels[0:num_block[0]*self.block_points].reshape(-1, self.block_points)
        if num_block[1]:
            block_res = np.zeros((self.block_points, 3))
            label_res = np.zeros(self.block_points)
            block_res[0:num_block[1], :] = normalized_motor[N_points-num_block[1]:, :]
            label_res[0:num_block[1]] = labels[N_points-num_block[1]:]
            block_res = np.array([block_res])
            data_motor = np.vstack([data_motor, block_res])
            data_label = np.vstack([data_label, label_res])
       # labels = labels.reshape((-1, self.block_points))  # num_block*N
    
       # print('current data size after normalized: ', data_motor[0].shape)
        if self.mode == 'alpha':
            return data_motor[:32, :, :], data_label[:32, :]
        elif self.mode == 'beta':
            batch_motor, batch_label = np.zeros((32, 4096, 3)), np.zeros((32, 4096))
            if len(data_motor) >= 32 :
                batch_motor[:, :2048, :], batch_label[:, :2048] = data_motor[:32, :2048, :], data_label[:32, :2048]
                return batch_motor, batch_label
            else:
                batch_motor[:len(data_motor), :2048, :], batch_label[:len(data_motor), :2048] = data_motor[:, :2048, :], data_label[:, :2048]
                return batch_motor, batch_label
        elif self.mode == 'gama' :
            batch_motor, batch_label = np.zeros((32, 4096, 3)), np.zeros((32, 4096))
            if len(data_motor) >= 32 :
                batch_motor[:, 2048:, :], batch_label[:, 2048:] = data_motor[:32, 2048:, :], data_label[:32, 2048:]
                return batch_motor, batch_label
            else:
                batch_motor[:len(data_motor), 2048:, :], batch_label[:len(data_motor), 2048:] = data_motor[:, 2048:, :], data_label[:, 2048:]
                return batch_motor, batch_label



    def __len__(self):
        return len(self.scene_points_list)


class ScannetDatasetNoLabelMotor():  # for test
    # prepare to give prediction on each points
    def __init__(self, root, block_points=4096, split='test', test_area='Validation', block_size=50.0, padding=0.001):
        self.block_points = block_points
        self.block_size = block_size
        self.padding = padding
        self.root = root
        self.split = split
        self.test_area = test_area
        self.scene_points_num = []
        assert split in ['train', 'test']
        if self.split == 'train':
            self.file_list = [d for d in os.listdir(root) if d.find('%s' % test_area) is -1]
        else:
            self.file_list = [d for d in os.listdir(root) if d.find('%s' % test_area) is not -1]
        self.scene_points_list = []

        self.motor_coord_min, self.motor_coord_max = [], []
        for file in self.file_list:
            data = np.load(root + file)
            points = data[:, :3]
            self.scene_points_list.append(data[:, :6])    # num_files*num_points*6


            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.motor_coord_min.append(coord_min), self.motor_coord_max.append(coord_max)

    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]
       # print('size of init_points[]', point_set_ini.shape)
        points = point_set_ini[:,:6] 
       # print('size of points[]', points.shape)
 
       # coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
        N_points = points.shape[0]
        '''
        under Test: coor_min = [-255.2752533  -110.75037384  519.2489624 ]
                    coor_max = [ 45.58950043 125.44611359 797.55413818]
        under Validation: coor_min = [-0.22853388 -0.6627815  -3.93524766]
                          coor_max = [ 0.76870452  0.74199943 -2.73275002]
        '''
        normalized_motor = pc_normalize(points[:, :3])
       # normalized_motor = np.concatenate((normalized_motor, points[:, 3:]), axis=1)  # num_point * 6
       # normalized_motor = np.hstack((normalized_motor, labels_o.reshape(len(labels_o), 1))) # num_point * 4 / 7
        np.random.shuffle(normalized_motor)
       # labels = normalized_motor[:, -1]
        normalized_motor = normalized_motor[:,:3]


        ### pad 0 into last block, which not enough to 4096 ###
        num_block= divmod(N_points, self.block_points)  # num and res
        data_motor = normalized_motor[0 : num_block[0]*self.block_points, :].reshape((num_block[0], self.block_points, normalized_motor.shape[1]))  # num_block*N*3
       # data_label = labels[0:num_block[0]*self.block_points].reshape(-1, self.block_points)
        if num_block[1]:
            block_res = np.zeros((self.block_points, 3))
           # label_res = np.zeros(self.block_points)
            block_res[0:num_block[1], :] = normalized_motor[N_points-num_block[1]:, :]
           # label_res[0:num_block[1]] = labels[N_points-num_block[1]:]
            block_res = np.array([block_res])
            data_motor = np.vstack([data_motor, block_res])
           # data_label = np.vstack([data_label, label_res])
    
       # print('current data size after normalized: ', data_motor[0].shape)
        return data_motor


    def __len__(self):
        return len(self.scene_points_list)


if __name__ == '__main__':
    data_root = '/data/yxu/PointNonLocal/data/stanford_indoor3d/'
    num_point, test_area, block_size, sample_rate = 4096, 'Validation', 1.0, 0.01

    point_data = MotorDataset(split='train', data_root=data_root, num_point=num_point, test_area=test_area, block_size=block_size, sample_rate=sample_rate, transform=None)
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
