"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
# from data_utils.S3DISDataLoader import MotorDataset # ScannetDatasetGridMotor
# from data_utils.S3DISDataLoader import ScannetDatasetGridMotor
from data_utils.MotorDataLoader import ScannetDatasetwholeMotor
from data_utils.indoor3d_util import g_label2color
import torch
import torch.nn as nn
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
from models import dgcnn
import provider
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['clampingSystem', 'cover', 'gearContainer', 'charger', 'bottom', 'bolt']   # ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
          # 'board', 'clutter']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in testing [default: 32]')
    parser.add_argument('--gpu', type=str, default='1', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=4096, help='point number of one batch [default: 4096]')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--root', type=str, required=True, help='file need to be tested')
    parser.add_argument('--visual', action='store_true', default=True, help='visualize result [default: False]')
    parser.add_argument('--test_area', type=str, default='Validation', help='area for testing, option: 1-6 [default: 5]')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting [default: 5]')
    parser.add_argument('--k', type=int, default=20, metavar='N', help='Num of nearest neighbors to use')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N', choices=['cos', 'step'], help='Scheduler to use, [cos, step]')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N', help='Dimension of embeddings')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    return parser.parse_args()


def add_vote(vote_label_pool, point_idx, pred_label, weight):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b, n] != 0 and not np.isinf(weight[b, n]):
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
   # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/sem_seg/' + args.log_dir
    visual_dir = experiment_dir + '/visual/'
    visual_dir = Path(visual_dir)
    visual_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    NUM_CLASSES = 6
    BATCH_SIZE = args.batch_size
    NUM_POINT = args.num_point

    root = args.root # '/home/ies/hyu/data/Training_set_PointNet_0611/' # '/home/ies/hyu/dgcnn.pytorch/data/stanford_indoor3d/'  

    TEST_DATASET_WHOLE_SCENE = ScannetDatasetwholeMotor(root, split='test', test_area=args.test_area, block_points=NUM_POINT)
    log_string("The number of test data is: %d" % len(TEST_DATASET_WHOLE_SCENE))
    

    '''MODEL LOADING'''
    device = torch.device("cuda:0")
   # model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]

    MODEL = dgcnn.DGCNN_semseg(args).to(device)
    model = nn.DataParallel(MODEL)


    # classifier = MODEL.get_model(NUM_CLASSES).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    classifier = model.eval()

   # if args.test_area == 'Validation':
    with torch.no_grad():
        scene_id = TEST_DATASET_WHOLE_SCENE.file_list
        scene_id = [x[:-4] for x in scene_id]
        num_batches = len(TEST_DATASET_WHOLE_SCENE)

        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]

        log_string('---- EVALUATION WHOLE SCENE----')
        for batch_idx in range(num_batches):  # number of files
            print("Inference [%d/%d] %s ..." % (batch_idx + 1, num_batches, scene_id[batch_idx]))
            # total_seen_class_tmp = [0 for _ in range(NUM_CLASSES)]
            # total_correct_class_tmp = [0 for _ in range(NUM_CLASSES)]
            # total_iou_deno_class_tmp = [0 for _ in range(NUM_CLASSES)]
            if args.visual:
                fout = open(os.path.join(visual_dir, scene_id[batch_idx] + '_pred_.obj'), 'w')
                fout_gt = open(os.path.join(visual_dir, scene_id[batch_idx] + '_gt_2.obj'), 'w')

            whole_scene_data = TEST_DATASET_WHOLE_SCENE.scene_points_list[batch_idx]
            whole_scene_label = TEST_DATASET_WHOLE_SCENE.semantic_labels_list[batch_idx] 
            N_points = len(whole_scene_data)

            # vote_label_pool = np.zeros((whole_scene_label.shape[0], NUM_CLASSES))
            # for _ in tqdm(range(args.num_votes), total=args.num_votes):
                # scene_data, scene_label, scene_smpw, scene_point_index = TEST_DATASET_WHOLE_SCENE[batch_idx]
                # num_blocks = scene_data.shape[0]
                # s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE
                # batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 9))
            
            scene_data, scene_label = TEST_DATASET_WHOLE_SCENE[batch_idx]

            num_blocks = scene_data.shape[0]
            s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE
            batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 3))

            batch_label = np.zeros((BATCH_SIZE, NUM_POINT))
            pred_label = np.array([])
            batch_point_index = np.zeros((BATCH_SIZE, NUM_POINT))
            batch_smpw = np.zeros((BATCH_SIZE, NUM_POINT))
            for sbatch in range(s_batch_num):
                start_idx = sbatch * BATCH_SIZE
                end_idx = min((sbatch + 1) * BATCH_SIZE, num_blocks)
                real_batch_size = end_idx - start_idx   # = 32 or ?
                batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]
                batch_label[0:real_batch_size, ...] = scene_label[start_idx:end_idx, ...]
                # batch_point_index[0:real_batch_size, ...] = scene_point_index[start_idx:end_idx, ...]
                # batch_smpw[0:real_batch_size, ...] = scene_smpw[start_idx:end_idx, ...]
               # batch_data[:, :, 3:6] /= 1.0

                torch_data = torch.Tensor(batch_data)
                torch_data = torch_data.float().cuda()
                torch_data = torch_data.transpose(2, 1)
                seg_pred = classifier(torch_data)
                seg_pred = seg_pred.permute(0, 2, 1).contiguous()
                pred = seg_pred.max(dim=2)[1]
                batch_pred_label = pred.detach().cpu().numpy()   # B*N
                
            # pred_batch_label = batch_pred_label.reshape(-1, 1)
                pred_label = np.vstack([pred_label, batch_pred_label.reshape(-1,1)]) if pred_label.size else batch_pred_label.reshape(-1,1)
            
            pred_label = pred_label[0:N_points, :]
                
                # for i in range(real_batch_size):
                #     current_pts = scene_data[start_idx+i, :, :]
                #     current_label = scene_label[start_idx+i, :]
                #     current_pts[:,3:6] *= 255.0
                #     current_pred_label = batch_pred_label[i, :] 
                #     for k in range(NUM_POINT):
                #         color = g_label2color[current_pred_label[k]]
                #         color_gt = g_label2color[scene_label[start_idx+i, k]]
                #         if args.visual:
                #             fout.write('v %f %f %f %d %d %d\n' % (
                #                 current_pts[k, 0], current_pts[k, 1], current_pts[k, 2], color[0], color[1],
                #                 color[2]))
                #             fout_gt.write(
                #                 'v %f %f %f %d %d %d\n' % (
                #                 current_pts[k, 0], current_pts[k, 1], current_pts[k, 2], color_gt[0],
                #                 color_gt[1], color_gt[2]))
            #             filename = os.path.join(visual_dir, scene_id[batch_idx] + '.txt')
            #             with open(filename, 'w') as pl_save:
            #                 pl_save.write(str(int(k)) + '\n')
            # pl_save.close()
            # if args.visual:
            #     fout.close()
            #     fout_gt.close()
            filename = os.path.join(visual_dir, scene_id[batch_idx] + '.txt')
            with open(filename, 'w') as pl_save:
                for i in pred_label:
                    pl_save.write(str(int(i)) + '\n')
                pl_save.close()
            for n in range(N_points):
                color = g_label2color[int(pred_label[n])]
                color_gt = g_label2color[whole_scene_label[n]]
                if args.visual:
                    fout.write(
                        'v %f %f %f %d %d %d\n' % (
                            whole_scene_data[n, 0], whole_scene_data[n, 1], whole_scene_data[n, 2], color[0], color[1],
                            color[2]))
                    fout_gt.write(
                        'v %f %f %f %d %d %d\n' % (
                            whole_scene_data[n, 0], whole_scene_data[n, 1], whole_scene_data[n, 2], color_gt[0],
                            color_gt[1], color_gt[2]))
            if args.visual:
                fout.close()
                fout_gt.close()



       # IoU = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6)
        iou_per_class_str = '------- IoU --------\n'
        print(iou_per_class_str)
        # for l in range(NUM_CLASSES):
        #     iou_per_class_str += 'class %s, IoU: %.3f \n' % (
        #         seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])),
        #         total_correct_class[l] / float(total_iou_deno_class[l]))
        # log_string(iou_per_class_str)
       # log_string('eval point avg class IoU: %f' % np.mean(IoU))
        # log_string('eval whole scene point avg class acc: %f' % (
        #     np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))
        # log_string('eval whole scene point accuracy: %f' % ( 
        #         np.sum(total_correct_class) / float(np.sum(total_seen_class) + 1e-6)))
    print("Done!")


if __name__ == '__main__':
    args = parse_args()
    main(args)
