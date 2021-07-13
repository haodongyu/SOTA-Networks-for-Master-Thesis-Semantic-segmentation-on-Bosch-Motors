"""
Author: Benny
Date: Nov 2019
"""
import argparse
from operator import imod
import os
from data_utils.S3DISDataLoader import S3DISDataset
from data_utils.S3DISDataLoader import MotorDataset
import torch
import torch.optim as optim
import torch.nn as nn
import torchmetrics
from pytorch_lightning.core.lightning import LightningModule
import datetime
import logging
from pathlib import Path
import sys
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from models import dgcnn
from util import cal_loss, IOStream
import shutil
from tqdm import tqdm
import provider
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
tm_train_accuracy = torchmetrics.Accuracy()  # torchmetrics calculate accuracy
tm_val_accuracy = torchmetrics.Accuracy()

classes = ['clampingSystem', 'cover', 'gearContainer', 'charger', 'bottom', 'bolt']                        # ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
                                                                                                           # 'board', 'clutter']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='dgcnn', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', default=32, type=int, help='Epoch to run [default: 32]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='SGD', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int, default=4096, help='Point Number [default: 4096]')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--test_area', type=str, default='Validation', help='Which area to use for test, option: 1-6 [default: ]')
    parser.add_argument('--k', type=int, default=20, metavar='N', help='Num of nearest neighbors to use')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N', choices=['cos', 'step'], help='Scheduler to use, [cos, step]')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N', help='Dimension of embeddings')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')

    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
   # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('sem_seg')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    tensorBo_path = Path('./tensor/')
    tensorBo_path.mkdir(exist_ok=True)
    if args.log_dir is None:
        tensorBo_path = tensorBo_path.joinpath(timestr)
    else:
        tensorBo_path = tensorBo_path.joinpath(args.log_dir)
    tensorBo_path.mkdir(exist_ok=True)
    
    writer = SummaryWriter(BASE_DIR + '/tensor/' + args.log_dir)


    root = '/home/ies/hyu/data/Training_set_PointNet_0611/'
    NUM_CLASSES = 6
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size

    print("start loading training data ...")
    TRAIN_DATASET = MotorDataset(split='train', data_root=root, num_point=NUM_POINT, test_area=args.test_area, block_size=1.0, sample_rate=1.0, transform=None)
    print("start loading test data ...")
    TEST_DATASET = MotorDataset(split='test', data_root=root, num_point=NUM_POINT, test_area=args.test_area, block_size=1.0, sample_rate=1.0, transform=None)

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=8,
                                                  pin_memory=True, drop_last=True,
                                                  )
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=8,
                                                 pin_memory=True, drop_last=True)
    weights = torch.Tensor(TRAIN_DATASET.labelweights).cuda()

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    '''MODEL LOADING'''
    device = torch.device("cuda:0")
    MODEL = dgcnn.DGCNN_semseg(args).to(device)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    # shutil.copy('models/pointnet2_utils.py', str(experiment_dir))

   # model = MODEL.get_model(NUM_CLASSES).cuda()
    model = nn.DataParallel(MODEL)

    if args.optimizer == 'SGD':
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.learning_rate*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epoch, eta_min=1e-3)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, 20, 0.5, args.epoch)

    criterion = cal_loss

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
       # model = model.apply(weights_init)

    # if args.optimizer == 'Adam':
    #     optimizer = torch.optim.Adam(
    #         model.parameters(),
    #         lr=args.learning_rate,
    #         betas=(0.9, 0.999),
    #         eps=1e-08,
    #         weight_decay=args.decay_rate
    #     )
    # else:
    #     optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    # def bn_momentum_adjust(m, momentum):
    #     if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
    #         m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    # MOMENTUM_ORIGINAL = 0.1
    # MOMENTUM_DECCAY = 0.5
    # MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0
    best_iou = 0

    for epoch in range(start_epoch, args.epoch):
        '''Train on chopped scenes'''
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in opt.param_groups:
            param_group['lr'] = lr
        # momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        # if momentum < 0.01:
        #     momentum = 0.01
        # print('BN momentum updated to: %f' % momentum)
        # model = model.apply(lambda x: bn_momentum_adjust(x, momentum))
        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        model = model.train()

        for i, (points, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            opt.zero_grad()

            points = points.data.numpy()
            points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
            points = torch.Tensor(points)
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)
            

            seg_pred = model(points)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()

            batch_label = target.cpu().data.numpy()
            target_to_metric = target.cpu()
            target = target.view(-1, 1)[:, 0]
            loss = criterion(seg_pred.view(-1, 13), target)
            loss.backward()
            opt.step()

            pred = seg_pred.max(dim=2)[1]
            pred_cpu = pred.cpu() 
            pred_np = pred.detach().cpu().numpy()
           # print('111111111size of pred_np', pred_np.shape)
           # print('22222222222size of batch label', batch_label.shape)
           # pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_np == batch_label)   ####### ???
            total_correct += correct
            total_seen += (BATCH_SIZE * NUM_POINT)
            loss_sum += loss
        log_string('Training mean loss: %f' % (loss_sum / num_batches))
        log_string('Training accuracy: %f' % (total_correct / float(total_seen)))
        train_accuracy = tm_train_accuracy(pred_cpu, target_to_metric)
        ######### calculate metric ############
        # log_string('Training accuracy for metric: %f' % (train_accuracy))
        # tm_train_accuracy.reset()
        

        ###### write the tensorboard ################
        writer.add_scalar('Training mean loss', (loss_sum / num_batches), epoch)
        writer.add_scalar('Training accuracy', (total_correct / float(total_seen)), epoch)

        if epoch % 5 == 0:
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        '''Evaluate on chopped scenes'''
        with torch.no_grad():
            num_batches = len(testDataLoader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            labelweights = np.zeros(NUM_CLASSES)
            total_seen_class = [0 for _ in range(NUM_CLASSES)]
            total_correct_class = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
            noBG_seen_class = [0 for _ in range(NUM_CLASSES-1)]
            noBG_correct_class = [0 for _ in range(NUM_CLASSES-1)]
            noBG_iou_deno_class = [0 for _ in range(NUM_CLASSES-1)]
            model = model.eval()

            log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            for i, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                points = points.data.numpy()
                points = torch.Tensor(points)
                points, target = points.float().cuda(), target.long().cuda()
                points = points.transpose(2, 1)

                seg_pred = model(points)
                pred_val = seg_pred.contiguous().cpu().data.numpy()
              #  seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
                seg_pred = seg_pred.permute(0, 2, 1).contiguous()

                batch_label = target.cpu().data.numpy()
                target_to_metric1 = target
                target = target.view(-1, 1)[:, 0]
                loss = criterion(seg_pred.view(-1, 13), target)
                loss_sum += loss
               # pred_val = np.argmax(pred_val, 2)
                pred = seg_pred.max(dim=2)[1]
                pred_np = pred.detach().cpu().numpy()
                correct = np.sum((pred_np == batch_label))
                total_correct += correct
                total_seen += (BATCH_SIZE * NUM_POINT)
                tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
                labelweights += tmp

               # print("11111111111 size of pred_val", pred_np.shape)
               # print("22222222222 size of batch_label", batch_label.shape)

                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum((pred_np == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum(((pred_np == l) | (batch_label == l)))

                ####### calculate without Background ##############
                for l in range(1, NUM_CLASSES):
                    noBG_seen_class[l-1] += np.sum((batch_label == l))
                    noBG_correct_class[l-1] += np.sum((pred_np == l) & (batch_label == l))
                    noBG_iou_deno_class[l-1] += np.sum(((pred_np == l) | (batch_label == l)))

            labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float64) + 1e-6))
            log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
            log_string('eval point avg class IoU: %f' % (mIoU))
            log_string('eval point accuracy: %f' % (total_correct / float(total_seen)))
            log_string('eval point avg class acc: %f' % (
                np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float64) + 1e-6))))
            
            ########## metrics ################
            # val_accuracy = tm_val_accuracy(pred, target_to_metric1)
            # log_string('eval point accuracy for metric: %f' % (val_accuracy)) 
            # tm_val_accuracy.reset()

            ####### log without Background #################
            noBG_mIoU = np.mean(np.array(noBG_correct_class) / (np.array(noBG_iou_deno_class, dtype=np.float64) + 1e-6))
            log_string('eval point avg class IoU without background: %f' % (noBG_mIoU))
            log_string('eval point accuracy without background: %f' % (sum(noBG_correct_class) / float(sum(noBG_seen_class))))
            log_string('eval point avg class acc without background: %f' % (
                np.mean(np.array(noBG_correct_class) / (np.array(noBG_seen_class, dtype=np.float64) + 1e-6))))

            iou_per_class_str = '------- IoU --------\n'
            for l in range(NUM_CLASSES):
                iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
                    seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), labelweights[l],
                    total_correct_class[l] / float(total_iou_deno_class[l]))

            log_string(iou_per_class_str)
            log_string('Eval mean loss: %f' % (loss_sum / num_batches))
            log_string('Eval accuracy: %f' % (total_correct / float(total_seen)))

            if mIoU >= best_iou:
                best_iou = mIoU
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'class_avg_iou': mIoU,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Saving model....')
            log_string('Best mIoU: %f' % best_iou)
        ############ write tensor #####################
        writer.add_scalar('Validation mean loss', (loss_sum / num_batches), epoch)
        writer.add_scalar('Validation accuracy', (total_correct / float(total_seen)), epoch)

        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)