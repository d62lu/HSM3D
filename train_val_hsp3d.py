"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
from sklearn.metrics import confusion_matrix
from data_utils.DataLoader import Toronto3D_HSP_Dataset_fast_fps_knn
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
import time
from ErrorMatrix import ConfusionMatrix
from torch.nn import DataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
import gc
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = BASE_DIR
# sys.path.append(os.path.join(ROOT_DIR))
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

import numpy as np
from collections import Counter


def compute_iou_per_class(pred, gt, num_classes=None):
    """
    计算每个类别的 IoU 以及 mIoU

    参数：
    - pred: [N, 1]，预测标签
    - gt:   [N, 1]，真实标签
    - num_classes: 类别总数（若为 None，则自动从 gt/pred 中推测）

    返回：
    - iou_per_class: dict{class_id: IoU}
    - miou: 所有类 IoU 的平均值（仅包括出现过的类）
    """
    pred = pred.flatten()
    gt = gt.flatten()

    if num_classes is None:
        num_classes = max(pred.max(), gt.max()) + 1

    iou_per_class = {}
    for cls in range(num_classes):
        pred_mask = pred == cls
        gt_mask = gt == cls

        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()

        if union == 0:
            continue  # 该类未出现于 gt 或 pred 中，不计入 mIoU

        iou = intersection / union
        iou_per_class[cls] = iou

    miou = np.mean(list(iou_per_class.values())) if iou_per_class else 0.0
    return iou_per_class, miou


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='hsm3d_model', help='model name [default: hsm3d_model]')
    parser.add_argument('--batch_size', type=int, default=1, help='Fixed Batch Size during training [default: 1], not suitable for other batch size')
    parser.add_argument('--epoch', default=200, type=int, help='Epoch to run [default: 200]')
    parser.add_argument('--learning_rate', default=0.01, type=float, help='Initial learning rate [default: 0.01]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--NUM_class', type=str, default='8', help='NUMBER of classes [default: 8]')
    parser.add_argument('--optimizer', type=str, default='SGD', help='Adam or SGD [default: SGD]')
    parser.add_argument('--log_dir', type=str, default='HSM3D', help='Log path [default: HSM3D]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int, default=1000000, help='Point Number [default: 1000000]')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.8, help='Decay rate for lr decay [default: 0.8]')
    parser.add_argument('--weighted_loss', type=bool, default=True, help='weighted loss') #
    parser.add_argument('--grid_size', type=float, default=1/1024, help='grid_size [default: 1/1024]')
    parser.add_argument('--dataset', type=str, default='toronto3d', help='dataset root [default: ../data/]')
    parser.add_argument('--sample_overlap', type=float, default=0.985, help='overlap in selecting superpoint sequence')
    parser.add_argument('--ca_K', type=int, default=10, help='number of nearest neighbors for cross-attention-based superpoint refinement, fixed')
    parser.add_argument('--sm_K', type=int, default=10, help='number of nearest neighbors for cross-attention-based superpoint refinement, 5-50')
    parser.add_argument('--test_demo', type=str, default='train', help='train or test. if test, then the test set is used for training to speed up the data loading')
    parser.add_argument('--p_dim', type=int, default=13, help='input point dimension: x y z r g b intensity geof1 geof2 geof3 geof4 sp_index label')


    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True, parents=True)
    experiment_dir = experiment_dir.joinpath('HSM3D', args.dataset)
    experiment_dir.mkdir(exist_ok=True, parents=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True, parents=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True, parents=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True, parents=True)

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
    
    NUM_CLASSES = args.NUM_class
    In_CHANNEL = args.p_dim - 2

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))

    classifier = MODEL.HSM3D(num_class=NUM_CLASSES, grid_size = args.grid_size, in_channels = In_CHANNEL, sm_K=args.sm_K).cuda()
    classifier = DataParallel(classifier)
    criterion = MODEL.get_loss().cuda()

    
    
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    
    if args.weighted_loss == True:
        print("Use weighted loss ...")
        criterion = MODEL.get_loss_weighted().cuda()
    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        # start_epoch = checkpoint['epoch']
        start_epoch = 0
        classifier.load_state_dict(checkpoint['model_state_dict'])
        eta_min_ratio = 0.01
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)
        eta_min_ratio = 0.01
    
    
    BATCH_SIZE = args.batch_size

    

            
    classes = ['Ground','Road_markings','Natural','Building','Utility_line','Pole', 'Car','Fence'] 
    class2label = {cls: i for i, cls in enumerate(classes)}
    seg_classes = class2label
    seg_label_to_cat = {}
    for i, cat in enumerate(seg_classes.keys()):
        seg_label_to_cat[i] = cat
    

    root = 'data/' + args.dataset
    args.data_root = root     
    print("start loading training data ...")
    if args.test_demo == 'train':
        TRAIN_DATASET = Toronto3D_HSP_Dataset_fast_fps_knn(args, split='train', label_number=NUM_CLASSES)
    else:
        TRAIN_DATASET = Toronto3D_HSP_Dataset_fast_fps_knn(args, split='test', label_number=NUM_CLASSES)
    print("start loading test data ...")
    TEST_DATASET = Toronto3D_HSP_Dataset_fast_fps_knn(args, split='test',  label_number=NUM_CLASSES)

    weight = torch.tensor(TRAIN_DATASET.labelweights).cuda()
    print('label weights:', weight)



    

    



    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                                                  pin_memory=True, drop_last=True,
                                                  worker_init_fn=lambda x: np.random.seed(x + int(time.time())))
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
                                                 pin_memory=True, drop_last=False)


    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of val data is: %d" % len(TEST_DATASET))

    
 

    
    
    
    
    

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.decay_rate)
    
    scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=args.learning_rate * eta_min_ratio)


    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum


    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0
    best_iou = 0


    for epoch in range(start_epoch, args.epoch):
        '''Train on chopped scenes'''
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        
        '''Adjust learning rate and BN momentum'''

        logger.info('Learning rate is: %f' %(optimizer.state_dict()['param_groups'][0]['lr']))

        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        num_batches = len(trainDataLoader)
        print('The number of training batches is: %d' % num_batches)

        classifier = classifier.train()


        pbar = tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9)

        scaler = torch.cuda.amp.GradScaler()

        for i, (points, neightbor_sp_idx_list, gt_labels, _) in pbar:

            
            optimizer.zero_grad()

            pbar.set_postfix({'spnum': torch.unique(neightbor_sp_idx_list[:, 0]).shape[0]})

            if args.test_demo == 'train':
                points = points.data.numpy()
                points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
                points[:, :, :3] = provider.random_scale_point_cloud(points[:, :, :3])
                points = torch.Tensor(points)



            points, gt_labels, neightbor_sp_idx_array = points.float().cuda(), gt_labels.long().cuda(), neightbor_sp_idx_list.long().cuda()
            points = points.transpose(2, 1)
            target = gt_labels
            B, N = target.shape



            with torch.cuda.amp.autocast():
    
                start_time = time.time()
                pre_total = classifier(points, neightbor_sp_idx_array).contiguous().view(-1, NUM_CLASSES)
    
                target = target.view(-1, 1)[:, 0]

                
                loss = criterion(pre_total, target, weight)


            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


            
            gc.collect()
            torch.cuda.empty_cache()

        if args.test_demo == 'train':
            scheduler.step()
        '''Evaluate on chopped scenes'''
       
        with torch.no_grad():

                num_batches = len(testDataLoader)

                classifier = classifier.eval()

                log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
    
                cat_names = classes

                confusion = ConfusionMatrix(num_classes=NUM_CLASSES, labels=cat_names)
                confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)



                pbar = tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9)

                for i, (points, neightbor_sp_idx_list, gt_labels, ab_sp_idx) in pbar:
        
                    pbar.set_postfix({'spnum': torch.unique(neightbor_sp_idx_list[:, 0]).shape[0]})
                    start_time = time.time()
        
                    

                    points, gt_labels, neightbor_sp_idx_array = points.float().cuda(), gt_labels.long().cuda(), neightbor_sp_idx_list.long().cuda()
                    points = points.transpose(2, 1)
 
                    target = gt_labels


                    
                    pre_list = classifier(points, neightbor_sp_idx_array)
          
                    pred = pre_list.argmax(dim=-1).cpu().numpy()            # (B, N)
                    label = gt_labels.cpu().numpy()                         # (B, N)

                    # 计算 confusion matrix
                    for b in (range(pred.shape[0])):
                 
                        idx = np.arange(pred[b].shape[0])
                        pred_b = pred[b][idx].reshape(-1)
                        label_b = label[b][idx].reshape(-1)

                        confusion += confusion_matrix(label_b, pred_b, labels=list(range(NUM_CLASSES)))
                    del label_b, pred_b
                    gc.collect()


                # 计算 per-class IoU
                intersection = np.diag(confusion)
                union = (confusion.sum(0) + confusion.sum(1) - intersection)
                iou_per_class = intersection / (union + 1e-6)
                mIoU = np.mean(iou_per_class)

   

                log_string('eval point avg class IoU: %f' % (mIoU))


                if mIoU >= best_iou:

                    best_iou = mIoU

                    logger.info('Save model...')
                    savepath = str(checkpoints_dir) + '/best_model.pth'
                    log_string('Saving at %s' % savepath)
                    state = {
                        'epoch': epoch,
                        'class_avg_iou': mIoU,
       
                        'model_state_dict': classifier.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
     
                    }
                    torch.save(state, savepath)
                    log_string('Saving model....')


                log_string('Best mIoU: %f' % best_iou)
       
        gc.collect()
        torch.cuda.empty_cache()
        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)
