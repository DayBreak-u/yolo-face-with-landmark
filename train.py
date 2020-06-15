import argparse

import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

import test  # import test.py to get mAP after each epoch
from nn import *
from utils.datasets import *
from utils.utils import *
from hyp import  hyp

wdir = 'weights'
# Print focal loss if gamma > 0
if hyp['fl_gamma']:
    print('Using FocalLoss(gamma=%g)' % hyp['fl_gamma'])


def train(hyp):
    assert opt.net in ['mbv3_small_1', 'mbv3_small_75', 'mbv3_large_1', 'mbv3_large_75',
                       "mbv3_large_75_light", "mbv3_large_1_light", 'mbv3_small_75_light', 'mbv3_small_1_light',

                      ]


    epochs = opt.epochs  # 500200 batches at bs 64, 117263 images = 273 epochs
    batch_size = opt.batch_size
    accumulate = max(round(64 / batch_size), 1)  # accumulate n times before optimizer update (bs 64)
    weights = opt.weights  # initial training weights
    backbone_weights = opt.backbone_weights  # initial training weights

    imgsz_min, imgsz_max, imgsz_test = opt.img_size  # img sizes (min, max, test)

    # Image Sizes
    gs = 64  # (pixels) grid size
    assert math.fmod(imgsz_min, gs) == 0, '--img-size %g must be a %g-multiple' % (imgsz_min, gs)
    opt.multi_scale |= imgsz_min != imgsz_max  # multi if different (min, max)
    if opt.multi_scale:
        if imgsz_min == imgsz_max:
            imgsz_min //= 1.5
            imgsz_max //= 0.667
        grid_min, grid_max = imgsz_min // gs, imgsz_max // gs
        imgsz_min, imgsz_max = int(grid_min * gs), int(grid_max * gs)
    img_size = imgsz_max  # initialize with max size

    # Configure run
    init_seeds()

    train_path = opt.train_path



    best = wdir + 'best.pt'
    results_file = 'results.txt'

    # Remove previous results
    for f in glob.glob('*_batch*.jpg') + glob.glob(results_file):
        os.remove(f)

    # Initialize model


    if opt.net.startswith("mbv3_small_1"):
        backone = mobilenetv3_small(backbone_weights)
    elif opt.net.startswith("mbv3_small_75"):
        backone = mobilenetv3_small(backbone_weights,width_mult = 0.75)
    elif opt.net.startswith("mbv3_large_1"):
        backone = mobilenetv3_large(backbone_weights)
    elif opt.net.startswith("mbv3_large_75"):
        backone = mobilenetv3_large(backbone_weights,width_mult = 0.75)


    if 'light' in opt.net:
        model = DarknetWithShh(backone, hyp, light_head=True).to(device)
    else:
        model = DarknetWithShh(backone, hyp).to(device)

    # Optimizer
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if '.bias' in k:
            pg2 += [v]  # biases
        elif 'Conv2d.weight' in k:
            pg1 += [v]  # apply weight_decay
        else:
            pg0 += [v]  # all else

    if opt.adam:
        hyp['lr0'] *= 0.1  # reduce lr (i.e. SGD=5E-3, Adam=5E-4)
        optimizer = optim.Adam(pg0, lr=hyp['lr0'])
        # optimizer = AdaBound(pg0, lr=hyp['lr0'], final_lr=0.1)
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    print('Optimizer groups: %g .bias, %g Conv2d.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    start_epoch = 0
    best_fitness = 0.0

    if weights.endswith('.pt'):  # pytorch format
        chkpt = torch.load(weights, map_location=device)

        # load model
        try:
            chkpt['model'] = {k: v for k, v in chkpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(chkpt['model'], strict=False)
        except KeyError as e:
            s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
                "See https://github.com/ultralytics/yolov3/issues/657" % (opt.weights, opt.cfg, opt.weights)
            raise KeyError(s) from e
        if opt.resume:
            # load optimizer
            if chkpt['optimizer'] is not None:
                optimizer.load_state_dict(chkpt['optimizer'])
                best_fitness = chkpt['best_fitness']

            # load results
            if chkpt.get('training_results') is not None:
                with open(results_file, 'w') as file:
                    file.write(chkpt['training_results'])  # write results.txt

            start_epoch = chkpt['epoch'] + 1
        del chkpt


    # Mixed precision training https://github.com/NVIDIA/apex


    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.95 + 0.05  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch - 1  # see link below
    # https://discuss.pytorch.org/t/a-problem-occured-when-resuming-an-optimizer/28822

    # Plot lr schedule
    # y = []
    # for _ in range(epochs):
    #     scheduler.step()
    #     y.append(optimizer.param_groups[0]['lr'])
    # plt.plot(y, '.-', label='LambdaLR')
    # plt.xlabel('epoch')
    # plt.ylabel('LR')
    # plt.tight_layout()
    # plt.savefig('LR.png', dpi=300)

    # Initialize distributed training
    if device.type != 'cpu' and torch.cuda.device_count() > 1 and torch.distributed.is_available():
        dist.init_process_group(backend='nccl',  # 'distributed backend'
                                init_method='tcp://127.0.0.1:9999',  # distributed training init method
                                world_size=1,  # number of nodes for distributed training
                                rank=0)  # distributed training node rank
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        model.yolo_layers = model.module.yolo_layers  # move yolo layer indices to top level

    # Dataset
    dataset = LoadImagesAndLabels(train_path, img_size, batch_size,
                                  augment=True,
                                  hyp=hyp,  # augmentation hyperparameters
                                  rect=opt.rect,  # rectangular training
                                  cache_images=opt.cache_images,
                                  single_cls=opt.single_cls,
                                  point_number=hyp['point_num'],
                                  flip_idx_pair=hyp['flip_idx_pair']
                                  )

    # Dataloader
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             shuffle=not opt.rect,  # Shuffle=True unless rectangular training is used
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    # Testloader


    # Model parameters

    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)


    # Model EMA
    ema = torch_utils.ModelEMA(model)

    # Start training
    nb = len(dataloader)  # number of batches
    n_burn = max(3 * nb, 500)  # burn-in iterations, max(3 epochs, 500 iterations)

    # torch.autograd.set_detect_anomaly(True)
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    t0 = time.time()
    print('Image sizes %g - %g train, %g test' % (imgsz_min, imgsz_max, imgsz_test))
    print('Using %g dataloader workers' % nw)
    print('Starting training for %g epochs...' % epochs)
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()


        mloss = torch.zeros(5).to(device)  # mean losses
        print(('\n' + '%10s' * 10) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls','land' , 'total', 'targets', 'img_size','lr'))
        pbar = tqdm(enumerate(dataloader), total=nb , ncols = 50)  # progress bar
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device).float() / 255.0

            targets = targets.to(device)

            # Burn-in
            if ni <= n_burn:
                xi = [0, n_burn]  # x interp
                model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
                accumulate = max(1, np.interp(ni, xi, [1, 64 / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    x['weight_decay'] = np.interp(ni, xi, [0.0, hyp['weight_decay'] if j == 1 else 0.0])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [0.9, hyp['momentum']])

            # Multi-Scale
            if opt.multi_scale:
                if ni / accumulate % 1 == 0:  #  adjust img_size (67% - 150%) every 1 batch
                    img_size = random.randrange(grid_min, grid_max + 1) * gs
                sf = img_size / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to 32-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            pred = model(imgs)

            #deBug
            # img = imgs[0].cpu().numpy()
            # img = np.swapaxes(img,0,2)
            # img = np.swapaxes(img,0,1)
            # img *= 255
            #
            # labels = targets.cpu().numpy()
            # # print(labels)
            # labels = labels[labels[:,0] == 0 ]
            # # print("\n",labels)
            # #
            # img = np.ascontiguousarray(img)
            # img_h,img_w,_ = img.shape
            #
            # for label in labels:
            #     label = label[1:]
            #     cx,cy,w,h = label[1:5]
            #     x1,y1,x2,y2 = cx - w/2,cy-h/2,cx+w/2,cy+h/2
            #     box = [int(x1*img_w),int(y1*img_h),int(x2*img_w),int(y2*img_h)]
            #
            #     cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(0,255,0))
            #     for i in range(hyp["point_num"]):
            #         cv2.circle(img, (int(label[5+i*2]*img_w), int(label[5+i*2+1]*img_h)), 1, (0, 0, 255), 4)
            #
            # cv2.imwrite("debug_imgs/{}.jpg".format(ni), img)

            # Loss
            loss, loss_items = compute_loss(pred, targets, model,point_number=hyp['point_num'])
            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss_items)
                return results

            # Backward
            loss *= batch_size / 64  # scale loss

            loss.backward()

            # Optimize
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
                ema.update(model)

            # Print
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)

            s = ('%10s' * 2 + '%10.3g' * 7 + "%10.5g") % ('%g/%g' % (epoch, epochs - 1), mem, *mloss, len(targets), img_size,scheduler.get_lr()[0])
            pbar.set_description(s)


            # end batch ------------------------------------------------------------------------------------------------

        # Update scheduler
        scheduler.step()

        # Process epoch results
        ema.update_attr(model)
        final_epoch = epoch + 1 == epochs


        # Tensorboard
        if tb_writer:
            tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss',
                    'train/land_loss', ]
            for x, tag in zip(list(mloss[:-1]) , tags):
                tb_writer.add_scalar(tag, x, epoch)



        chkpt = {'epoch': epoch,
                 'best_fitness': best_fitness,
                 'model': ema.ema.module.state_dict() if hasattr(model, 'module') else ema.ema.state_dict(),
                 'optimizer': None if final_epoch else optimizer.state_dict()}

        # Save last, best and delete
        torch.save(chkpt, "{}/{}_last.pt".format(wdir,opt.net))
        if  final_epoch:
            torch.save(chkpt, "{}/{}_final.pt".format(wdir,opt.net))

        del chkpt

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=250)  # 500200 batches at bs 16, 117263 COCO images = 273 epochs
    parser.add_argument('--batch-size', type=int, default=16)  # effective bs = batch_size * accumulate = 16 * 4 = 64
    parser.add_argument('--net', type=str, default='mbv3_large_1', help='net')
    parser.add_argument('--train_path', type=str, default='/mnt/data1/yanghuiyu/project/object_detect/yolov4/my_data/wider_landmark_yolo_train.txt', help='*.txt path')
    # parser.add_argument('--train_path', type=str, default='./data/wider_landmark98_yolo_train.txt', help='*.txt path')
    parser.add_argument('--multi-scale', action='store_true', help='adjust (67%% - 150%%) img_size every 10 batches')
    parser.add_argument('--img-size', nargs='+', type=int, default=[320, 640], help='[min_train, max-train, test]')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--backbone_weights', type=str, default='./pretrained/mobilenetv3-large-1cd25616.pth', help='initial backbone_weights path')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    opt = parser.parse_args()
    last = wdir + '/{}_last.pt'.format(opt.net)
    opt.weights = last if opt.resume else opt.weights
    check_git_status()

    print(opt)
    opt.img_size.extend([opt.img_size[-1]] * (3 - len(opt.img_size)))  # extend to 3 sizes (min, max, test)
    device = torch_utils.select_device(opt.device, batch_size=opt.batch_size)
    if device.type == 'cpu':
        mixed_precision = False

    # scale hyp['obj'] by img_size (evolved at 320)
    # hyp['obj'] *= opt.img_size[0] / 320.

    tb_writer = None

    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter(comment=opt.name)
    train(hyp)  # train normally


