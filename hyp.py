hyp = {'nc' : 1 , # class number
       'giou': 3.54,  # giou loss gain
       'cls': 37.4  ,  # cls loss gain
       'land' : 0.01,
       'cls_pw': 1.0,  # cls BCELoss positive_weight
       'obj': 64.3 * 5,  # obj loss gain (*=img_size/320 if img_size != 320)
       'obj_pw': 1.0,  # obj BCELoss positive_weight
       'iou_t': 0.2,  # iou training threshold
       'lr0': 0.01,  # initial learning rate (SGD=5E-3, Adam=5E-4)
       'lrf': 0.0005,  # final learning rate (with cos scheduler)
       'momentum': 0.937,  # SGD momentum
       'weight_decay': 0.0005,  # optimizer weight decay
       'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
       'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.36,  # image HSV-Value augmentation (fract  asdkmion)
       'degrees': 15  ,  # image rotation (+/- deg)
       'translate': 0.1 ,  # image translation (+/- fraction)
       'scale': (0.5,3.0) ,  # image scale (+/- gain)
       'shear': 10  ,
       'point_num' : 5 ,
       'out_channels':64,
       'anchors': [[[12,12], [20,20], [32,32] ],
                   [ [48,48],[72,72], [128,128]],
                   [[192,192],[320,320],[480,480]]] , #12,12,  20,20,  32,32,  48,48,  96,96,  128,128,  196,196,  300,300,  430,430
       'flip_idx_pair' : [[0,1],[3,4]],
       # 'flip_idx_pair':[[0, 32],  [1,  31], [2,  30], [3,  29], [4,  28], [5, 27], [6, 26], [7, 25], [8, 24], [9, 23], [10, 22],
       #       [11, 21], [12, 20], [13, 19], [14, 18], [15, 17],  # check
       #       [33, 46], [34, 45], [35, 44], [36, 43], [37, 42], [38, 50], [39, 49], [40, 48], [41, 47],  # elbrow
       #       [60, 72], [61, 71], [62, 70], [63, 69], [64, 68], [65, 75], [66, 74], [67, 73],
       #       [55, 59], [56, 58],
       #       [76, 82], [77, 81], [78, 80], [87, 83], [86, 84],
       #       [88, 92], [89, 91], [95, 93], [96, 97]],

       }




