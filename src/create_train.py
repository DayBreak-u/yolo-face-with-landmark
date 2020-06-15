import  glob
import random
root = "/home/yanghuiyu/datas/yololandmark98_wider_train/*.jpg"
# root = "/home/yanghuiyu/datas/coco/coco/labels/train2017/*.jpg"

imgs = glob.glob(root)
random.shuffle(imgs)
with open("../data/wider_landmark98_yolo_train.txt","w") as fr:
    for im in imgs:
        fr.write("{}\n".format(im))