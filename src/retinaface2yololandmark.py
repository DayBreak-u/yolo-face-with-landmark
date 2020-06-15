import os
import cv2

txt_path = "/home/yanghuiyu/datas/retinaface/train/label.txt"
save_path = "/home/yanghuiyu/datas/yololandmark_wider_train/"
os.makedirs(save_path)


f2 = open(txt_path, "r")
# lines = f.readlines() + f2.readlines()
lines = f2.readlines()
isFirst = True
labels = []
words = []
imgs_path = []
for line in lines:
    line = line.rstrip()
    if line.startswith('#'):
        if isFirst is True:
            isFirst = False
        else:
            labels_copy = labels.copy()
            words.append(labels_copy)
            labels.clear()
        path = line[1:].strip()
        # path = txt_path.replace('label.txt','images/') + path
        path = os.path.join(txt_path.replace(txt_path.split("/")[-1], "images"), path)

        imgs_path.append(path)
    else:
        line = line.split(' ')
        label = [float(x) for x in line]
        labels.append(label)

words.append(labels)
image_count = 0
for path,word in zip(imgs_path,words):
    img = cv2.imread(path)
    img_height = img.shape[0]
    img_width = img.shape[1]
    rel_bbox_list = []

    for anno in word:

        landmark =  []
        for zu in [[4,5],[7,8],[10,11],[13,14],[16,17]]:
            if anno[zu[0]] == -1:
                landmark.append("-1")
                landmark.append("-1")
            else:
                landmark.append(str(float(anno[zu[0]] * 1.0  / img_width)))
                landmark.append(str(float(anno[zu[1]] * 1.0  / img_height)))

        # for i in range(5):
        #     x1,y1,w,h = landmark[2*i],landmark[2*i+1],0.001,0.001
        #     if x1 == -1 or y1 == -1 :
        #         continue
        #
        #     if x1 <= 0 or x1 >= img_width:
        #         continue
        #     if y1 <= 0 or y1 >= img_height:
        #         continue
        #
        #
        #     rel_cx = str(float(x1 * 1.0  / img_width))
        #     rel_cy = str(float(y1 * 1.0  / img_height))
        #     rel_w = str(float(w / img_width))
        #     rel_h = str(float(h / img_height))
        #
        #
        #     string_bbox = str(i+1) +  " " + rel_cx + " " + rel_cy + " " + rel_w + " " + rel_h
        #     rel_bbox_list.append(string_bbox)


        x1, y1, w, h = anno[:4]
        if w < 10 or h < 10:
            img[int(y1):int(y1+h),int(x1):int(x1+w),:] = 127
            continue
        # cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)
        rel_cx = str(float((x1 + int(w/2)) / img_width))
        rel_cy = str(float((y1 + int(h/2)) / img_height))
        rel_w = str(float(w / img_width))
        rel_h = str(float(h / img_height))

        string_bbox = "0 " + rel_cx + " " + rel_cy + " " + rel_w + " " + rel_h + " " + " ".join(landmark)
        rel_bbox_list.append(string_bbox)
    image_count += 1
    print(image_count)

    save_image_name = "wider_" + str(image_count)

    cv2.imwrite( save_path + save_image_name + ".jpg", img)
    with open(save_path + save_image_name + ".txt", "w") as f:
        for i in rel_bbox_list:
            f.write(i + "\n")