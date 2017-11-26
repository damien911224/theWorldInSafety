from darkflow.net.build import TFNet
import cv2
import os
import glob

# path = os.path.join('/home', '/najm', '/darkflow')

# options = {"model": "cfg/tiny-yolo-voc.cfg", "load": "tiny-yolo-voc.weights", "gpu": 0.7}
# tfnet = TFNet(options)



# *********** Change image name *************

image_folder = 'imagespath'
files = os.listdir(image_folder)
j= 1
i = 1
for file in files:
    if file[0:3] == 'adu':
        os.rename(os.path.join(image_folder, file), os.path.join(image_folder, 'adults' + str(i) + '.jpg'))
        i = i + 1

    else:
        os.rename(os.path.join(image_folder, file), os.path.join(image_folder, 'kids' + str(j) + '.jpg'))
        j = j + 1
# *********************************************

# image_folder = '/home/najm/darkflow/plz/img'
# image_paths = glob.glob(os.path.join(image_folder, '*'))
# for path in image_paths:
#     img = cv2.imread(path)
#     result = tfnet.return_predict(img)
#
#     txt_name = os.path.splitext(path)[0]+'.txt'
#     txt_name = path.split('/')[-1]
#     txt_name = '/home/najm/darkflow/plz/img/' + txt_name.split('.')[0] + '.txt'
#     #print txt_name
#
#     f = open(txt_name, 'a+')
#
#     for i in range(0, len(result), 1):
#         #print result
#         # print(result[i]['topleft']['x'])
#         centerX = (result[i]['topleft']['x'] + result[i]['bottomright']['x']) / 2
#         centerY = (result[i]['topleft']['y'] + result[i]['bottomright']['y']) / 2
#         width = abs(result[i]['topleft']['x'] - result[i]['bottomright']['x'])
#         height = abs(result[i]['topleft']['y'] - result[i]['bottomright']['y'])
#         print "HELLOW~~"
#         f.write("{} {} {} {} {}\n".format(0, centerX, centerY, width, height))
#         # 0 : adults  // 1: kids
#     f.close()

# ************* make training_list **********
# image_folder = '/images/kids'
# image_paths = glob.glob(os.path.join(image_folder, '*'))
# for path in image_paths:
#     img = cv2.imread(path)
#     result = tfnet.return_predict(img)
#     txt_name = os.path.splitext(path)[0]+'.jpg'
#     f = open('/home/najm/darkflow/MyTraining/training_list.txt', 'a+')
#     f.write("{}\n".format(txt_name))



# ********** Remove all txt files ************
# image_folder = '/images/kids'
# image_paths = glob.glob(os.path.join(image_folder, '*.txt'))
# for path in image_paths:
#     os.remove(path)
# *********************************************