from lxml import etree
from darkflow.net.build import TFNet
import cv2
import os
import glob

#options = {"model": "cfg/tiny-yolo-voc.cfg", "load": "weights/tiny-yolo-voc.weights", "gpu": 0.7}
options = {"pbLoad": "own/my-yolo.pb", "metaLoad": "own/my-yolo.meta", "gpu": 0.6}

tfnet = TFNet(options)

image_folder = 'sample_img'
image_paths = glob.glob(os.path.join(image_folder, '*'))

for path in image_paths:
    img = cv2.imread(path)
    result = tfnet.return_predict(img)
    filename = path.split('/')[-1]
    txt_name = os.path.splitext(filename)[0] + '.xml'
    root =etree.Element("annotation")
    x_folder = etree.Element("folder")
    x_folder.text = image_folder
    x_filename = etree.Element("filename")
    x_filename.text = filename
    x_source = etree.Element("source")
    x_database = etree.Element("database")
    x_database.text = "Unknown"
    x_source.append(x_database)
    etree.SubElement(x_source, "annotation").text = "None"
    etree.SubElement(x_source, "image").text = "None"
    etree.SubElement(x_source, "flickrid").text = "None"
    x_owner = etree.Element("owner")
    etree.SubElement(x_owner, "flickrid").text = "None"
    etree.SubElement(x_owner, "name").text = "None"
    x_size = etree.Element("size")

    imgHeight, imgWidth, imgChannels = cv2.imread(path).shape

    etree.SubElement(x_size, "width").text = str(imgWidth)
    etree.SubElement(x_size, "height").text = str(imgHeight)
    etree.SubElement(x_size, "depth").text = str(imgChannels)

    x_segmented = etree.Element("segmented")
    x_segmented.text = str(0)

    root.append(x_folder)
    root.append(x_filename)
    root.append(x_source)
    root.append(x_owner)
    root.append(x_size)
    root.append(x_segmented)

    f = open('annotest/'+txt_name, 'a+')
    for i in range(0, len(result), 1):
        print result[i]['label']
        if (result[i]['label'] == 'Adult' or result[i]['label'] == 'Child') and result[i]['confidence'] > 0.000000:
            x_object = etree.Element("object")
            if txt_name[0:3] == 'adu':
                etree.SubElement(x_object, "name").text = "Adult"
            else:
                etree.SubElement(x_object, "name").text = "Child"
            etree.SubElement(x_object, "pose").text = "Unspecified"
            etree.SubElement(x_object, "truncated").text = str(0)
            etree.SubElement(x_object, "difficult").text = str(0)
            x_bndbox = etree.Element("bndbox")
            etree.SubElement(x_bndbox, "xmin").text = str(result[i]['topleft']['x'])
            etree.SubElement(x_bndbox, "ymin").text = str(result[i]['topleft']['y'])
            etree.SubElement(x_bndbox, "xmax").text = str(result[i]['bottomright']['x'])
            etree.SubElement(x_bndbox, "ymax").text = str(result[i]['bottomright']['y'])
            x_object.append(x_bndbox)
            root.append(x_object)
    # x_output = etree.tostring(root, pretty_print=True, encoding='UTF-8')
    # print x_output
    x_output = etree.tostring(root, pretty_print=True, encoding='UTF-8')
    f.write(x_output.decode('utf-8'))
    f.close()

print "COMPLETED !!"