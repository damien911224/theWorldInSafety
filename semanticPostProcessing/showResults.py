from darkflow.net.build import TFNet
import cv2

options = {"pbLoad": "own/my-yolo.pb", "metaLoad": "own/my-yolo.meta", "gpu":0.6}

tfnet = TFNet(options)

imgcv = cv2.imread("./sample_img/t2.jpg")
result = tfnet.return_predict(imgcv)
print(result)