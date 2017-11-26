import cv2
import os
from darkflow.net import build


class SemanticPostProcessor:

    def __init__(self):
        self.build_net()


    def build_net(self):
        print "Semantic Post Process is started ! "
        root_folder =  os.path.abspath('../../semanticPostProcessing')
        options = {"pbLoad": os.path.join(root_folder, "own/my-yolo.pb"),
                   "metaLoad": os.path.join(root_folder, "own/my-yolo.meta"),
                   "threshold": 0.1, "gpu": 0.9}
        self.tfnet = build.TFNet(options)


    def semantic_post_process(self, clip):
        adult = False
        child = False
        flag = False
        frame_semantics = []
        for frame in clip['frames']:
            boxes = []
            img = cv2.imread(frame['image'])
            if img is None:
                continue
            result = self.tfnet.return_predict(img)
            for i in range(0, len(result), 1):
                bounding_box = dict()
                bounding_box['label'] = result[i]['label']
                bounding_box['confidence'] = result[i]['confidence']
                bounding_box['topleft_x'] = result[i]['topleft']['x']
                bounding_box['topleft_y'] = result[i]['topleft']['y']
                bounding_box['bottomright_x'] = result[i]['bottomright']['x']
                bounding_box['bottomright_y'] = result[i]['bottomright']['y']
                boxes.append(bounding_box)
                if result[i]['label'] == 'Adult':
                    adult = True
                if result[i]['label'] == 'Child':
                    child = True
                if adult and child:
                    flag = True

            frame_semantics.append(boxes)
        return flag, frame_semantics