import cv2
import os
from darkflow.net import build
from multiprocessing import Pool, Manager
import copy_reg
import types
import sys


class SemanticPostProcessor:

    def __init__(self):
        self.build_net()
        copy_reg.pickle(types.MethodType, self._pickle_method)


    def build_net(self):
        global tfnet
        print "Semantic Post Process is started ! "
        root_folder = os.path.abspath('../../semanticPostProcessing')
        #root_folder = os.path.abspath('../../theWorldInSafety/semanticPostProcessing')
        options = {"pbLoad": os.path.join(root_folder, "own/my-yolo.pb"),
                   "metaLoad": os.path.join(root_folder, "own/my-yolo.meta"),
                   "threshold": 0.1, "gpu": 0.8}
        tfnet = build.TFNet(options)


    def _pickle_method(self, m):
        if m.im_self is None:
            return getattr, (m.im_class, m.im_func.func_name)
        else:
            return getattr, (m.im_self, m.im_func.func_name)


    def semantic_post_multi_process(self, clip, semantic_step):
        global semantic_flag

        semantic_flag = False

        pool = Pool(processes=4)

        frame_paths = []
        for frame in clip['frames']:
            frame_paths.append(frame['image'])

        clip_list = range(0, len(frame_paths), semantic_step)

        sampled_frames = []
        for index in clip_list:
            sampled_frames.append(frame_paths[index])

        manager = Manager()
        container = manager.list()
        for _ in range(len(frame_paths)):
            container.append([])

        pool.map(self.single_frame_multi_processing,
                 zip(sampled_frames,
                     clip_list,
                     [container] * len(clip_list)
                     )
                 )

        return semantic_flag, container


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
            result = tfnet.return_predict(img)
            for i in range(0, len(result), 1):
                if result[i]['confidence'] > 0.4:
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


    def single_frame_semantics(self, frame):
        boxes = []
        img = frame
        if img is None:
            return boxes
        result = tfnet.return_predict(img)
        for i in range(0, len(result), 1):
            if result[i]['confidence'] > 0.4:
                bounding_box = dict()
                bounding_box['label'] = result[i]['label']
                bounding_box['confidence'] = result[i]['confidence']
                bounding_box['topleft_x'] = result[i]['topleft']['x']
                bounding_box['topleft_y'] = result[i]['topleft']['y']
                bounding_box['bottomright_x'] = result[i]['bottomright']['x']
                bounding_box['bottomright_y'] = result[i]['bottomright']['y']
                boxes.append(bounding_box)

        return boxes


    def single_frame_multi_processing(self, items):
        global semantic_flag

        frame_path = items[0]
        index = items[1]
        container = items[2]

        adult = False
        child = False

        boxes = []
        img = cv2.imread(frame_path)
        print img.shape
        if img is None:
            return boxes
        result = tfnet.return_predict(img)
        for i in range(0, len(result), 1):
            if result[i]['confidence'] > 0.3:
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
                    semantic_flag = True

        container[index] = boxes



if __name__ == "__main__":
    print "HELLOW~"
    clip = dict()
    temp_save_folder = '/home/najm/theWorldInSafety/semanticPostProcessing/asd'
    video_file = '/home/najm/theWorldInSafety/semanticPostProcessing/test2.mp4'
    video_cap = cv2.VideoCapture(video_file)
    index = 1
    frames = []
    while True:
        ok, frame = video_cap.read()
        if not ok:
            break
        frame_path = os.path.join(temp_save_folder, 'img_{:07}.jpg'.format(index))
        #print frame_path
        index += 1
        if index >200:
            break
        #cv2.imwrite(frame_path, frame)
        frame_dict = dict()
        frame_dict['image'] = frame_path
        frames.append(frame_dict)
    clip['frames'] = frames
    #print clip
    semantic_step=2

    semanticPostProcessor = SemanticPostProcessor()
    #ok, bounding_boxes = semanticPostProcessor.semantic_post_process(clip)
    semanticPostProcessor.multi_process(clip, semantic_step)
