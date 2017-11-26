import cv2
from darkflow.net import build


class SemanticPostProcessor:
    def __init__(self):
        self.build_net()

    def build_net(self):
        print "Semantic Post Process is started ! "
        options = {"pbLoad": "own/my-yolo.pb", "metaLoad": "own/my-yolo.meta", "gpu": 0.6}
        self.tfnet = build.TFNet(options)

    def semantic_post_process(self, clip):
        adult = False
        child = False
        flag = False
        frame_semantics = []
        for j in range(0, len(clip['frames']), 1):
            boxes = []
            img = cv2.imread(clip['frames'][j])
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


# if __name__ == '__main__':
#     clip = dict()
#     temp_save_folder = '/home/najm/theWorldInSafety/semanticPostProcessing/temp'
#     video_file = '/home/najm/theWorldInSafety/semanticPostProcessing/testVideo.mp4'
#     video_cap = cv2.VideoCapture(video_file)
#     index = 1
#     frames = []
#     while True:
#         ok, frame = video_cap.read()
#         if not ok:
#             break
#         if index > len(frame):
#             break
#         frame_path = os.path.join(temp_save_folder, 'img_{:07}.jpg'.format(index))
#         index += 1
#         # cv2.imwrite(frame_path, frame)
#         frames.append(frame_path)
#     clip['frames'] = frames
#     semanticPostProcessor = SemanticPostProcessor()
#     ok, bounding_boxes = semanticPostProcessor.semantic_post_process(clip)
#     print ok, bounding_boxes