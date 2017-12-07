from darkflow.net.build import TFNet
import cv2
import glob
import os

options = {"pbLoad": "own/my-yolo.pb", "metaLoad": "own/my-yolo.meta", "gpu": 0.6}
tfnet = TFNet(options)


# TEST FOR IMAGES
def test_images():
    image_folder = 'sample_img'
    image_paths = glob.glob(os.path.join(image_folder, '*'))
    fail = 1

    for path in image_paths:
        imgcv = cv2.imread(path)
        result = tfnet.return_predict(imgcv)
        height, width, a = imgcv.shape
        thick = int((height + width) // 300)

        if len(result) is 0:
            print "DETECTION FAIL:", fail, "/", len(image_paths)
            fail += 1

        for j in range(0, len(result), 1):
            if result[j]['confidence'] > 0.3:
                tl_x = result[j]['topleft']['x']
                tl_y = result[j]['topleft']['y']
                br_x = result[j]['bottomright']['x']
                br_y = result[j]['bottomright']['y']
                confidence = result[j]['confidence']
                label = result[j]['label']
                if result[j]['label'] == 'Adult':
                    box_colors = (189, 166, 36)
                else:
                    box_colors = (128, 65, 217)
                cv2.rectangle(imgcv, (tl_x, tl_y), (br_x, br_y), box_colors, thick)
                cv2.putText(imgcv, ("{0}".format(label)), (tl_x, tl_y - 12), 2, 1.0,
                            box_colors,
                            2)
                path = path.split('/')[-1]
                cv2.imwrite(path, imgcv)

                #cv2.imshow("a", imgcv)
                #cv2.waitKey(0)
        print "COMPLETED !!!"


# TEST FOR VIDEOS
def test_videos():
    video = "test2.mp4"
    cap = cv2.VideoCapture(video)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) + 1
    size = (width, height)
    thick = int((height + width) // 300)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter("output.avi", fourcc, fps, size)
    while cap.isOpened():

        ret, frame = cap.read()
        if ret is not True:
            break
        result = tfnet.return_predict(frame)
        for i in range(0, len(result), 1):
            if result[i]['confidence'] > 0.000:
                tl_x = result[i]['topleft']['x']
                tl_y = result[i]['topleft']['y']
                br_x = result[i]['bottomright']['x']
                br_y = result[i]['bottomright']['y']
                confidence = result[i]['confidence']
                label = result[i]['label']
                if label == 'Adult':
                    box_colors = (189, 166, 36)
                else:
                    box_colors = (128, 65, 217)
                # print label, confidence
                cv2.rectangle(frame, (tl_x, tl_y), (br_x, br_y), box_colors, thick)
                cv2.putText(frame, ("{0} {1:.2f}".format(label, confidence)), (tl_x, tl_y - 12), 2, 1.0, box_colors,
                            2)
        out.write(frame)
        # cv2.imshow('frame', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

    cap.release()
    out.release()
    # cv2.destroyAllWindows()
    print "COMPLETED !!"

if __name__ == "__main__":
    test_images()
    # test_videos()
