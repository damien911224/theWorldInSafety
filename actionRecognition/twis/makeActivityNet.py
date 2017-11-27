import json
import os
from pytube import YouTube
import glob
import cv2
import re

def parseActivityJsonFile(version='1.2'):
    default_path = "/home/damien/temporal-segment-networks/data/activitynet_splits"
    json_fp = open(os.path.join(default_path, "activity_net.v{version}.min.json".format(version=version)), "r")

    js = json.loads(json_fp.read())

    json_fp.close()

    video_list = []
    for db in js['database']:
        subset = js['database'][db]['subset']
        if subset != 'testing':
            category = js['database'][db]['annotations'][0]['label']
            found_pi = 0
            for pi in xrange(len(js['taxonomy'])):
                if js['taxonomy'][pi]['nodeName'] == category:
                    found_pi = pi
                    break
            parent_category = js['taxonomy'][found_pi]['parentName']
            url = js['database'][db]['url']
            segment = js['database'][db]['annotations'][0]['segment']
            duration = js['database'][db]['duration']

            video_list.append({'category': category, 'parent_category': parent_category,
                               'url': url, 'segment': segment, 'duration': duration})

    return video_list

def downloadActivityNetVideos(version='1.2'):
    save_path = '/media/damien/DATA/cvData/ActivityNet/{version}'.format(version=version)

    video_list = parseActivityJsonFile(version)
    error_list = []
    error_list_indices = []

    for ii in xrange(len(video_list)):
        parent_full_path = save_path + "/" + video_list[ii]['parent_category']

        if not os.path.exists(parent_full_path):
            try:
                os.mkdir(parent_full_path)
            except OSError:
                pass

        category_full_path = parent_full_path + "/" + video_list[ii]['category']

        if not os.path.exists(category_full_path):
            try:
                os.mkdir(category_full_path)
            except OSError:
                pass

        video_url = video_list[ii]['url']
        try:
            youtube = YouTube(video_url)
        except:
            error_list.append(video_list[ii])
            error_list_indices.append(ii)
            print "ActivityNet V{0}: {1:.2f}% File ".format(version, float(ii) / float(
                len(video_list)) * 100) + category_full_path + ' !! YOUTUBE ERROR !!'
            continue

        filename = youtube.filename
        filename = re.sub('[^a-zA-Z0-9]','', filename)

        youtube.set_filename(filename)

        temp_path = "/home/damien/temp/"
        video = youtube.videos[-2]
        video_full_path = temp_path + filename + '.' + video.extension
        outvideo_path = os.path.join(category_full_path, filename + ".avi")

        if os.path.exists(outvideo_path):
            print "ActivityNet V{0}: {1:.2f}% File ".format(version, float(ii) / float(
                len(video_list)) * 100) + outvideo_path + ' !! PASS !!'
            continue

        if os.path.exists(video_full_path):
            try:
                os.remove(video_full_path)
            except OSError:
                pass

        try:
            video.download(temp_path)
        except:
            error_list.append(video_list[ii])
            error_list_indices.append(ii)
            print "ActivityNet V{0}: {1:.2f}% File ".format(version, float(ii) / float(
                len(video_list)) * 100) + video_full_path + ' !! YOUTUBE DOWNLOAD ERROR !!'
            continue

        if youtube:
            try:
                video_cap = cv2.VideoCapture(video_full_path)
            except:
                error_list.append(video_list[ii])
                error_list_indices.append(ii)
                print "ActivityNet V{0}: {1:.2f}% File ".format(version, float(ii) / float(
                    len(video_list)) * 100) + video_full_path + ' !! VIDEO CAP ERROR !!'
                continue

            if video_cap.isOpened():
                video_fps = video_cap.get(cv2.CAP_PROP_FPS)
                video_frame_count = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
                video_fourcc = cv2.VideoWriter_fourcc(*'DIVX')
                video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                start_frame = int(float(video_list[ii]['segment'][0]) * video_fps)
                end_frame = int(float(video_list[ii]['segment'][1]) * video_fps)

                if os.path.exists(outvideo_path):
                    try:
                        os.remove(outvideo_path)
                    except OSError:
                        pass

                video_writer = cv2.VideoWriter()
                try:
                    video_writer.open(outvideo_path,
                                      video_fourcc, video_fps,
                                      (video_width, video_height))
                except:
                    error_list.append(video_list[ii])
                    error_list_indices.append(ii)
                    print "ActivityNet V{0}: {1:.2f}% File ".format(version, float(ii) / float(
                        len(video_list)) * 100) + video_full_path + ' !! WRITER ERROR !!'
                    continue

                video_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                frame_count = start_frame
                while True:
                    ret, frame = video_cap.read()
                    if not ret:
                        break
                    video_writer.write(frame)
                    if frame_count == end_frame:
                        break
                    frame_count += 1
                video_writer.release()

            video_cap.release()
            try:
                os.remove(video_full_path)
            except OSError:
                pass
        else:
            error_list.append(video_list[ii])
            error_list_indices.append(ii)
            print "ActivityNet V{0}: {1:.2f}% File ".format(version, float(ii) / float(
                len(video_list)) * 100) + video_full_path + ' !! OTHER ERROR !!'
            continue
        print "ActivityNet V{0}: {1:.2f}% File ".format(version, float(ii)/float(len(video_list))*100) + video_full_path + ' done'

    print ""
    print "ERROR LIST for ActivityNet v{}".format(version)
    print ""

    for ii in xrange(len(error_list)):
        print "No{0:04d} category: {1}".format(ii+1, error_list[ii]['category'])

    print ""
    print "ActivityNet {} Download Videos Done".format(version)
    print ""

    return error_list_indices

error_list_indices_1 = downloadActivityNetVideos(version='1.2')
error_list_indices_2 = downloadActivityNetVideos(version='1.3')

error_indices_1_fp = open("/home/damien/temp/error_indices_1.txt", "w")
for x in error_list_indices_1:
    error_indices_1_fp.write("%d\n" %x)

error_indices_2_fp = open("/home/damien/temp/error_indices_2.txt", "w")
for x in error_list_indices_2:
    error_indices_2_fp.write("%d\n" %x)

error_indices_1_fp.close()
error_indices_2_fp.close()

