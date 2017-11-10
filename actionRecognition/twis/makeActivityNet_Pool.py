import json
import os
import glob
import cv2
import re
import time
from pytube import YouTube
from multiprocessing import Pool, Value, Lock, current_process
from ctypes import c_int, c_float

def error_increment():
    with error_counter_lock:
        error_counter.value += 1


def showProgress(progress_type='', file_name='', message=' DONE', process_start_time=0.0, process_id=0):
    global num_workers

    if progress_type == 'Downloading':
        with counter_lock:
            num_counter.value += 1
            current_time = time.time()
            elapsed_time_per_process = current_time - process_start_time
            whole_elapsed_time = current_time - global_start_time
            time_counter.value += whole_elapsed_time / float(num_counter.value)
            average_duration = time_counter.value / float(num_counter.value) / 3600.0
            remaining_hours = float(len_video_list - num_counter.value) * average_duration

            print \
            "ActivityNet_v{0}|{7:}|{5:05d}th|{1:06.3f}%|Remaining: {2:.2f}Hours|Current: {3}th Worker|AvgDuration: {6:.2f}Secs|OneDuration: {4:.2f}Secs\n".format(version,
            float(num_counter.value)/float(len_video_list)*100.0, remaining_hours,
            process_id%num_workers+1, elapsed_time_per_process, num_counter.value, average_duration * 3600.0, progress_type) + \
            "                                            |FileName: " + file_name + message

    elif progress_type == 'Clipping':
        with counter_lock:
            num_counter.value += 1
            current_time = time.time()
            elapsed_time_per_process = current_time - process_start_time
            whole_elapsed_time = current_time - global_start_time
            time_counter.value += whole_elapsed_time / float(num_counter.value)
            average_duration = time_counter.value / float(num_counter.value) / 3600.0
            remaining_hours = float(len_video_list - num_counter.value) * average_duration

            print \
                "ActivityNet_v{0}|{7:}|{5:05d}th|{1:06.3f}%|Remaining: {2:.2f}Hours|Current: {3}th Worker|AvgDuration: {6:.2f}Secs|OneDuration: {4:.2f}Secs\n".format(
                    version,
                    float(num_counter.value) / float(len_video_list) * 100.0, remaining_hours,
                    process_id % 8 + 1, elapsed_time_per_process, num_counter.value, average_duration * 3600.0,
                    progress_type) + \
                "                                            |FileName: " + file_name + message

    elif progress_type == 'ErrorChecking':
        with counter_lock:
            num_counter.value += 1
            current_time = time.time()
            elapsed_time_per_process = current_time - process_start_time
            whole_elapsed_time = current_time - global_start_time
            time_counter.value += whole_elapsed_time / float(num_counter.value)
            average_duration = time_counter.value / float(num_counter.value) / 3600.0
            remaining_hours = float(len_video_list - num_counter.value) * average_duration

            print \
                "ActivityNet_v{0}|{7:}|{5:05d}th|{1:06.3f}%|Remaining: {2:.2f}Hours|Current: {3}th Worker|AvgDuration: {6:.2f}Secs|OneDuration: {4:.2f}Secs\n".format(
                    version,
                    float(num_counter.value) / float(len_video_list) * 100.0, remaining_hours,
                    process_id % 8 + 1, elapsed_time_per_process, num_counter.value, average_duration * 3600.0,
                    progress_type) + \
                "                                            |FileName: " + file_name + message


def appendList(type, video_items):
    identity = video_items[0]
    path = video_items[1]

    if type == 'full':
        with video_path_lock:
            full_video_paths[identity] = path

    elif type == 'clip':
        with video_path_lock:
            clip_video_paths[identity] = path


def parseActivityJsonFile(version='1.2'):
    default_path = "/home/damien/temporal-segment-networks/data/activitynet_splits"
    json_fp = open(os.path.join(default_path, "activity_net.v{version}.min.json".format(version=version)), "r")

    js = json.loads(json_fp.read())

    json_fp.close()

    video_list = []
    identity_list = []
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
            identity = str(url).split('=')[-1]
            segment = js['database'][db]['annotations'][0]['segment']
            duration = js['database'][db]['duration']

            video_list.append({'identity': identity, 'category': category, 'parent_category': parent_category,
                               'url': url, 'segment': segment, 'duration': duration, 'subset': subset})
        else:
            url = js['database'][db]['url']
            identity = str(url).split('=')[-1]

            video_list.append({'identity': identity, 'url': url, 'subset': subset})

        identity_list.append(identity)

    return video_list, identity_list


def downloadFullVideos(video_items):
    start_time = time.time()
    current = current_process()

    save_folder = video_items[0]
    video_file_dic = video_items[1]

    subset = video_file_dic['subset']
    video_url = video_file_dic['url']
    video_identity = video_file_dic['identity']

    try:
        youtube = YouTube(video_url)
    except:
        message = ' !! YOUTUBE ERROR !!'
        showProgress(progress_type='Downloading',file_name=video_url, message=message, process_start_time=start_time, process_id=int(current._identity[0]-1))
        return

    filename = video_identity
    youtube.set_filename(filename)

    video = youtube.get_videos()[-2]
    video_full_path = os.path.join(save_folder, subset, filename + '.' + video.extension)
    video_save_folder = os.path.join(save_folder, subset)
    if not os.path.exists(video_save_folder):
        try:
            os.mkdir(video_save_folder)
        except OSError:
            pass


    if os.path.exists(video_full_path):
        message = ' !! PASS !!'
        showProgress(progress_type='Downloading', file_name=video_full_path, message=message, process_start_time=start_time,
                     process_id=int(current._identity[0] - 1))
        return

    try:
        video.download(video_save_folder)
    except:
        message = ' !! YOUTUBE DOWNLOAD ERROR !!'
        showProgress(progress_type='Downloading', file_name=video_full_path, message=message, process_start_time=start_time,
                     process_id=int(current._identity[0] - 1))
        return

    message = ' !! DOWNLOADING DONE !! '
    showProgress(progress_type='Downloading', file_name=video_full_path, message=message,
                 process_start_time=start_time,
                 process_id=int(current._identity[0] - 1))


def downloadFullVideosWithClipping(video_items):
    start_time = time.time()
    current = current_process()

    version = video_items[0]
    video_file_dic = video_items[1]
    save_path = '/media/damien/DATA/cvData/ActivityNet/v{version}/FullVideos'.format(version=version)

    regex = re.compile('[^a-zA-Z0-9() .,-_&]')
    parent_full_path = save_path + "/" + regex.sub('', video_file_dic['parent_category'].strip().replace('/', '').replace('  ',' '))

    if not os.path.exists(parent_full_path):
        try:
            os.mkdir(parent_full_path)
        except OSError:
            pass

    category_full_path = parent_full_path + "/" + regex.sub('', video_file_dic['category'].strip().replace('/', '').replace('  ',' '))

    if not os.path.exists(category_full_path):
        try:
            os.mkdir(category_full_path)
        except OSError:
            pass

    video_url = video_file_dic['url']

    try:
        youtube = YouTube(video_url)
    except:
        message = ' !! YOUTUBE ERROR !!'
        showProgress(progress_type='Downloading',file_name=category_full_path, message=message, process_start_time=start_time, process_id=int(current._identity[0]-1))
        return

    filename = video_file_dic['identity']
    youtube.set_filename(filename)

    video = youtube.get_videos()[-2]
    video_full_path = os.path.join(category_full_path, filename +'.' + video.extension)

    if not os.path.exists(video_full_path):
        try:
            video.download(category_full_path)
        except:
            message = ' !! YOUTUBE DOWNLOAD ERROR !!'
            showProgress(progress_type='Downloading', file_name=video_full_path, message=message, process_start_time=start_time,
                         process_id=int(current._identity[0] - 1))
            return

    if video_file_dic['identity'] in full_video_paths:
        del full_video_paths[video_file_dic['identity']]
    video_append_items = [video_file_dic['identity'], video_full_path]
    appendList(type='full', video_items=video_append_items)
    message = clipDownloadedVideo(video_file_dic, video_full_path, version)
    showProgress(progress_type='Downloading', file_name=video_full_path, message=message,
                 process_start_time=start_time,
                 process_id=int(current._identity[0] - 1))


def clipDownloadedVideo(video_file_dic, video_full_path, version):
    save_path = '/media/damien/DATA/cvData/ActivityNet/v{version}/ClipVideos'.format(version=version)

    regex = re.compile('[^a-zA-Z0-9() .,-_&]')
    parent_full_path = save_path + "/" + regex.sub('', video_file_dic['parent_category'].strip().replace('/', '').replace('  ',' '))

    if not os.path.exists(parent_full_path):
        try:
            os.mkdir(parent_full_path)
        except OSError:
            pass

    category_full_path = parent_full_path + "/" + regex.sub('', video_file_dic['category'].strip().replace('/', '').replace('  ',' '))

    if not os.path.exists(category_full_path):
        try:
            os.mkdir(category_full_path)
        except OSError:
            pass

    outvideo_path = os.path.join(category_full_path, video_file_dic['identity'] + '.avi')

    if os.path.exists(outvideo_path):
        temp_cap = cv2.VideoCapture(outvideo_path)
        if temp_cap.isOpened():
            video_fps = temp_cap.get(cv2.CAP_PROP_FPS)
            start_frame = int(float(video_file_dic['segment'][0]) * video_fps)
            end_frame = int(float(video_file_dic['segment'][1]) * video_fps)
            duration = end_frame - start_frame + 1
            temp_duration = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if duration > temp_duration:
                try:
                    os.remove(outvideo_path)
                except OSError:
                    pass
            else:
                if not video_file_dic['identity'] in clip_video_paths:
                    video_append_items = [video_file_dic['identity'], outvideo_path]
                    appendList(type='clip', video_items=video_append_items)
                message = ' !! PASS !!'
                showProgress(progress_type='Clipping', file_name=outvideo_path, message=message,
                             process_start_time=start_time,
                             process_id=int(current._identity[0] - 1))
                return

            temp_cap.release()
        else:
            message = ' !! PASS !!'
            showProgress(progress_type='Clipping', file_name=outvideo_path, message=message,
                         process_start_time=start_time,
                         process_id=int(current._identity[0] - 1))
            return

    try:
        video_cap = cv2.VideoCapture(video_full_path)
    except:
        message = ' !! VIDEO CAP ERROR !!'
        showProgress(progress_type='Clipping', file_name=video_full_path, message=message,
                     process_start_time=start_time,
                     process_id=int(current._identity[0] - 1))
        return

    if video_cap.isOpened():
        video_fps = video_cap.get(cv2.CAP_PROP_FPS)
        video_fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        start_frame = int(float(video_file_dic['segment'][0]) * video_fps)
        end_frame = int(float(video_file_dic['segment'][1]) * video_fps)

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
            video_cap.release()
            message = ' !! WRITER ERROR !!'
            return message

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

    if video_file_dic['identity'] in clip_video_paths:
        del clip_video_paths[video_file_dic['identity']]
    video_append_items = [video_file_dic['identity'], outvideo_path]
    appendList(type='clip', video_items=video_append_items)
    message = ' DONE'
    return message


def clipDownloadedVideos_Pool(video_items):
    start_time = time.time()
    current = current_process()

    version = video_items[0]
    video_file_dic = video_items[1]
    save_path = '/media/damien/DATA/cvData/ActivityNet/v{version}/ClipVideos'.format(version=version)

    regex = re.compile('[^a-zA-Z0-9() .,-_&]')
    parent_full_path = save_path + "/" + regex.sub('', video_file_dic['parent_category'].strip().replace('/', '').replace('  ',' '))

    if not os.path.exists(parent_full_path):
        try:
            os.mkdir(parent_full_path)
        except OSError:
            pass

    category_full_path = parent_full_path + "/" + regex.sub('', video_file_dic['category'].strip().replace('/', '').replace('  ',' '))

    if not os.path.exists(category_full_path):
        try:
            os.mkdir(category_full_path)
        except OSError:
            pass

    outvideo_path = os.path.join(category_full_path, video_file_dic['identity'] + '.avi')

    if os.path.exists(outvideo_path):
        temp_cap = cv2.VideoCapture(outvideo_path)
        if temp_cap.isOpened():
            video_fps = temp_cap.get(cv2.CAP_PROP_FPS)
            start_frame = int(float(video_file_dic['segment'][0]) * video_fps)
            end_frame = int(float(video_file_dic['segment'][1]) * video_fps)
            duration = end_frame - start_frame + 1
            temp_duration = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if duration > temp_duration:
                try:
                    os.remove(outvideo_path)
                except OSError:
                    pass
            else:
                if not video_file_dic['identity'] in clip_video_paths:
                    video_append_items = [video_file_dic['identity'], outvideo_path]
                    appendList(type='clip', video_items=video_append_items)
                message = ' !! PASS !!'
                showProgress(progress_type='Clipping', file_name=outvideo_path, message=message,
                             process_start_time=start_time,
                             process_id=int(current._identity[0] - 1))
                return

            temp_cap.release()
        else:
            message = ' !! PASS !!'
            showProgress(progress_type='Clipping', file_name=outvideo_path, message=message,
                         process_start_time=start_time,
                         process_id=int(current._identity[0] - 1))
            return

    if video_file_dic['identity'] in full_video_paths:
        video_full_path = full_video_paths[video_file_dic['identity']]
    else:
        message = ' !! NO FULL VIDEO !!'
        showProgress(progress_type='Clipping', file_name=outvideo_path, message=message,
                     process_start_time=start_time,
                     process_id=int(current._identity[0] - 1))
        return

    try:
        video_cap = cv2.VideoCapture(video_full_path)
    except:
        message = ' !! VIDEO CAP ERROR !!'
        showProgress(progress_type='Clipping', file_name=video_full_path, message=message,
                     process_start_time=start_time,
                     process_id=int(current._identity[0] - 1))
        return

    if video_cap.isOpened():
        video_fps = video_cap.get(cv2.CAP_PROP_FPS)
        video_fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        start_frame = int(float(video_file_dic['segment'][0]) * video_fps)
        end_frame = int(float(video_file_dic['segment'][1]) * video_fps)

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
            video_cap.release()
            message = ' !! WRITER ERROR !!'
            showProgress(progress_type='Clipping', file_name=outvideo_path, message=message,
                         process_start_time=start_time,
                         process_id=int(current._identity[0] - 1))
            return

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

    if video_file_dic['identity'] in clip_video_paths:
        del clip_video_paths[video_file_dic['identity']]
    video_append_items = [video_file_dic['identity'], outvideo_path]
    appendList(type='clip', video_items=video_append_items)
    message = ' DONE'
    showProgress(progress_type='Clipping', file_name=outvideo_path, message=message,
                 process_start_time=start_time,
                 process_id=int(current._identity[0] - 1))


def checkErrors(video_items):
    version = video_items[0]
    video_item = video_items[1]
    video_identity = video_item['identity']
    video_full_path = clip_video_paths[video_identity]

    temp_cap = cv2.VideoCapture()
    if temp_cap.isOpened():
        video_fps = temp_cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(float(video_item['segment'][0]) * video_fps)
        end_frame = int(float(video_item['segment'][1]) * video_fps)
        duration = end_frame - start_frame + 1
        temp_duration = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if duration > temp_duration:
            error_increment()
            message = ' !! ERROR !!'
            showProgress(progress_type='ErrorChecking', file_name=video_full_path, message=message,
                         process_start_time=start_time,
                         process_id=int(current._identity[0] - 1))
            try:
                os.remove(video_full_path)
            except OSError:
                pass
        else:
            message = ' !! PASS !!'
            showProgress(progress_type='ErrorChecking', file_name=video_full_path, message=message,
                         process_start_time=start_time,
                         process_id=int(current._identity[0] - 1))
            return

        temp_cap.release()
    else:
        message = ' !! PASS !!'
        showProgress(progress_type='ErrorChecking', file_name=video_full_path, message=message,
                     process_start_time=start_time,
                     process_id=int(current._identity[0] - 1))
        return


def downloadActivityNetDataSet(version='1.2'):
    global len_video_list
    global num_workers
    global num_counter
    global time_counter
    global counter_lock
    global global_start_time

    num_workers = 16

    global_start_time = time.time()
    video_list, identity_list = parseActivityJsonFile(version=version)

    len_video_list = len(video_list)
    num_counter  = Value(c_int)  # defaults to 0
    time_counter = Value(c_float)
    counter_lock = Lock()

    save_folder = '/media/damien/DATA/cvData/ActivityNet/v{version}'.format(version=version)

    # pool = Pool(num_workers)
    # pool.map(downloadFullVideos, zip([save_folder] * len(video_list), video_list))
    # pool.close()
    # pool.join()

    downloaded_files = glob.glob(os.path.join(save_folder, '*/*'))
    downloaded_identities = []
    for downloaded_file in downloaded_files:
        downloaded_identities.append(downloaded_file.split('/')[-1].split('.')[-2])


    non_downloaded_identities = []
    for identity in identity_list:
        if identity not in downloaded_identities:
            non_downloaded_identities.append(identity)

    with open('activity_v{}'.format(version), 'w') as f:
        for identity in non_downloaded_identities:
            f.write('{}\n'.format(identity))

    print "----------------------------------------------"
    print "ActivityNet {} Download Videos Done".format(version)
    print "Non Downloaded File Count: {:05d}".format(len(non_downloaded_identities))
    print "----------------------------------------------"


def downloadErrors(version='1.2'):
    global len_video_list
    global num_workers
    global num_counter
    global time_counter
    global counter_lock
    global global_start_time

    num_workers = 16

    global_start_time = time.time()
    video_list, identity_list = parseActivityJsonFile(version=version)

    home_folder = os.path.abspath('../../..')
    save_folder = os.path.join(home_folder, 'temp', 'activityNetError', 'v{}'.format(version))

    read_error_identities = []
    with open(os.path.join(save_folder, 'activity_v{}'.format(version)), 'r') as f:
        while True:
            identity = f.readline()[:-1]

            if not identity:
                break
            read_error_identities.append(identity)

    print 'ERROR: {:05d}'.format(len(read_error_identities))

    while True:
        read_error_identities = []
        with open(os.path.join(save_folder, 'activity_v{}'.format(version)), 'r') as f:
            while True:
                identity = f.readline()[:-1]

                if not identity:
                    break
                read_error_identities.append(identity)

        print 'ERROR: {:05d}'.format(len(read_error_identities))

        error_videos = []
        error_identities = []
        for video in video_list:
            video_identity = video['identity']
            if video_identity in read_error_identities:
                error_videos.append(video)
                error_identities.append(video_identity)

        len_video_list = len(error_videos)
        num_counter  = Value(c_int)  # defaults to 0
        time_counter = Value(c_float)
        counter_lock = Lock()

        pool = Pool(num_workers)
        pool.map(downloadFullVideos, zip([save_folder] * len(error_videos), error_videos))
        pool.close()
        pool.join()

        downloaded_files = glob.glob(os.path.join(save_folder, '*/*'))
        downloaded_identities = []
        for downloaded_file in downloaded_files:
            downloaded_identities.append(downloaded_file.split('/')[-1].split('.')[-2])

        non_downloaded_identities = []
        for identity in error_identities:
            if identity not in downloaded_identities:
                non_downloaded_identities.append(identity)

        with open(os.path.join(save_folder, 'activity_v{}'.format(version)), 'w') as f:
            for identity in non_downloaded_identities:
                f.write('{}\n'.format(identity))

        print "----------------------------------------------"
        print "ActivityNet {} Download Videos Done".format(version)
        print "Non Downloaded File Count: {:05d}".format(len(non_downloaded_identities))
        print "----------------------------------------------"

        if len(non_downloaded_identities) == 0:
            break


if __name__ == '__main__':
    version = '1.3'
    downloadErrors(version=version)

    version = '1.2'
    downloadErrors(version=version)