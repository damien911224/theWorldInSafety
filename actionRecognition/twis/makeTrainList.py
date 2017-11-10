import sys
import os
import glob
import random
import cv2
import string
import shutil
import time
from multiprocessing import Pool, Value, Lock, current_process, Manager
from ctypes import c_int, c_float
from pipes import quote


def showProgress(progress_type='', message=' DONE', process_start_time=0.0, process_id=0):
    global len_video_list
    global num_counter
    global time_counter
    global counter_lock
    global global_start_time
    global global_num_worker

    if progress_type == 'Copying':
        with counter_lock:
            num_counter.value += 1
            current_time = time.time()
            elapsed_time_per_process = current_time - process_start_time
            whole_elapsed_time = current_time - global_start_time
            time_counter.value += whole_elapsed_time / float(num_counter.value)
            average_duration = time_counter.value / float(num_counter.value) / 3600.0
            remaining_hours = float(len_video_list - num_counter.value) * average_duration

            print \
            "{0}|{1:05d}th|{2:06.3f}%|Remaining: {3:.2f}Hours|Current: {4:02d} Worker|AvgDuration: {5:.2f}Secs|OneDuration: {6:.2f}Secs\n".format(
                progress_type, num_counter.value,float(num_counter.value)/float(len_video_list)*100.0, remaining_hours,
                process_id%global_num_worker+1, average_duration * 3600.0, elapsed_time_per_process) + \
                " " * (len(progress_type)+16) + "|Message: " + message


def makeTrainList(version='3'):
    v_file_list = glob.glob("/media/damien/DATA2/cvData/TWIS/v{}/"+ "Violence/*.avi".format(version))
    n_file_list = glob.glob("/media/damien/DATA2/cvData/TWIS/v{}/" + "Normal/*.avi".format(version))

    num_of_splits = 3

    for i in range(0, num_of_splits):
        train_list_f = open("../data/twis_splits/trainlist0%d.txt" %(i+1), "w")
        test_list_f = open("../data/twis_splits/testlist0%d.txt" %(i+1), "w")

        v_train_indices = random.sample(range(0, (len(v_file_list)-1)), int(len(v_file_list)*0.80))
        v_test_indices = []

        for i in xrange(len(v_file_list)):
            if i not in v_train_indices:
                v_test_indices.append(i)

        print len(v_train_indices)
        print len(v_test_indices)

        for i in xrange(len(v_train_indices)):
            train_list_f.write(v_file_list[v_train_indices[i]].split('/')[-2] + "/" + v_file_list[v_train_indices[i]].split('/')[-1] + " 1\n")

        for i in xrange(len(v_test_indices)):
            test_list_f.write(v_file_list[v_test_indices[i]].split('/')[-2] + "/" + v_file_list[v_test_indices[i]].split('/')[-1] + "\n")

        ratio_of_normal_to_violence = float(len(v_file_list)) / float(len(n_file_list))

        n_train_indices = random.sample(range(0, (len(n_file_list)-1)), int(len(n_file_list)*0.80*ratio_of_normal_to_violence))
        n_test_indices = []

        n_test_count = 0
        while True:
            i = random.sample(range(0, (len(n_file_list)-1)), 1)
            if i[0] not in n_train_indices:
                n_test_indices.append(i[0])
                n_test_count += 1
                if n_test_count >= len(n_file_list)*0.20*ratio_of_normal_to_violence:
                    break

        print len(n_train_indices)
        print len(n_test_indices)

        for i in xrange(len(n_train_indices)):
            train_list_f.write(n_file_list[n_train_indices[i]].split('/')[-2] + "/" + n_file_list[n_train_indices[i]].split('/')[-1] + " 2\n")

        for i in xrange(len(n_test_indices)):
            test_list_f.write(n_file_list[n_test_indices[i]].split('/')[-2] + "/" + n_file_list[n_test_indices[i]].split('/')[-1] + "\n")

        train_list_f.close()
        test_list_f.close()


def extractOpticalFlow(video_file_path, out_full_path):
    new_size = (340, 256)
    out_format = 'dir'
    df_path = "~/temporal-segment-networks/lib/dense_flow/"
    dev_id = 0

    image_full_path = out_full_path + "/images"
    optical_flow_full_path = out_full_path + "/optical_flow"

    video_cap = cv2.VideoCapture(video_file_path)
    if video_cap.isOpened():
        frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_cap.release()

    if not os.path.exists(out_full_path):
        try:
            os.mkdir(out_full_path)
        except OSError:
            pass

    if not os.path.exists(image_full_path):
        try:
            os.mkdir(image_full_path)
        except OSError:
            pass

    if not os.path.exists(optical_flow_full_path):
        try:
            os.mkdir(optical_flow_full_path)
        except OSError:
            pass

    image_path = '{}/images/img'.format(out_full_path)
    optical_flow_x_path = '{}/optical_flow/flow_x'.format(out_full_path)
    optical_flow_y_path = '{}/optical_flow/flow_y'.format(out_full_path)


    cmd = os.path.join(
        df_path + 'build/extract_gpu') + ' -f {} -x {} -y {} -i {} -b 20 -t 1 -d {} -s 1 -o {} -w {} -h {}'.format(
        quote(video_file_path), quote(optical_flow_x_path), quote(optical_flow_y_path), quote(image_path), dev_id,
        out_format, new_size[0], new_size[1])

    os.system(cmd)
    sys.stdout.flush()


def copyAndRenameFile(src_file_path):
    start_time = time.time()
    identity_length = 13

    current = current_process()
    current_id = current._identity[0] -1

    if not os.path.exists(dst_folder):
        try:
            os.mkdir(dst_folder)
        except OSError:
            pass

    video_cap = cv2.VideoCapture(src_file_path)
    video_frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))-1

    while True:
        identity = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(identity_length))
        if not identity in identity_list:
            break

    src_file_name = src_file_path.split('/')[-1].split('.')[-2]
    dst_file_name = identity + '_{:05d}'.format(video_frame_count)
    dst_category = src_file_path.split('/')[-2]
    dst_ext = src_file_path.split('.')[-1]
    dst_file_path = os.path.join(dst_folder, dst_category, dst_file_name + '.' + dst_ext)

    dst_sub_folder_path = os.path.join(dst_folder, dst_category)
    if not os.path.exists(dst_sub_folder_path):
        try:
            os.mkdir(dst_sub_folder_path)
        except OSError:
            pass

    identity_list.append(identity)

    shutil.copy(src_file_path, dst_file_path)
    video_cap.release()

    frame_src_file_path = os.path.join(frame_src_folder, src_file_name)
    frame_dst_file_path = os.path.join(frame_dst_folder, dst_file_name)

    if not os.path.exists(frame_dst_file_path):
        try:
            os.mkdir(frame_dst_file_path)
        except OSError:
            pass

    image_dst_path = os.path.join(frame_dst_file_path, 'images')
    flow_dst_path = os.path.join(frame_dst_file_path, 'optical_flow')

    if not os.path.exists(image_dst_path):
        try:
            os.mkdir(image_dst_path)
        except OSError:
            pass

    if not os.path.exists(flow_dst_path):
        try:
            os.mkdir(flow_dst_path)
        except OSError:
            pass

    if os.path.exists(frame_src_file_path):
        image_src_list = glob.glob(frame_src_file_path + '/images/*')
        flow_src_list = glob.glob(frame_src_file_path + '/optical_flow/*')

        image_file_counter = len(image_src_list)
        flow_file_counter = len(flow_src_list)/2

        if image_file_counter < video_frame_count or flow_file_counter < video_frame_count:
            extractOpticalFlow(src_file_path, frame_dst_file_path)
        else:
            for image_src_path in image_src_list:
                shutil.copy(image_src_path, os.path.join(image_dst_path, image_src_path.split('/')[-1]))
            for flow_src_path in flow_src_list:
                shutil.copy(flow_src_path, os.path.join(flow_dst_path, flow_src_path.split('/')[-1]))
    else:
        extractOpticalFlow(src_file_path, frame_dst_file_path)

    showProgress('Copying', src_file_path + ' TO '  + dst_file_path, start_time, current_id)


def changeFilesWithIds():
    global global_start_time
    global len_video_list
    global num_counter
    global time_counter
    global counter_lock
    global global_start_time
    global global_num_worker

    global dst_folder
    global frame_src_folder
    global frame_dst_folder
    global identity_list

    global_start_time = time.time()
    num_counter = Value(c_int)
    time_counter = Value(c_float)
    counter_lock = Lock()
    global_num_worker = 11
    manager = Manager()
    identity_list = manager.list()

    src_folder = '/media/damien/DATA/cvData/TWIS/v2'
    dst_folder = '/media/damien/DATA2/cvData/TWIS/v3'
    frame_src_folder = '/media/damien/DATA/cvData/TSN_data/TWIS/v2'
    frame_dst_folder = '/media/damien/DATA2/cvData/TSN_data/TWIS/v3'

    n_file_list = glob.glob(src_folder + '/Normal/*')
    v_file_list = glob.glob(src_folder + '/Violence/*')
    v_file_length = len(v_file_list)
    n_file_sample_length = v_file_length * 3
    n_file_sample_indices = random.sample(range(0, len(n_file_list)), n_file_sample_length)

    n_file_sample_list = []
    for index in n_file_sample_indices:
        n_file_sample_list.append(n_file_list[index])

    src_file_list = v_file_list + n_file_sample_list
    len_video_list = len(src_file_list)

    pool = Pool(processes=global_num_worker)
    pool.map(copyAndRenameFile, src_file_list)
    pool.close()


if __name__ == '__main__':
     makeTrainList()