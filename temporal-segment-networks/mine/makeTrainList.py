import sys
import os
import glob
import random

v_file_list = glob.glob("/media/damien/DATA/cvData/TWIS/"+ "Violence/*.avi")
n_file_list = glob.glob("/media/damien/DATA/cvData/TWIS/" + "Normal/*.avi")

num_of_splits = 3

for i in range(0, num_of_splits):
    train_list_f = open("/home/damien/temporal-segment-networks/data/twis_splits/trainlist0%d.txt" %(i+1), "w")
    test_list_f = open("/home/damien/temporal-segment-networks/data/twis_splits/testlist0%d.txt" %(i+1), "w")

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