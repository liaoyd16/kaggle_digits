
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import csv
import copy
import pickle
import logger
from logger import Logger

# training meta
total_rnd = 100000
batchsize = 10


''' training prep: loading functions '''
def load_train(trainname, batchsize=1):
    reader = csv.reader(open(trainname, "r"))

    trainset  = []
    mini_label = []
    mini_batch = None
    cnt = 0
    batch_cnt = 0
    for row in reader:

        #label
        try:
            label = int(row[0])
        except:
            continue
        
        # pic
        pic = []
        for i in range(1,len(row)):
            pic.append(int(row[i]))
        pic = torch.tensor(pic, dtype=torch.float).view(1,-1)
        
        # to minibatch
        mini_label.append(label)
        if mini_batch is None:
            mini_batch = pic
        else:
            mini_batch = torch.cat((mini_batch, pic), dim=0)

        batch_cnt += 1

        # train set
        if batch_cnt == batchsize:
            mini_label = torch.tensor(mini_label)
            trainset.append([mini_label, mini_batch])
            batch_cnt = 0
            cnt += 1

            mini_label = []
            mini_batch = None

    return trainset, cnt

def load_valid(valid_name):
    reader = csv.reader(open(valid_name, "r"))

    labels = []
    validset = None
    cnt = 0
    for row in reader:
        cnt += 1

        labels.append(int(row[0]))

        pic = []
        for c in row[1:]:
            pic.append(int(c))
        pic = torch.tensor(pic, dtype=torch.float).view(1,-1)
        if validset is None:
            validset = pic
        else:
            validset = torch.cat((validset, pic), dim = 0)
        # print(validset)

    labels = torch.tensor(labels)

    return labels, validset, cnt

def load_test(testname):
    reader = csv.reader(open(testname, "r"))
    
    testset = []
    cnt = 0
    for row in reader:
        try:
            cnt += 1

            pic = []
            for c in row:
                pic.append(int(c))
            pic = torch.tensor(pic, dtype='Float')
            testset.append(pic)
        except:
            continue

    return testset, cnt

def rand(maxrand):
    return random.randint(0, maxrand)

def get_batch_by_index(trainset, index):
    # target = trainset[index][0]
    # x = trainset[index][1]
    one_batch = trainset[index]
    labels = one_batch[:][0]
    pics   = one_batch[:][1]
    for i in range(batchsize):
        pics[i] += noise(28*28)
        
    return labels, pics

def noise(size):
    return F.relu(torch.tensor(np.random.randn(size), dtype=torch.float)) * 0.001

def training(classifier, lossF, optimizer, bestname):
    ''' loading '''
    # load data
    dataset_root = "/Users/liaoyuanda/Desktop/kaggle_digits/dataset/"
    trainset, trainsize = load_train(dataset_root + "training.csv", batchsize=batchsize)
    validlabels, validset, validsize = load_valid(dataset_root + "valid.csv")

    # logger
    # logging_root = "/Users/liaoyuanda/Desktop/kaggle_digits/logging"
    # lg = Logger(logging_root)

    # pickle file
    pkl_hand = open(bestname + ".pickle", "wb")
    ''' loading '''

    '''' misc '''
    train_template = "train:#{}: loss={}, accuracy={}"
    valid_template = "valid: average-loss={}, accuracy={}"

    valid_check_wait = 1000 #batches
    train_corr = 0
    total_seen = 0

    # turns_out = 30
    highest_accu = 0.

    best_model = None

    train_accu = 0.
    '''' misc '''

    train_j = 0
    for i in range(total_rnd):

        # index = (total_rnd + rand(trainsize)) % trainsize
        targets, xs = get_batch_by_index(trainset, i%trainsize)

        # forward
        ys = classifier(xs).view(batchsize, 10)
        ys_np = ys.detach().numpy()
        for j in range(batchsize):
            if np.argmax(ys_np[j])==targets[j]:
                train_corr += 1
            total_seen += 1

        # loss
        loss = lossF(ys, targets)
        # if i % 1000==0:
        #     for j in range(5):
        #         print("ys[{}] = {}, targets[{}] = {}".format(j, ys_np[j][targets[j]], j, targets[j]))

        train_j += 1
        if train_j == 1000:
            print("train accu for this random-round = {}".format(train_corr/total_seen))
            train_j = 0
            train_corr = 0
            total_seen = 0

        # bp
        loss.backward()

        # update
        optimizer.step()
        optimizer.zero_grad()

        # valid check
        if valid_check_wait == 1:
            corr = 0
            avg_loss = 0

            v_targets = validlabels
            v_xs      = validset
            v_ys      = classifier.forward_pass(v_xs, validsize)

            v_ys = v_ys.view(-1, 10)

            # targets_vecs = torch.tensor(np.zeros((validsize, 10)), dtype=torch.float)
            # for j in range(validsize):
                # targets_vecs[j][v_targets[j]] = 1.

            # loss, images, accuracy
            avg_loss = lossF(v_ys, v_targets) / validsize

            v_ys = v_ys.detach().numpy()
            avg_loss = avg_loss.detach().numpy()
            
            # lg.scalar_summary("loss", -np.log10(avg_loss), i)

            # images = []
            err_cnt = 0
            for j in range(validsize):
                hit = np.argmax(v_ys[j])
                if hit == v_targets[j]: 
                    corr += 1

            # lg.scalar_summary("error rate", 1 - (corr / validsize), i)

            accu = corr / validsize
            if accu > highest_accu:
                # turns_out = 30
                print("new record: {} in round {}".format(accu, i))
                best_model = copy.copy(classifier)
                highest_accu = accu
            # else:
                # turns_out -= 1
                # if turns_out==0:
                    # break   # quit validating

            # for debug:
            print( valid_template.format( -np.log10(avg_loss), (corr / validsize) ) )

            valid_check_wait = 1000
        else:
            valid_check_wait -= 1

    pickle.dump(best_model, pkl_hand)
