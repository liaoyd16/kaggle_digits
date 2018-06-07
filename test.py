# test.py

import real_vgg
from real_vgg import VGG
import digits_cnn
from digits_cnn import Classifier

import numpy as np

import torch

import csv

dataset_root = "/Users/liaoyuanda/Desktop/kaggle_digits/dataset/"
model_root = "/Users/liaoyuanda/Desktop/kaggle_digits/model/"

import pickle

def load_test_set(filename):
    test_set = []

    test_hand = open(filename, "r")
    reader = csv.reader(test_hand)
    for entry in reader:
        try:
            pic = []
            for c in entry:
                pic.append(int(c))

            # pic = torch.tensor(pic, dtype=torch.float)i
            test_set.append(pic)
        except Exception as e:
            print(e)
            continue

    return torch.tensor(test_set, dtype=torch.float)

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

def test_model(model):

    # load testset: [np.array([]), ...]
    testset = load_test_set(dataset_root + "test.csv")
    print(testset.size())

    # run valid
    v_labels, v_set, cnt = load_valid(dataset_root + "valid.csv")

    corr = 0
    v_output = model.forward_pass(v_set, cnt).detach().numpy()
    for i in range(cnt):
        guess = np.argmax(v_output[i])
        if guess==v_labels[i]:
            corr += 1
    print("accuracy = {}".format(corr/cnt))

    # get ans
    ans_list = []

    answers = model.forward_pass(testset, testset.size()[0]).detach().numpy()

    for i in range(28000):
        answer = np.argmax( answers[i] )
        ans_list.append( (i+1, answer) )

    return ans_list

if __name__=="__main__":
    best_name = "resnet_best.pickle"
    pkl_hand = open(model_root + best_name, "rb")
    model = pickle.load(pkl_hand)

    print(type(model))

    ans_write_fn = "./ans.csv"
    ans_write_fh = open(ans_write_fn, "w")
    ans_writer = csv.writer(ans_write_fh)

    ans_writer.writerow(["ImageId", "Label"])

    ans_list = test_model(model)
    for ans in ans_list:
        print(ans)
        ans_writer.writerow(ans)
