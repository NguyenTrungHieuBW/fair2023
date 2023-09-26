import random
import numpy as np
import tensorflow as tf
from iterator import Iterator
from model import *
from utils import *
import os
import time
SEED = 3
MAX_EPOCH = 10
TEST_FREQ = 1000
LR = 1e-3
EMBEDDING_DIM = 18  # dimension of embedding vector
HIDDEN_SIZE = 36
ATTENTION_SIZE = 36
MODEL_TYPE = "SLi_Rec_Adaptive"

MODEL_DICT = {"ASVD": Model_ASVD, "DIN": Model_DIN, "LSTM": Model_LSTM, "LSTMPP": Model_LSTMPP, "NARM": Model_NARM, "CARNN": Model_CARNN,  # baselines
              "Time1LSTM": Model_Time1LSTM, "Time2LSTM": Model_Time2LSTM, "Time3LSTM": Model_Time3LSTM, "DIEN": Model_DIEN,
              "A2SVD": Model_A2SVD, "T_SeqRec": Model_T_SeqRec, "TC_SeqRec_I": Model_TC_SeqRec_I, "TC_SeqRec_G": Model_TC_SeqRec_G,  # our models
              "TC_SeqRec": Model_TC_SeqRec, "SLi_Rec_Fixed": Model_SLi_Rec_Fixed, "SLi_Rec_Adaptive": Model_SLi_Rec_Adaptive}

# MODEL_TYPE : "SLi_Rec_Adaptive", seed=3


def train(train_file="data/train_data", test_file="data/test_data", save_path="saved_model/", model_type=MODEL_TYPE, seed=SEED):
    tf.set_random_seed(seed)  # dam bao moi lan random deu giong nhau
    np.random.seed(seed)
    random.seed(seed)
    cur_time = time.time()
    with open('/content/fair2023/sli_rec/time.txt', 'w') as writefile:
        # Write the data to the file
        writefile.write(str(cur_time))
    train_file
    
    with tf.Session() as sess:
        if model_type in MODEL_DICT:
            # set current model = Model_SLi_Rec_Adaptive
            cur_model = MODEL_DICT[model_type]
        else:  # neu model k dc support thi tra ve error
            print("{0} is not implemented".format(model_type))
            return
        train_data, test_data = Iterator(train_file), Iterator(test_file)
        # print(train_file)
        # so luong user, so luong item, so luong category
        user_number, item_number, cate_number = train_data.get_id_numbers()
        model = cur_model(user_number, item_number, cate_number,
                          EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        itr = 0
        learning_rate = LR
        best_auc = 0.0
        best_model_path = save_path + model_type
        train_loss_list, train_accuracy_list = [], []
        test_auc_list, test_loss_list, test_accuracy_list = [], [], []
        for i in range(MAX_EPOCH):
            train_loss_sum = 0.0
            train_accuracy_sum = 0.0
            for source, target in train_data:
                user, targetitem, targetcategory, item_history, cate_history, timeinterval_history, timelast_history, timenow_history, mid_mask, label, seq_len = prepare_data(
                    source, target)
                train_loss, train_acc = model.train(sess, [user, targetitem, targetcategory, item_history, cate_history, timeinterval_history,
                                                           timelast_history, timenow_history, mid_mask, label, seq_len, learning_rate])
                train_loss_sum += train_loss
                train_accuracy_sum += train_acc
                itr += 1
                if (itr % TEST_FREQ) == 0:
                    cur_time = time.time()
                    with open('/content/fair2023/sli_rec/time.txt', 'a') as writefile:
                        # Write the data to the file
                        writefile.write("Iter: {0}, Time: {1}\n".format(
                            itr, str(cur_time)))
                    train_loss_list.append(train_loss_sum / TEST_FREQ)
                    train_accuracy_list.append(train_accuracy_sum / TEST_FREQ)
                    with open('/content/fair2023/sli_rec/information.txt', 'a') as writefile:
                        writefile.write("Iter: {0}, training loss = {1}, training accuracy = {2}\n".format(
                            itr, train_loss_sum / TEST_FREQ, train_accuracy_sum / TEST_FREQ))
                    print("Iter: {0}, training loss = {1}, training accuracy = {2}".format(
                        itr, train_loss_sum / TEST_FREQ, train_accuracy_sum / TEST_FREQ))
                    test_auc, test_loss, test_acc = evaluate_epoch(
                        sess, test_data, model)

                    test_auc_list.append(test_auc)
                    test_loss_list.append(test_loss)
                    test_accuracy_list.append(test_acc)

                    with open('/content/fair2023/sli_rec/result.txt', 'w') as writefile:
                        # Write the data to the file
                        writefile.write(",".join(str(item)
                                        for item in train_loss_list) + "\n")

                        # Write train_accuracy_list to the file
                        writefile.write(",".join(str(item)
                                        for item in train_accuracy_list) + "\n")

                        # Write test_auc_list to the file
                        writefile.write(",".join(str(item)
                                        for item in test_auc_list) + "\n")

                        # Write test_loss_list to the file
                        writefile.write(",".join(str(item)
                                        for item in test_loss_list) + "\n")

                        # Write test_accuracy_list to the file
                        writefile.write(",".join(str(item)
                                        for item in test_accuracy_list) + "\n")

                    with open('/content/fair2023/sli_rec/information.txt', 'a') as writefile:
                        writefile.write("test_auc: {0}, testing loss = {1}, testing accuracy = {2}\n".format(
                            test_auc, test_loss, test_acc))
                    print("test_auc: {0}, testing loss = {1}, testing accuracy = {2}".format(
                        test_auc, test_loss, test_acc))

                    if test_auc > best_auc:
                        best_auc = test_auc
                        model.save(sess, best_model_path)
                        with open('/content/fair2023/sli_rec/information.txt', 'a') as writefile:
                            writefile.write(
                                "Model saved in {0}\n".format(best_model_path))
                        print("Model saved in {0}".format(best_model_path))
                    train_loss_sum = 0.0
                    train_accuracy_sum = 0.0


if __name__ == "__main__":
    train()
