import os
os.environ['DGLBACKEND'] = 'tensorflow'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
import numpy as np
import random
from tqdm import tqdm
import dgl
from sklearn.model_selection import StratifiedKFold
from dgl.nn.tensorflow import *
import pandas as pd
from sklearn import metrics
from gat_data_change.open_code.graphs_construction import get_samples, create_graph_generator_train_all_path,load_graphs_construction
from gat_data_change.open_code.MGATBCmodel import MGATBC,MGATBC_re


ex_num = 10
ex_fold_num = 5
in_num = 5
in_fold_num = 5
classifiers = in_num * in_fold_num
learning_rate = 1e-3
batch_size = 20
epochs = 200
ROIs = 120
new_node_feature_num = 60
test_num = 25

graph1, graph2, graph3, graph4,label_all = load_graphs_construction()


all_mean_result_ex = np.array(())
all_mean_result_ex_SEN = np.array(())
all_mean_result_ex_SPE =np.array(())
all_mean_result_ex_AUC = np.array(())
all_mean_result_ex_F1 = np.array(())

all_result_ex = np.zeros((ex_num, ex_fold_num))
all_result_ex_SEN = np.zeros((ex_num, ex_fold_num))
all_result_ex_SPE= np.zeros((ex_num, ex_fold_num))
all_result_ex_AUC = np.zeros((ex_num, ex_fold_num))
all_result_ex_F1 = np.zeros((ex_num, ex_fold_num))
seed = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
# External five-fold cross-validation was repeated 10 times.
for index,jj in tqdm(enumerate(seed)):
    np.random.seed(jj)
    random.seed(jj)
    tf.random.set_seed(jj)
    dgl.seed(jj)
    kf = StratifiedKFold(n_splits=ex_fold_num,shuffle=True,random_state=jj)

    graph_dicts_1 = np.array(graph1)
    graph_dicts_2 = np.array(graph2)

    graph_array_3_4 = np.vstack([graph3,graph4])
    graph_array_3_4 = graph_array_3_4.T

    graph_array_1_2 = np.vstack(
        [graph_dicts_1, graph_dicts_2])
    graph_array_1_2 = graph_array_1_2.T

    num_classes = np.max(label_all) + 1
    h = 0
    everyresult_ex = np.array(())
    everyresult_ex_SEN = np.array(())
    everyresult_ex_SPE = np.array(())
    everyresult_ex_AUC = np.array(())
    everyresult_ex_F1 = np.array(())
    for train_index, test_index in tqdm(kf.split(graph_array_1_2, label_all)):
        h += 1
        train_graphs_1_2, test_graphs_1_2 = graph_array_1_2[train_index], graph_array_1_2[test_index]
        train_label, test_label = label_all[train_index], label_all[test_index]
        train_graphs_3_4, test_graphs_3_4 = graph_array_3_4[train_index],graph_array_3_4[test_index]

        test_samples_1 = get_samples(test_graphs_1_2, test_label, 0)
        test_samples_2 = get_samples(test_graphs_1_2, test_label, 1)


        all_brain_num_result = np.array(())
        brain_list = [7,8,9]
        # Parameter optimization; Take the optimization of PT parameters in a small range as an example.
        for _brain_ in brain_list:
            train_graphs_3 = train_graphs_3_4[:,_brain_]
            train_graphs_4 = train_graphs_3_4[:,_brain_+10]
            train_graphs_3_4 = np.vstack([train_graphs_3,train_graphs_4]).T

            num_train_data_ex = train_graphs_1_2.shape[0]
            all_mean_result = np.array(())
            in_all_result = np.zeros((in_num,in_fold_num))
            # Internal five-fold cross-validation was repeated five times.
            for pp in range(in_num):
                kf_in = StratifiedKFold(n_splits=in_fold_num,shuffle=True,random_state=pp)
                everyresult = np.array(())
                m = 0
                for train_index_in,dev_index in kf_in.split(train_graphs_1_2,train_label):

                    m +=1
                    train_X_in,train_y_in = train_graphs_1_2[train_index_in],train_label[train_index_in]
                    dev_X,dev_y = train_graphs_1_2[dev_index],train_label[dev_index]
                    train_X_in_3_4,dev_X_3_4 = train_graphs_3_4[train_index_in],train_graphs_3_4[dev_index]

                    num_train_data_in = train_X_in.shape[0]


                    train_samples_1 = get_samples(train_X_in, train_y_in, 0)
                    train_samples_2 = get_samples(train_X_in, train_y_in, 1)
                    train_samples_3 = get_samples(train_X_in_3_4, train_y_in, 0)
                    train_samples_4 = get_samples(train_X_in_3_4, train_y_in, 1)

                    dev_samples_1 = get_samples(dev_X, dev_y, 0)
                    dev_samples_2 = get_samples(dev_X, dev_y, 1)
                    dev_samples_3 = get_samples(dev_X_3_4, dev_y, 0)
                    dev_samples_4 = get_samples(dev_X_3_4, dev_y, 1)

                    train_data_1234 = [train_samples_1, train_samples_2,
                                                           train_samples_3,
                                                           train_samples_4]
                    train_batch_generator_1234 = create_graph_generator_train_all_path(
                        train_data_1234, batch_size, shuffle=True,
                        infinite=True)


                    model = MGATBC(ROIs, new_node_feature_num, 1, num_classes)
                    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

                    checkpoint = tf.train.Checkpoint(myAwesomeModel=model)
                    # training
                    for epoch in range(epochs):
                        bg1, bg2, bg3, bg4, label_final = next(train_batch_generator_1234)
                        with tf.GradientTape() as tape:
                            logits = model(bg1, bg2, bg3, bg4, training=True)
                            losses = tf.nn.softmax_cross_entropy_with_logits(
                                logits=logits,
                                labels=tf.one_hot(label_final, depth=num_classes)
                            )
                            losses = tf.reduce_mean(losses)
                        vars = tape.watched_variables()
                        grads = tape.gradient(losses, vars)
                        optimizer.apply_gradients(zip(grads, vars))
                    # save model
                    path = checkpoint.save(
                        r'trained_model\model_brain_%d_exnum_%d_exfold_%d_innum_%d_infold_%d.ckpt' % (_brain_,jj,h,pp,m))

                    dev_X_1, dev_Y = map(list, zip(*dev_samples_1))
                    dev_X_2, dev_Y = map(list, zip(*dev_samples_2))
                    dev_X_3, dev_Y = map(list, zip(*dev_samples_3))
                    dev_X_4, dev_Y = map(list, zip(*dev_samples_4))

                    dev_bg_1 = dgl.batch(dev_X_1)
                    dev_bg_2 = dgl.batch(dev_X_2)
                    dev_bg_3 = dgl.batch(dev_X_3)
                    dev_bg_4 = dgl.batch(dev_X_4)
                    probs_Y = tf.nn.softmax(
                        model(dev_bg_1, dev_bg_2, dev_bg_3, dev_bg_4, training=False))
                    argmax_Y = np.argmax(probs_Y, axis=-1)

                    dev_acc = sum((dev_Y == argmax_Y)) / len(dev_Y)
                    everyresult = np.append(everyresult, dev_acc)
                in_all_result[pp,:] = everyresult

                mean_acc_in = (np.sum(everyresult)) / in_fold_num
                all_mean_result = np.append(all_mean_result,mean_acc_in)

            acc_final = (np.sum(all_mean_result)) / in_num
            np.savetxt(r"result\indicator_in\exnum_%d_exfold_%d_brain_%d_of_in_acc .txt" %
                       (jj,h,_brain_), acc_final.reshape(-1,1) ,fmt="%.18f",delimiter=',')

            all_brain_num_result = np.append(all_brain_num_result,acc_final)
        # optimal parameters
        optim_brain_num = np.argmax(all_brain_num_result) + brain_list[0]
        print("ex_num_%d, ex_fold_%d,optim brain:" % (jj,h),optim_brain_num)

        all_ex_result = ()
        # The best twenty-five classifiers make predictions on the test set separately and perform majority voting.
        for innum in range(in_num):
            every_fold_ex = np.zeros((in_fold_num,test_num))
            for infold in range(in_fold_num):

                model_to_be_restored = MGATBC_re(ROIs, new_node_feature_num, 1, num_classes)  # 建立新的模型
                checkpoint = tf.train.Checkpoint(myAwesomeModel=model_to_be_restored)
                checkpoint.restore(
                    r'trained_model\model_brain_%d_exnum_%d_exfold_%d_innum_%d_infold_%d.ckpt-1' % (optim_brain_num,jj,h,innum,infold+1))

                test_samples_3 = get_samples(test_graphs_3_4, test_label, optim_brain_num)
                test_samples_4 = get_samples(test_graphs_3_4, test_label, optim_brain_num + 10)

                test_X_1, test_Y = map(list, zip(*test_samples_1))
                test_X_2, test_Y = map(list, zip(*test_samples_2))
                test_X_3, test_Y = map(list, zip(*test_samples_3))
                test_X_4, test_Y = map(list, zip(*test_samples_4))

                test_bg_1 = dgl.batch(test_X_1)
                test_bg_2 = dgl.batch(test_X_2)
                test_bg_3 = dgl.batch(test_X_3)
                test_bg_4 = dgl.batch(test_X_4)
                y_pred = tf.nn.softmax(
                    model_to_be_restored(test_bg_1, test_bg_2, test_bg_3, test_bg_4, training=False).numpy())

                y_predddd = np.argmax(y_pred, axis=-1)
                every_fold_ex[infold,:] = y_predddd
            all_ex_result += (every_fold_ex,)
        all_ex_result_y_pred = np.vstack((all_ex_result[0],all_ex_result[1],all_ex_result[2],all_ex_result[3],all_ex_result[4]))

        y_pred_fusion = np.sum(all_ex_result_y_pred,axis=0)
        y_pred_final = np.array(())
        y_fusion_prob = np.zeros((test_num,2))
        test_X_1, test_Y = map(list, zip(*test_samples_1))
        for bb in range(test_num):
            one_prob = y_pred_fusion[bb] /classifiers
            zero_prob = 1-(y_pred_fusion[bb] /classifiers)
            prob = np.array((zero_prob,one_prob))
            y_fusion_prob[bb,:] = prob

            if (y_pred_fusion[bb] / classifiers)>=0.5:
                y_pred_final = np.append(y_pred_final,1)
            else:
                y_pred_final = np.append(y_pred_final,0)
        y_true_ex = test_Y


        final_acc = sum(y_pred_final == y_true_ex) / test_num
        SEN_TPR =metrics.recall_score(y_true_ex,y_pred_final,pos_label=1)
        SPE_TNR = metrics.recall_score(y_true_ex,y_pred_final,pos_label=0)
        AUC = metrics.roc_auc_score(y_true_ex,y_fusion_prob[:, 1])
        F1 = metrics.f1_score(y_true_ex,y_pred_final)

        everyresult_ex = np.append(everyresult_ex, final_acc)
        everyresult_ex_SEN = np.append(everyresult_ex_SEN, SEN_TPR)
        everyresult_ex_SPE = np.append(everyresult_ex_SPE, SPE_TNR)
        everyresult_ex_AUC = np.append(everyresult_ex_AUC, AUC)
        everyresult_ex_F1 = np.append(everyresult_ex_F1, F1)

    all_result_ex[index,:] = everyresult_ex
    all_result_ex_SEN[index, :] = everyresult_ex_SEN
    all_result_ex_SPE[index, :] = everyresult_ex_SPE
    all_result_ex_AUC[index, :] = everyresult_ex_AUC
    all_result_ex_F1[index, :] = everyresult_ex_F1

    mean_acc_ex = (np.sum(everyresult_ex)) / ex_fold_num
    mean_sen_ex = (np.sum(everyresult_ex_SEN)) / ex_fold_num
    mean_spe_ex = (np.sum(everyresult_ex_SPE)) / ex_fold_num
    mean_auc_ex = (np.sum(everyresult_ex_AUC)) / ex_fold_num
    mean_f1_ex = (np.sum(everyresult_ex_F1)) / ex_fold_num

    all_mean_result_ex = np.append(all_mean_result_ex,mean_acc_ex)
    all_mean_result_ex_SEN = np.append(all_mean_result_ex_SEN,mean_sen_ex)
    all_mean_result_ex_SPE = np.append(all_mean_result_ex_SPE,mean_spe_ex)
    all_mean_result_ex_AUC = np.append(all_mean_result_ex_AUC,mean_auc_ex)
    all_mean_result_ex_F1 = np.append(all_mean_result_ex_F1,mean_f1_ex)

    mean_ex_dataframe = pd.DataFrame(
        list({
            'acc':mean_acc_ex,
            'sen':mean_sen_ex,
            'spe':mean_spe_ex,
            'auc':mean_auc_ex,
            'f1':mean_f1_ex,
        }.items()),
    )
    outputpath1 = r'result\indicator_ex\every_mean_acc_num_%d.csv'% (jj)
    mean_ex_dataframe.to_csv(outputpath1, sep=',', index=True, header=True)



    all_ex_dataframe = pd.DataFrame(
        {
            'n-fold':['1','2','3','4','5'],
            'acc':list(everyresult_ex),
            'sen':list(everyresult_ex_SEN),
            'spe':list(everyresult_ex_SPE),
            'auc':list(everyresult_ex_AUC),
            'f1':list(everyresult_ex_F1),
        },
    )
    outputpath2 = r"result\indicator_ex\all_acc_num_%d.csv" % (jj)
    all_ex_dataframe.to_csv(outputpath2, sep=',', index=True, header=True)


acc_final_ex = (np.sum(all_mean_result_ex)) / ex_num
sen_final_ex = (np.sum(all_mean_result_ex_SEN)) / ex_num
spe_final_ex = (np.sum(all_mean_result_ex_SPE)) / ex_num
auc_final_ex = (np.sum(all_mean_result_ex_AUC)) / ex_num
f1_final_ex = (np.sum(all_mean_result_ex_F1)) / ex_num

final_ex = pd.DataFrame(
    list({
        'acc': acc_final_ex,
        'sen': sen_final_ex,
        'spe': spe_final_ex,
        'auc': auc_final_ex,
        'f1': f1_final_ex,
    }.items())
)
outputpath3 = r"result\indicator_ex\final_ex.csv"
final_ex.to_csv(outputpath3, sep=',', index=True, header=True)




