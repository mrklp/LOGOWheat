# -*- coding:utf-8 -*-
import sys
import os
import h5py
import math
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from getopt import getopt
from scipy.spatial import distance
import tensorflow.keras.backend as K
from bgi.bert4keras.lamb import LAMB
from bgi.bert4keras.optimizers import Adam
from tensorflow.keras.layers import Lambda, Dense
from bgi.common.callbacks import LRSchedulerPerStep
from tensorflow.keras.callbacks import ModelCheckpoint
from bgi.bert4keras.models import build_transformer_model
from bgi.bert4keras.optimizers import extend_with_weight_decay
from bgi.bert4keras.optimizers import extend_with_layer_adaptation
from bgi.common.refseq_utils import get_word_dict_for_n_gram_number
from bgi.bert4keras.optimizers import extend_with_piecewise_linear_lr
from bgi.bert4keras.optimizers import extend_with_gradient_accumulation

os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

########################################### Parameters ############################################
usage = '''
#############################################################################################################################
#                                                                                                                           #
#  Usage: python getScore.py  -r refSeq -a altSeq -w  epi_weights.hdf5  -p 16 -o  ./variantscore/                           #
#                                                                                                                           #
#############################################################################################################################
'''

opts,args = getopt(sys.argv[1:],'r:a:o:p:w:h',['help'])
for opt_name, opt_value in opts:
    if opt_name in ("-h", "--help"):
        print(usage)
        sys.exit()
    elif opt_name == "-o":
        save_path = opt_value
    elif opt_name == "-p":
        num_parallel_calls = int(opt_value)
    elif opt_name == "-w":
        weight_path = opt_value
    elif opt_name == "-r":
        refSeq = opt_value
    elif opt_name == "-a":
        altSeq = opt_value
    else:
        print(usage)
        sys.exit()

test_output_path = save_path
########################################### Function ##############################################

def seqtonum(seq):
    acgt = {'A': 0,
            'G': 1,
            'C': 2,
            'T': 3}
    
    num = np.zeros((4,len(seq)),dtype=int)
    for i in range(len(seq)):
        if seq[i] != 'N':
            num[acgt[seq[i]],i] = 1
    return num

def load_npz_record(x_data):
    """
    parse_function
    """
    
    def parse_function(x_data):
        input_token = x_data
        input_segment = K.zeros_like(input_token, dtype='int64')
        x = {
            'Input-Token': input_token,
            'Input-Segment': input_segment,
        }
        return x
    
    dataset = tf.data.Dataset.from_tensor_slices((x_data))
    dataset = dataset.map(map_func=parse_function, num_parallel_calls=num_parallel_calls)
    return dataset

def getTestData():
    n_gram = 5
    stride = 1
    slice = 200000
    step=stride
    shuffle=False
    break_in_10w=False
    
    print("step:{}, kernel:{}, shuffle:{}, break_in_10w:{} \n".format(step, n_gram, shuffle, break_in_10w))
    actg_value = np.array([1, 2, 3, 4])
    n_gram_value = np.ones(n_gram)
    for ii in range(n_gram):
        n_gram_value[ii] = int(n_gram_value[ii] * (10 ** (n_gram - ii - 1)))
    
    print("n_gram_value: ", n_gram_value)
    num_word_dict = get_word_dict_for_n_gram_number(n_gram=n_gram)
    
    x_test = []
    test_counter = 1
    data = testArray
    index = 0
    for ii in tqdm(range(data.shape[0])):
        actg = np.matmul(actg_value, data[ii, :, :])
        gene = []
        for kk in range(0, len(actg), step):
            actg_temp_value = 0
            if kk + n_gram <= len(actg):
                actg_temp_value = np.dot(actg[kk:kk + n_gram], n_gram_value)
                actg_temp_value = int(actg_temp_value)
            else:
                for gg in range(kk, len(actg)):
                    actg_temp_value += actg[gg] * (10 ** (n_gram - gg % n_gram - 1))
                actg_temp_value = actg_temp_value * (10 ** (kk % n_gram))
            gene.append(num_word_dict.get(actg_temp_value, 0))
        
        x_test.append(np.array(gene))
        index += 1
        
        if index % 10000 == 0 and index > 0:
            print("Index:{}, Gene len:{}".format(index, len(gene)))
        
        if break_in_10w == True:
            if index % 100000 == 0 and index > 0:
                x_test = np.array(x_test);
                print(np.array(x_test).shape)
                save_dict = {
                    'x': x_test
                }
                save_file = os.path.join(test_output_path,
                                        '{}_{}_bp_{}_gram_{}_{}_step_{}.npz'.format( id,
                                                                                    (x_test.shape[1]),
                                                                                    n_gram,
                                                                                    index,
                                                                                    step,
                                                                                    test_counter))
                np.savez(save_file, **save_dict)
                print("Writing to", save_file)
                x_test = []
                test_counter += 1
        else:
            if len(x_test) == data.shape[0]:
                x_test = np.array(x_test)
                
                print(np.array(x_test).shape)
                
                save_dict = {
                    'x': x_test
                }
                save_file = os.path.join(test_output_path,
                                        '{}_{}_bp_{}_gram_{}_{}_step_{}.npz'.format( id,
                                                                                    (x_test.shape[1]),
                                                                                    n_gram,
                                                                                    index,
                                                                                    step,
                                                                                    test_counter))
                np.savez_compressed(save_file, **save_dict)
                print("Writing to", save_file)
                del x_test
                test_counter += 1
    
    print("Finish test data")

def getPred():
    batch_size = 512
    num_gpu = 1
    max_seq_len = 1000
    initial_epoch = 0
    ngram = 5
    stride = 1
    
    
    # Dynamic allocation of video memory
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    word_from_index = 10
    word_dict = get_word_dict_for_n_gram_number(word_index_from=word_from_index, n_gram=ngram)
    
    num_classes = 7
    only_one_slice = True
    vocab_size = len(word_dict) + word_from_index + 3
    
    max_depth = 2
    model_dim = 256
    embedding_size = 128
    num_heads = 8
    class_prediction = 'False'
    word_prediction = 'False'
    adversarial = 'False'
    use_position = 'True'
    use_segment = 'False'
    use_conv = 'True'
    verbose = 1
    
    shuffle_size = 4000
    prefetch_buffer_size = 4
    steps_per_epoch = 4000
    train_optimizer = 'adam'
    
    word_seq_len = max_seq_len // ngram
    print("max_seq_len: ", max_seq_len, " word_seq_len: ", word_seq_len)
    
    # Distributed Training
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    if strategy.num_replicas_in_sync >= 1:
        num_gpu = strategy.num_replicas_in_sync
    
    with strategy.scope():
        # Model configuration
        config = {
            "attention_probs_dropout_prob": 0,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0,
            "embedding_size": embedding_size,
            "hidden_size": model_dim,
            "initializer_range": 0.02,
            "intermediate_size": model_dim * 4,
            "max_position_embeddings": 512,
            "num_attention_heads": num_heads,
            "num_hidden_layers": max_depth,
            "num_hidden_groups": 1,
            "net_structure_type": 0,
            "gap_size": 0,
            "num_memory_blocks": 0,
            "inner_group_num": 1,
            "down_scale_factor": 1,
            "type_vocab_size": 0,
            "vocab_size": vocab_size,
            "custom_masked_sequence": False,
            "use_position_ids": use_position,
            "custom_conv_layer": use_conv,
            "use_segment_ids": use_segment
        }
        bert = build_transformer_model(
            configs=config,
            model='bert',
            return_keras_model=False,
        )
        
        output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)
        output = Dense(
            name='CLS-Activation',
            units=num_classes,
            activation='sigmoid',
            kernel_initializer=bert.initializer
        )(output)
        
        albert = tf.keras.models.Model(bert.model.input, output)
        albert.summary()
        
        # Optimizer
        optimizer = 'adam'
        albert.compile(optimizer=optimizer, loss=[tf.keras.losses.BinaryCrossentropy()],
                    metrics=['accuracy', tf.keras.metrics.AUC()])
    
    with strategy.scope():
        pretrain_weight_path = weight_path
        if pretrain_weight_path is not None and len(pretrain_weight_path) > 0:
            albert.load_weights(pretrain_weight_path, by_name=True)
            print("Load weights: ", pretrain_weight_path)
    
    lr_scheduler = LRSchedulerPerStep(model_dim,
                                    warmup=2500,
                                    initial_epoch=initial_epoch,
                                    steps_per_epoch=steps_per_epoch)
    
    loss_name = "val_loss"
    GLOBAL_BATCH_SIZE = batch_size * num_gpu
    print("GLOBAL_BATCH_SIZE: ", GLOBAL_BATCH_SIZE)
    print("shuffle_size: ", shuffle_size)
    # Read verification file
    
    files = os.listdir(test_output_path)
    test_slice_files = []
    for file_name in files:
        if str(file_name).endswith('.npz'):
            test_slice_files.append(os.path.join(test_output_path, file_name))
    
    print("test_slice_files: ", test_slice_files)
    x_valid_len = 0
    test_data = []
    for test_file in test_slice_files:
        loaded = np.load(test_file)
        x_test = loaded['x']
        
        x_valid_len = len(x_test)
        for ii in range(ngram):
            if only_one_slice is True:
                kk = ii  # random.randint(0, stride - 1)
                slice_indexes = []
                max_slice_seq_len = x_test.shape[1] // ngram * ngram
                for gg in range(kk, max_slice_seq_len, ngram):
                    slice_indexes.append(gg)
                x_test_slice = x_test[:, slice_indexes]
                test_data.append(x_test_slice)
            else:
                test_data.append(x_test)
    
    y_preds = []
    for ii in range(ngram):
        valid_dataset = load_npz_record(test_data[ii])
        valid_dataset = valid_dataset.batch(GLOBAL_BATCH_SIZE)
        valid_dataset = valid_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        y_pred = albert.predict(valid_dataset, steps=math.ceil(len(test_data[0]) / (GLOBAL_BATCH_SIZE)), verbose=1)
        y_preds.append(y_pred)
    
    return y_preds

############################################### Main ##############################################

os.system("rm %s*.npz"%(test_output_path))

testArray = np.array([seqtonum(refSeq),seqtonum(altSeq)]) 

getTestData()
y_preds = getPred()
y_pred = np.mean(y_preds,axis=0) 

print(
'''
########################################################################################

                                Variant Score: %s

        ATAC-Seq    H3K27ac    H3K27me3    H3K36me3    H3K4me1    H3K4me3    H3K9ac

RefSeq  %s

AltSeq  %s

########################################################################################
'''%(np.around(distance.euclidean(y_pred[0], y_pred[1]), decimals=6),'   '.join([str(i) for i in np.around(y_pred[0], decimals=6)]),
'   '.join([str(i) for i in np.around(y_pred[1], decimals=6)])))


###################################################################################################