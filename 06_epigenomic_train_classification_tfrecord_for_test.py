# -*- coding:utf-8 -*-

import argparse
import json
import math
import os
import random
import sys
import numpy as np

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import roc_auc_score
from tensorflow.keras.layers import Lambda, Dense
from multiprocessing import Pool
import tensorflow.keras.backend as K

sys.path.append("../../")
from bgi.bert4keras.models import build_transformer_model
from bgi.common.callbacks import LRSchedulerPerStep
from bgi.common.refseq_utils import get_word_dict_for_n_gram_number

from bgi.bert4keras.optimizers import Adam
from bgi.bert4keras.optimizers import extend_with_weight_decay
from bgi.bert4keras.optimizers import extend_with_layer_adaptation
from bgi.bert4keras.optimizers import extend_with_piecewise_linear_lr
from bgi.bert4keras.optimizers import extend_with_gradient_accumulation

from bgi.bert4keras.lamb import LAMB

def load_npz_record(x_data, y_data):
    """
    parse_function
    """

    def parse_function(x_data, y_data):
        input_token = x_data
        input_segment = K.zeros_like(input_token, dtype='int64')
        x = {
            'Input-Token': input_token,
            'Input-Segment': input_segment,
        }
        y = {
            'CLS-Activation': y_data
        }

        return x, y

    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
    dataset = dataset.map(map_func=parse_function, num_parallel_calls=num_parallel_calls)
    return dataset

if __name__ == '__main__':

    _argparser = argparse.ArgumentParser(
        description='A simple example of the Transformer language model in Genomics',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _argparser.add_argument(
        '--save', type=str, required=True, metavar='PATH',
        help='A path where the best model should be saved / restored from')
    _argparser.add_argument(
        '--model-name', type=str, default='transformer_gene', metavar='NAME',
        help='The name of the saving model')
    # _argparser.add_argument(
    #     '--tensorboard-log', type=str, metavar='PATH', default=None,
    #     help='Path to a directory for Tensorboard logs')
    _argparser.add_argument(
        '--train-data', type=str, metavar='PATH', default=None,
        help='Path to a file of train data')
    _argparser.add_argument(
        '--valid-data', type=str, metavar='PATH', default=None,
        help='Path to a file of valid data')
    _argparser.add_argument(
        '--test-data', type=str, metavar='PATH', default=None,
        help='Path to a file of test data')
    _argparser.add_argument(
        '--weight-path', type=str, metavar='PATH', default=None,
        help='Path to a pretain weight')
    _argparser.add_argument(
        '--epochs', type=int, default=150, metavar='INTEGER',
        help='The number of epochs to train')
    _argparser.add_argument(
        '--lr', type=float, default=2e-4, metavar='FLOAT',
        help='Learning rate')
    _argparser.add_argument(
        '--batch-size', type=int, default=32, metavar='INTEGER',
        help='Training batch size')
    _argparser.add_argument(
        '--seq-len', type=int, default=256, metavar='INTEGER',
        help='Max sequence length')
    _argparser.add_argument(
        '--we-size', type=int, default=128, metavar='INTEGER',
        help='Word embedding size')
    _argparser.add_argument(
        '--model', type=str, default='universal', metavar='NAME',
        choices=['universal', 'vanilla'],
        help='The type of the model to train: "vanilla" or "universal"')
    _argparser.add_argument(
        '--CUDA-VISIBLE-DEVICES', type=int, default=0, metavar='INTEGER',
        help='CUDA_VISIBLE_DEVICES')
    _argparser.add_argument(
        '--num-classes', type=int, default=10, metavar='INTEGER',
        help='Number of total classes')
    _argparser.add_argument(
        '--vocab-size', type=int, default=20000, metavar='INTEGER',
        help='Number of vocab')
    _argparser.add_argument(
        '--slice', type=list, default=[6],
        help='Slice')
    _argparser.add_argument(
        '--ngram', type=int, default=6, metavar='INTEGER',
        help='length of char ngram')
    _argparser.add_argument(
        '--stride', type=int, default=2, metavar='INTEGER',
        help='stride size')
    _argparser.add_argument(
        '--has-segment', action='store_true',
        help='Include segment ID')
    _argparser.add_argument(
        '--word-prediction', action='store_true',
        help='Word prediction')
    _argparser.add_argument(
        '--class-prediction', action='store_true',
        help='class prediction')
    _argparser.add_argument(
        '--adversarial', action='store_true',
        help='adversarial training')
    _argparser.add_argument(
        '--num-heads', type=int, default=4, metavar='INTEGER',
        help='Heads of self attention')
    _argparser.add_argument(
        '--model-dim', type=int, default=128, metavar='INTEGER',
        help='Heads of self attention')
    _argparser.add_argument(
        '--transformer-depth', type=int, default=2, metavar='INTEGER',
        help='Heads of self attention')
    _argparser.add_argument(
        '--num-gpu', type=int, default=1, metavar='INTEGER',
        help='Number of GPUs')
    _argparser.add_argument(
        '--task', type=str, default='train', metavar='NAME',
        choices=['train', 'valid', 'test'],
        help='The type of the task')
    _argparser.add_argument(
        '--verbose', type=int, default=2, metavar='INTEGER',
        help='Verbose')
    _argparser.add_argument(
        '--steps-per-epoch', type=int, default=10000, metavar='INTEGER',
        help='steps per epoch')
    _argparser.add_argument(
        '--shuffle-size', type=int, default=1000, metavar='INTEGER',
        help='Buffer shuffle size')
    _argparser.add_argument(
        '--num-parallel-calls', type=int, default=16, metavar='INTEGER',
        help='Num parallel calls')
    _argparser.add_argument(
        '--prefetch-buffer-size', type=int, default=4, metavar='INTEGER',
        help='Prefetch buffer size')
    _argparser.add_argument(
        '--pool-size', type=int, default=16, metavar='INTEGER',
        help='Pool size of multi-thread')
    _argparser.add_argument(
        '--optimizer', type=str, default='adam', metavar='NAME',
        choices=['adam', 'lamb'],
        help='The type of the task')
    _argparser.add_argument(
        '--use-position', action='store_true',
        help='Using position ids')
    _argparser.add_argument(
        '--use-segment', action='store_true',
        help='Using segment ids')
    _argparser.add_argument(
        '--use-conv', action='store_true',
        help='Using Conv1D layer')

    _args = _argparser.parse_args()

    save_path = _args.save

    model_name = _args.model_name
    config_save_path = os.path.join(save_path, "{}_config.json".format(model_name))

    batch_size = _args.batch_size
    epochs = _args.epochs
    num_gpu = _args.num_gpu

    max_seq_len = _args.seq_len
    initial_epoch = 0

    # Dynamic allocation of video memory
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    ngram = _args.ngram
    stride = _args.stride

    word_from_index = 10
    word_dict = get_word_dict_for_n_gram_number(word_index_from=word_from_index, n_gram=ngram)

    num_classes = _args.num_classes
    only_one_slice = True
    # vocab_size = len(word_dict) + word_from_index
    vocab_size = len(word_dict) + word_from_index + 3

    slice = _args.slice

    max_depth = _args.transformer_depth
    model_dim = _args.model_dim
    embedding_size = _args.we_size
    num_heads = _args.num_heads
    class_prediction = _args.class_prediction
    word_prediction = _args.word_prediction
    adversarial = _args.adversarial

    use_position = _args.use_position
    use_segment = _args.use_segment
    use_conv = _args.use_conv
    verbose = _args.verbose

    shuffle_size = _args.shuffle_size
    num_parallel_calls = _args.num_parallel_calls
    prefetch_buffer_size = _args.prefetch_buffer_size
    steps_per_epoch = _args.steps_per_epoch
    train_optimizer = _args.optimizer

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
            # checkpoint_path=checkpoint_path,
            model='bert',
            return_keras_model=False,
        )

        output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)
        # output = tf.keras.layers.GlobalMaxPool1D()(bert.model.output)
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
        pretrain_weight_path = _args.weight_path
        if pretrain_weight_path is not None and len(pretrain_weight_path) > 0:
            albert.load_weights(pretrain_weight_path, by_name=True)
            print("Load weights: ", pretrain_weight_path)

    lr_scheduler = LRSchedulerPerStep(model_dim,
                                      warmup=2500,
                                      initial_epoch=initial_epoch,
                                      steps_per_epoch=steps_per_epoch)

    loss_name = "val_loss"
    # LOG_FILE_PATH = LoG_PATH + '_checkpoint-{}.hdf5'.format(epoch)
    filepath = os.path.join(save_path, model_name + "_weights_{epoch:02d}-{accuracy:.6f}-{val_accuracy:.6f}.hdf5")

    GLOBAL_BATCH_SIZE = batch_size * num_gpu
    print("GLOBAL_BATCH_SIZE: ", GLOBAL_BATCH_SIZE)
    print("shuffle_size: ", shuffle_size)

    mc = ModelCheckpoint(filepath, monitor=loss_name, save_best_only=False, verbose=1)

    is_training = False

    # Read verification file
    test_data_path = _args.test_data
    files = os.listdir(test_data_path)
    test_slice_files = []
    for file_name in files:
        if str(file_name).endswith('.npz'):
            test_slice_files.append(os.path.join(test_data_path, file_name))

    print("test_slice_files: ", test_slice_files)
    x_valid_len = 0
    test_data = []
    for test_file in test_slice_files:
        loaded = np.load(test_file)
        x_test = loaded['x']
        y_test = loaded['y']

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
        valid_dataset = load_npz_record(test_data[ii], y_test)
        valid_dataset = valid_dataset.batch(GLOBAL_BATCH_SIZE)
        valid_dataset = valid_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        y_pred = albert.predict(valid_dataset, steps=math.ceil(len(test_data[0]) / (GLOBAL_BATCH_SIZE)), verbose=1)
        y_preds.append(y_pred)

    np.save(save_path+"y_preds.npy",y_preds)
    np.save(save_path+"y_test.npy",y_test)
