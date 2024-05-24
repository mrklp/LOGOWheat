# Language of Genome for Wheat (LOGOWheat)
Deep learning-based prediction of regulatory effects for noncoding variants in wheats

# Usage
```
python getScore.py  -r refSeq -a altSeq -w  epi_weights.hdf5  -p 16 -o  ./variantscore/  

python getMap.py  -r refSeq -i chr6D_+_329839800_329840000 -w  epi_weights.hdf5  -p 16 -o  ./variantscore/

```

# Installation
```
conda create --name logowheat python==3.6.9 tensorflow-gpu==2.0 keras==2.3.1 numpy pandas tqdm scipy scikit-learn matplotlib jupyter notebook nb_conda
conda activate logowheat
pip install biopython==1.68
```

# Pre-training
```
1. Download the reference sequence of the bread wheat variety Chinese Spring
wget -c https://urgi.versailles.inra.fr/download/iwgsc/IWGSC_RefSeq_Assemblies/v1.0/Triticum_aestivum_IWGSC_v1.0.fa

2. To generate ref sequence, about 60G of space is required:
python 00_generate_refseq_sequence.py \
  --data Triticum_aestivum_IWGSC_v1.0.fa \
  --output ./train_5_gram \
  --chunk-size 10000 \
  --seq-size 1000 \
  --seq-stride 100 \
  --ngram 5 \
  --stride 1 \
  --slice-size 100000 \
  --hg-name Triticum_aestivum \
  --pool-size 100

3. To generate tfrecord, about 1T of space is required
python 01_generate_DNA_refseq_tfrecord.py \
  --data ./train_5_gram \
  --output ./train_5_gram_tfrecord \
  --chunk-size 10000 \
  --seq-size 1000 \
  --seq-stride 100 \
  --ngram 5 \
  --stride 1 \
  --slice-size 100000 \
  --hg-name Triticum_aestivum \
  --pool-size 100

4. Perform DNA sequence pre-training
python 02_train_gene_transformer_lm_hg_bert4keras_tfrecord.py \
  --save ./gram_5_weights \
  --train-data ./train_5_gram_tfrecord \
  --seq-len 1000 \
  --model-dim 256 \
  --transformer-depth 2 \
  --num-heads 8 \
  --batch-size 256 \
  --ngram 5 \
  --stride 5 \
  --model-name bert_5_gram_2_layer_8_heads_256_dim \
  --steps-per-epoch 4000 \
  --shuffle-size 4000   \
  --num-gpu=1
```

# Fine-tuning
```
5. Generate ref sequence for epigenomic data
python 03_epigenomic_data_loader.py   \
  --data /path/for/train/valid/test/mat/file/   \
  --output ./mat2npz/   \
  --ngram 5   \
  --stride 1   \
  --slice 200000   \
  --pool-size 100

6. Generate tfrecord for epigenomic data
python 04_epigenomic_tfrecord_utils.py \
  --data ./mat2npz/train_5_gram \
  --output ./mat2npz/train_5_gram_classification_tfrecord \
  --ngram 5 \
  --stride 1 \
  --slice 200000 \
  --pool-size 52 \
  --task classification

7. Perform epigenomic data fine-tuning
python 05_epigenomic_train_classification_tfrecord.py \
  --save ./epi_wheights/ \
  --weight-path ./gram_5_weights/bert_5_gram_2_layer_8_heads_256_dim_weights_100-0.887565.hdf5 \
  --train-data ./mat2npz/train_5_gram_classification_tfrecord  \
  --test-data ./mat2npz/test_5_gram_classification_tfrecord \
  --valid-data ./mat2npz/valid_5_gram_classification_tfrecord \
  --seq-len 1000 \
  --we-size 128 \
  --model-dim 256 \
  --transformer-depth 2 \
  --num-heads 8 \
  --batch-size 512 \
  --epochs 150 \
  --ngram 5 \
  --stride 1 \
  --num-classes 7 \
  --model-name epi_5_gram_2_layer_8_heads_256_dim \
  --steps-per-epoch 4000 \
  --shuffle-size 4000 \
  --use-conv \
  --use-position \
  --verbose 1 \
  --task train
```

# Predicting 
```
python 06_deep_sea_train_classification_tfrecord_for_test.py \
  --save ./validpred/ \
  --weight-path ./epi_wheights/epi_5_gram_2_layer_8_heads_256_dim_weights_100-0.792101-0.846889.hdf5 \
  --test-data ./mat2npz/valid_5_gram \
  --seq-len 1000 \
  --we-size 128 \
  --model-dim 256 \
  --transformer-depth 2 \
  --num-heads 8 \
  --batch-size 512 \
  --epochs 100 \
  --ngram 5 \
  --stride 1 \
  --num-classes 7 \
  --model-name epi_5_gram_2_layer_8_heads_256_dim \
  --steps-per-epoch 4000 \
  --shuffle-size 4000 \
  --use-conv \
  --use-position \
  --verbose 1 \
  --task valid

python 06_deep_sea_train_classification_tfrecord_for_test.py \
  --save ./testpred/ \
  --weight-path ./epi_wheights/epi_5_gram_2_layer_8_heads_256_dim_weights_100-0.792101-0.846889.hdf5 \
  --test-data ./mat2npz/test_5_gram \
  --seq-len 1000 \
  --we-size 128 \
  --model-dim 256 \
  --transformer-depth 2 \
  --num-heads 8 \
  --batch-size 512 \
  --epochs 100 \
  --ngram 5 \
  --stride 1 \
  --num-classes 7 \
  --model-name epi_5_gram_2_layer_8_heads_256_dim \
  --steps-per-epoch 4000 \
  --shuffle-size 4000 \
  --use-conv \
  --use-position \
  --verbose 1 \
  --task test
```
