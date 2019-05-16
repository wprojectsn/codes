import os

root_dir = os.path.expanduser("~")

train_data_path = os.path.join(root_dir, "/home/wwb/corpus/giga/giga_train_bin/*")
eval_data_path = os.path.join(root_dir, "/home/wwb/corpus/giga/giga_train_bin/*")
decode_data_path = os.path.join(root_dir, "/home/wwb/NHG/code/data/test/ductest_ref0")
train_ds_data_path = os.path.join(root_dir, "/home/wwb/NHG/code/data/ds/ductest")
vocab_path = os.path.join(root_dir, "/home/wwb/NHG/code/data/vocabulary/vocab")
log_root = os.path.join(root_dir, "/home/wwb/NHG/code/codespace/log_root")
concept_vocab_path = os.path.join(root_dir, "/home/wwb/NHG/code/data/vocabulary/concept_vocab")

traintimes = 450000
hidden_dim= 256
emb_dim= 128
batch_size= 64
max_enc_steps=60
max_dec_steps=20
beam_size=8
min_dec_steps=2
vocab_size=150000
concept_num = 2

lr=0.15
adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4
max_grad_norm=2.0

pointer_gen = True
is_coverage = False
DS_train = False
cov_loss_wt = 1.0
rein = 0.99
pi = 2.92

eps = 1e-12
max_iterations = 2000000

use_gpu=True

lr_coverage=0.15

use_maxpool_init_ctx = False
