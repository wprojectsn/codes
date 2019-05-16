from __future__ import unicode_literals, print_function, division

import sys

reload(sys)
sys.setdefaultencoding('utf8')

import os
import time

import tensorflow as tf
import torch
import random
import numpy as np
from model import Model
from torch.nn.utils import clip_grad_norm

from custom_adagrad import AdagradCustom
from torch.autograd import Variable

from data_util import data, config
from data_util.batcher import Batcher
from data_util.data import Vocab, Concept_vocab
from data_util.utils import calc_running_avg_loss
from train_util import get_input_from_batch, get_output_from_batch

use_cuda = config.use_gpu and torch.cuda.is_available()

class Train(object):
    def __init__(self):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.concept_vocab = Concept_vocab(config.concept_vocab_path, config.vocab_size)
        self.batcher = Batcher(config.train_data_path, self.vocab, self.concept_vocab, mode='train',
                               batch_size=config.batch_size, single_pass=False)
        self.ds_batcher = Batcher(config.train_ds_data_path, self.vocab, self.concept_vocab, mode='train',
                               batch_size=500, single_pass=False)
        time.sleep(15)
        
        train_dir = os.path.join(config.log_root, 'train_%d' % (int(time.time())))
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)

        self.model_dir = os.path.join(train_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        self.summary_writer = tf.summary.FileWriter(train_dir)

    def save_model(self, running_avg_loss, iter):
        state = {
            'iter': iter,
            'encoder_state_dict': self.model.encoder.state_dict(),
            'decoder_state_dict': self.model.decoder.state_dict(),
            'reduce_state_dict': self.model.reduce_state.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_loss
        }
        model_save_path = os.path.join(self.model_dir, 'model_%d_%d' % (iter, int(time.time())))
        torch.save(state, model_save_path)

    def setup_train(self, model_file_path=None):
        self.model = Model(model_file_path)

        params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()) + \
                 list(self.model.reduce_state.parameters())
        initial_lr = config.lr_coverage if config.is_coverage else config.lr
        self.optimizer = AdagradCustom(params, lr=initial_lr, initial_accumulator_value=config.adagrad_init_acc)

        start_iter, start_loss = 0, 0

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location= lambda storage, location: storage)
            start_iter = state['iter']
            start_loss = state['current_loss']

            if not config.is_coverage:
                self.optimizer.load_state_dict(state['optimizer'])
                if use_cuda:
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.cuda()

        return start_iter, start_loss
        

        
    def calc_Rouge_1(self,sub,string):
        new_sub = [str(x) for x in sub]
        new_sub.insert(0,'"')
        new_sub.append('"')
        token_c = ' '.join(new_sub) 
        summary = [[token_c]]
        new_string = [str(x) for x in string]
        new_string.insert(0,'"')
        new_string.append('"')
        token_r = ' '.join(new_string)
        reference = [[[token_r]]]
        
        rouge = Pythonrouge(summary_file_exist=False,
                    summary=summary, reference=reference,
                    n_gram=2, ROUGE_SU4=False, ROUGE_L=False,
                    recall_only=True, stemming=True, stopwords=True,
                    word_level=True, length_limit=True, length=30,
                    use_cf=False, cf=95, scoring_formula='average',
                    resampling=False, samples=10, favor=True, p=0.5)
        score = rouge.calc_score()
        return score['ROUGE-1']
        
    def calc_Rouge_2_recall(self, sub, string):
        token_c = sub
        token_r = string
        model = []
        ref = []
        if len(string) == 1 or len(string) == 1:
            score = 0.0
        else:
            i = 1
            while i < len(string):
                ref.append(str(token_r[i-1]) + str(token_r[i]))
                i += 1
            i = 1
            while i < len(sub):
                model.append(str(token_c[i-1]) + str(token_c[i]))
                i += 1
            sam = 0
            i = 0
            for i in range(len(ref)):
                for j in range(len(model)):
                    if ref[i] == model[j]:
                        sam += 1
                        model[j] = '-1'
                        break

            score = sam/float(len(ref))

        return score
        
    def calc_Rouge_L(self, sub, string):
        beta = 1.
        token_c = sub
        token_r = string
        if(len(string)< len(sub)):
            sub, string = string, sub
        lengths = [[0 for i in range(0,len(sub)+1)] for j in range(0,len(string)+1)]
        for j in range(1,len(sub)+1):
            for i in range(1,len(string)+1):
                if(string[i-1] == sub[j-1]):
                    lengths[i][j] = lengths[i-1][j-1] + 1
                else:
                    lengths[i][j] = max(lengths[i-1][j] , lengths[i][j-1])
        lcs = lengths[len(string)][len(sub)]
        
        prec = lcs/float(len(token_c))
        rec = lcs/float(len(token_r))

        if(prec!=0 and rec !=0):
            score = ((1 + beta**2)*prec*rec)/float(rec + beta**2*prec)
        else:
            score = 0.0
        return rec
        
    def calc_kl(self, dec, enc):
        kl = 0.
        dec = np.exp(dec)
        enc = np.exp(enc)
        all_dec = np.sum(dec)
        all_enc = np.sum(enc)
        for d,c in zip(dec,enc):
            d = d/all_dec
            c = c/all_enc
            kl = kl + c * np.log(c/d)
        return kl
        
    def calc_euc(self, dec, enc):
        euc = 0.
        for d,c in zip(dec,enc):
            euc = euc + np.sqrt(np.square(d - c))
        #print euc
        return euc
        
    def ds_loss(self, enc_batch_ds_emb, enc_padding_mask_ds, dec_batch_emb, dec_padding_mask):
        b1, t_k1, emb1 = list(enc_batch_ds_emb.size())
        b2, t_k2, emb2 = list(dec_batch_emb.size())
        enc_padding_mask_ds = enc_padding_mask_ds.unsqueeze(2).expand(b1, t_k1, emb1).contiguous()
        dec_padding_mask = dec_padding_mask.unsqueeze(2).expand(b2, t_k2, emb2).contiguous()
        enc_batch_ds_emb = enc_batch_ds_emb * enc_padding_mask_ds
        dec_batch_emb = dec_batch_emb * dec_padding_mask
        enc_batch_ds_emb = torch.sum(enc_batch_ds_emb,1)
        dec_batch_emb = torch.sum(dec_batch_emb,1)
        dec_title = dec_batch_emb.tolist()
        enc_article = enc_batch_ds_emb.tolist()
        dec_title_len = len(dec_title)
        enc_article_len = len(enc_article)
        dsloss = 0.
        for dec in dec_title:
            for enc in enc_article:
                dsloss = dsloss + self.calc_kl(dec,enc)
        dsloss = dsloss / float(dec_title_len * enc_article_len)
        print(dsloss)
        return dsloss
        
    def train_one_batch(self, batch,steps,batch_ds):
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, enc_batch_concept_extend_vocab, concept_p, position, concept_mask, extra_zeros, c_t_1, coverage = \
            get_input_from_batch(batch, use_cuda)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch, use_cuda)
        enc_batch_ds, enc_padding_mask_ds, enc_lens_ds, _, _, _, _, _, _, _, _ = \
            get_input_from_batch(batch_ds, use_cuda)
            
        self.optimizer.zero_grad()
        encoder_outputs, encoder_hidden, max_encoder_output, enc_batch_ds_emb, dec_batch_emb = self.model.encoder(enc_batch, enc_lens, enc_batch_ds, dec_batch)
        if config.DS_train:
            ds_final_loss = self.ds_loss(enc_batch_ds_emb, enc_padding_mask_ds, dec_batch_emb, dec_padding_mask)
        s_t_1 = self.model.reduce_state(encoder_hidden)
        s_t_0 = s_t_1
        c_t_0 = c_t_1
        if config.use_maxpool_init_ctx:
            c_t_1 = max_encoder_output
            c_t_0 = c_t_1

        step_losses = []
        for di in range(min(max_dec_len, config.max_dec_steps)):
            y_t_1 = dec_batch[:, di]
            final_dist, s_t_1,  c_t_1, attn_dist, p_gen, next_coverage = self.model.decoder('train', y_t_1, s_t_1,
                                                        encoder_outputs, enc_padding_mask, c_t_1,
                                                        extra_zeros, enc_batch_extend_vocab, enc_batch_concept_extend_vocab, concept_p, position, concept_mask, 
                                                                           coverage, di)
            target = target_batch[:, di]
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + config.eps)
            if config.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                coverage = next_coverage
                
            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)

        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_losses/dec_lens_var
        if config.DS_train:
            ds_final_loss = Variable(torch.FloatTensor([ds_final_loss]), requires_grad = False)
            ds_final_loss = ds_final_loss.cuda()
            loss = (config.pi - ds_final_loss)*torch.mean(batch_avg_loss)
        else:
            loss = torch.mean(batch_avg_loss)
        if steps > config.traintimes:
            scores = []
            sample_y = []
            s_t_1 = s_t_0
            c_t_1 = c_t_0
            for di in range(min(max_dec_len, config.max_dec_steps)):
                if di == 0:
                    y_t_1 = dec_batch[:, di]
                    sample_y.append(y_t_1.cpu().numpy().tolist())
                else:
                    sample_latest_tokens = sample_y[-1]
                    sample_latest_tokens = [t if t < self.vocab.size() else self.vocab.word2id(data.UNKNOWN_TOKEN) \
                                            for t in sample_latest_tokens]
                
                    y_t_1 = Variable(torch.LongTensor(sample_latest_tokens))
                    y_t_1 = y_t_1.cuda()
                
                final_dist, s_t_1, c_t_1, attn_dist, p_gen, next_coverage = self.model.decoder('train', y_t_1, s_t_1,
                                                            encoder_outputs, enc_padding_mask, c_t_1,
                                                            extra_zeros, enc_batch_extend_vocab,enc_batch_concept_extend_vocab, concept_p, position, concept_mask,
                                                                               coverage, di)
                sample_select = torch.multinomial(final_dist,1).view(-1)
                sample_log_probs = torch.gather(final_dist, 1, sample_select.unsqueeze(1)).squeeze()
                sample_y.append(sample_select.cpu().numpy().tolist())
                sample_step_loss = -torch.log(sample_log_probs + config.eps)
                sample_step_mask = dec_padding_mask[:, di]
                sample_step_loss = sample_step_loss * sample_step_mask
                scores.append(sample_step_loss)
            sample_sum_losses = torch.sum(torch.stack(scores, 1), 1)
            sample_batch_avg_loss = sample_sum_losses/dec_lens_var
        
            sample_y = np.transpose(sample_y).tolist()
        
        
            base_y = []
            s_t_1 = s_t_0
            c_t_1 = c_t_0
            for di in range(min(max_dec_len, config.max_dec_steps)):
                if di == 0:
                    y_t_1 = dec_batch[:, di]
                    base_y.append(y_t_1.cpu().numpy().tolist())
                else:
                    base_latest_tokens = base_y[-1]
                    base_latest_tokens = [t if t < self.vocab.size() else self.vocab.word2id(data.UNKNOWN_TOKEN) \
                                            for t in base_latest_tokens]
                
                    y_t_1 = Variable(torch.LongTensor(base_latest_tokens))
                    y_t_1 = y_t_1.cuda()
                
                final_dist, s_t_1, c_t_1, attn_dist, p_gen, next_coverage = self.model.decoder('train', y_t_1, s_t_1,
                                                            encoder_outputs, enc_padding_mask, c_t_1,
                                                            extra_zeros, enc_batch_extend_vocab,enc_batch_concept_extend_vocab, concept_p, position, concept_mask,
                                                                               coverage, di)
                base_log_probs, base_ids = torch.topk(final_dist, 1)
                base_y.append(base_ids[:,0].cpu().numpy().tolist())
        
            base_y = np.transpose(base_y).tolist()
        
            refs = dec_batch.cpu().numpy().tolist()
            sample_dec_lens_var = map(int,dec_lens_var.cpu().numpy().tolist())
            sample_rougeL = [self.calc_Rouge_L(sample[:reflen],ref[:reflen]) for sample,ref,reflen in zip(sample_y,refs,sample_dec_lens_var)]
            base_rougeL = [self.calc_Rouge_L(base[:reflen],ref[:reflen]) for base,ref,reflen in zip(base_y,refs,sample_dec_lens_var)]
            sample_rougeL = Variable(torch.FloatTensor(sample_rougeL), requires_grad = False)
            base_rougeL = Variable(torch.FloatTensor(base_rougeL), requires_grad = False)
            sample_rougeL = sample_rougeL.cuda()
            base_rougeL = base_rougeL.cuda()
            word_loss = -sample_batch_avg_loss * (base_rougeL - sample_rougeL)
            reinforce_loss = torch.mean(word_loss)
            loss = (1 - config.rein) * loss + config.rein * reinforce_loss
        
        loss.backward()

        clip_grad_norm(self.model.encoder.parameters(), config.max_grad_norm)
        clip_grad_norm(self.model.decoder.parameters(), config.max_grad_norm)
        clip_grad_norm(self.model.reduce_state.parameters(), config.max_grad_norm)

        self.optimizer.step()

        return loss.data[0]

    def trainIters(self, n_iters, model_file_path=None):
        iter, running_avg_loss = self.setup_train(model_file_path)
        start = time.time()
        while iter < n_iters:
            batch = self.batcher.next_batch()
            batch_ds = self.ds_batcher.next_batch()
            loss = self.train_one_batch(batch,iter,batch_ds)
            loss = loss.cpu()

            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, self.summary_writer, iter)
            iter += 1

            if iter % 100 == 0:
                self.summary_writer.flush()
            print_interval = 5
            if iter % print_interval == 0:
                print('steps %d , loss: %f' % (iter, loss))
                start = time.time()
            if iter % 50000 == 0:
                self.save_model(running_avg_loss, iter)

if __name__ == '__main__':
    model_filename = sys.argv[1]
    train_processor = Train()
    if model_filename == 'None':
        train_processor.trainIters(config.max_iterations)
    else:
        train_processor.trainIters(config.max_iterations,model_filename)
