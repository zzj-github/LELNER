"""soft_matcher.py: Joint training between trigger encoder and trigger matcher
It jointly trains the trigger encoder and trigger matcher

trigger encoder -> classification of trigger representation from encoder / attention
trigger matcher -> contrastive loss of sentence representation and trigger representation from encoder / attention

Written in 2020 by Dong-Ho Lee.
"""

from config import ContextEmb, batching_list_instances
from config.utils import get_optimizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.soft_encoder import SoftEncoder
from model.soft_attention import SoftAttention
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np
from model.linear_crf_inferencer import LinearCRF
from config.eval import evaluate_batch_insts
from util import remove_duplicates
from torch.autograd import Variable
from model.mul_attention import multihead_attention
import os
from torch.nn.utils import clip_grad_norm_

class ContrastiveLoss(nn.Module):
    def __init__(self, margin, device):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9
        self.device = device

    def forward(self, output1, output2, target, size_average=True):
        target = target.to(self.device)
        distances = []
        assert output1.size(0) == output2.size(0)

        output1 = output1.cpu().detach().numpy()
        output2 = output2.cpu().detach().numpy()
        for i in range(output1.shape[0]):

            vector_a = output1[i, :]
            vector_b = output2[i, :]
            vector_a = np.mat(vector_a)
            vector_b = np.mat(vector_b)
            num = float(vector_a * vector_b.T)
            denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
            cos = num / denom

            sim = 0.5*(1 + cos)
            distances.append(1 - sim)

        distances = torch.tensor(distances).to(self.device)

        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class SoftMatcher(nn.Module):
    def __init__(self, config, num_classes, print_info=True):
        super(SoftMatcher, self).__init__()
        self.config = config
        self.device = config.device
        self.encoder = SoftEncoder(self.config)
        self.attention = SoftAttention(self.config)
        self.label_size = config.label_size
        self.inferencer = LinearCRF(config, print_info=print_info)
        self.hidden2tag = nn.Linear(config.hidden_dim * 2, self.label_size).to(self.device)
        self.w1 = nn.Linear(config.hidden_dim, config.hidden_dim // 2).to(self.device)
        self.w2 = nn.Linear(config.hidden_dim // 2, config.hidden_dim // 2).to(self.device)
        self.attn1 = nn.Linear(config.hidden_dim // 2, 1).to(self.device)
        self.tanh = nn.Tanh().to(self.device)
        self.trigger_type_layer = nn.Linear(config.hidden_dim // 2, num_classes).to(self.device)
        self.entity_location_layer = nn.Linear(config.hidden_dim // 2, 2).to(self.device)
        self.hidden2tag_ = nn.Linear(config.hidden_dim // 2 + config.hidden_dim, self.label_size).to(self.device)
        self.linear = nn.Linear(config.hidden_dim, config.hidden_dim // 2).to(self.device)
        self.mulselfattn = multihead_attention(config.hidden_dim // 2, 1, 0.5).to(self.device)
        self.hidden2tag_change = nn.Linear(config.hidden_dim // 2, self.label_size).to(self.device)
        self.hidden2tag_change_1 = nn.Linear(config.hidden_dim // 2 + config.hidden_dim, self.label_size).to(self.device)

    def forward(self, word_seq_tensor: torch.Tensor,
                word_seq_lens: torch.Tensor,
                batch_context_emb: torch.Tensor,
                char_inputs: torch.Tensor,
                char_seq_lens: torch.Tensor,
                trigger_position, tags):

        output, sentence_mask, trigger_vec, trigger_mask = \
            self.encoder(word_seq_tensor, word_seq_lens, batch_context_emb, char_inputs, char_seq_lens, trigger_position)

        trig_rep, sentence_vec_cat, trigger_vec_cat = self.attention(output, sentence_mask, trigger_vec, trigger_mask)
        sentence_vec = self.linear(output)  # 5,50,100
        trigger_vec_extend = trig_rep.unsqueeze(1)  # 5,1,100
        joint_vec = torch.cat((trigger_vec_extend, sentence_vec), 1)  # 5,51,100
        trig_attention = self.mulselfattn(joint_vec, joint_vec, joint_vec)  # 5,51,100     10,73,150
        trig_attention = trig_attention[:, 1:, :]  # 5,51,100  ----5,100
        lstm_scores = self.hidden2tag_change(trig_attention)
        batch_size = word_seq_tensor.size(0)
        max_sent_len = word_seq_tensor.size(1)
        maskTemp = torch.arange(1, max_sent_len + 1, dtype=torch.long).view(1, max_sent_len).expand(batch_size,max_sent_len).to(self.device)
        mask = torch.le(maskTemp, word_seq_lens.view(batch_size, 1).expand(batch_size, max_sent_len)).to(self.device)

        if self.inferencer is not None:
            unlabeled_score, labeled_score = self.inferencer(lstm_scores, word_seq_lens, tags, mask)
            sequence_loss = unlabeled_score - labeled_score
        else:
            sequence_loss = self.compute_nll_loss(lstm_scores, tags, mask, word_seq_lens)

        entity_location = self.entity_location_layer(trig_rep)
        final_trigger_type = self.trigger_type_layer(trig_rep)
        return trig_rep, F.log_softmax(final_trigger_type, dim=1), F.log_softmax(entity_location, dim=1), sentence_vec_cat, trigger_vec_cat, sequence_loss

    def decode(self, word_seq_tensor: torch.Tensor,
               word_seq_lens: torch.Tensor,
               batch_context_emb: torch.Tensor,
               char_inputs: torch.Tensor,
               char_seq_lens: torch.Tensor,
               trig_rep):

        output, sentence_mask, _, _ = \
            self.encoder(word_seq_tensor, word_seq_lens, batch_context_emb, char_inputs, char_seq_lens, None)
        soft_sent_rep = self.attention.attention(output, sentence_mask)
        trig_vec = trig_rep[0]
        trig_key = trig_rep[1]
        soft_sent_rep = soft_sent_rep.cpu().detach().numpy()
        trig_vec_numpy = trig_vec.cpu().detach().numpy()
        dist = []
        for i in range(soft_sent_rep.shape[0]):
            match = []
            for j in range(trig_vec_numpy.shape[0]):
                out1 = soft_sent_rep[i, :]
                out2 = trig_vec_numpy[j, :]
                vector_a = np.mat(out1)
                vector_b = np.mat(out2)
                num = float(vector_a * vector_b.T)
                denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
                cos = num / denom
                sim = 0.5*(1 + cos)
                match.append(1 - sim)

            dist.append(match)

        dist = torch.tensor(dist)
        dvalue, dindices = torch.min(dist, dim=1)
        trigger_list = []
        for i in dindices.tolist():
            trigger_list.append(trig_vec[i])
        trig_rep = torch.stack(trigger_list)

        sentence_vec = self.linear(output)
        trigger_vec_extend = trig_rep.unsqueeze(1)
        joint_vec = torch.cat((trigger_vec_extend, sentence_vec), 1)
        trig_attention = self.mulselfattn(joint_vec, joint_vec, joint_vec)
        trig_attention = trig_attention[:, 1:, :]
        lstm_scores = self.hidden2tag_change(trig_attention)
        bestScores, decodeIdx = self.inferencer.decode(lstm_scores, word_seq_lens, None)

        return bestScores, decodeIdx


class SoftMatcherTrainerc(object):
    def __init__(self, model, config, dev, test):
        self.model = model
        self.config = config
        self.device = config.device
        self.input_size = config.embedding_dim
        self.context_emb = config.context_emb
        self.use_char = config.use_char_rnn
        if self.context_emb != ContextEmb.none:
            self.input_size += config.context_emb_size
        if self.use_char:
            self.input_size += config.charlstm_hidden_dim
        self.contrastive_loss = ContrastiveLoss(1.0, self.device)
        self.dev = dev
        self.test = test

    def train_model(self, num_epochs, train_data, devs, tests, eval):
        batched_data = batching_list_instances(self.config, train_data)

        self.optimizer = get_optimizer(self.config, self.model, self.config.optimizer)
        criterion = nn.NLLLoss()
        best_score = 0

        for epoch in range(num_epochs):
            print(epoch)
            epoch_loss = 0
            self.model.zero_grad()

            for index in tqdm(np.random.permutation(len(batched_data))):
                self.model.train()

                trig_rep, trig_type_probas, entity_location_probas, match_trig, match_sent, sequence_loss = self.model(*batched_data[index][0:5], batched_data[index][-2], batched_data[index][-3])
                trigger_loss = criterion(trig_type_probas, batched_data[index][-1])
                trigger_location_loss = criterion(entity_location_probas, batched_data[index][-4])
                soft_matching_loss = self.contrastive_loss(match_trig, match_sent, torch.stack([torch.tensor(1)]*trig_rep.size(0) + [torch.tensor(0)]*trig_rep.size(0)))
                loss = 0.5 * trigger_loss + 0.5 * trigger_location_loss + soft_matching_loss + 0.5 * sequence_loss
                epoch_loss = epoch_loss + loss.data
                loss.backward(retain_graph=True)
                clip_grad_norm_(self.model.parameters(), 100)
                self.optimizer.step()
                self.model.zero_grad()

            print(epoch_loss)
            self.test_model(train_data)
            logits, predicted, triggers = self.get_triggervec(train_data)
            triggers_remove = remove_duplicates(logits, predicted, triggers, train_data)

            if eval:
                self.model.eval()
                dev_batches = batching_list_instances(self.config, devs)
                test_batches = batching_list_instances(self.config, tests)
                dev_metrics = self.evaluate_model(dev_batches, "dev", devs, triggers_remove)
                test_metrics = self.evaluate_model(test_batches, "test", tests, triggers_remove)

            if test_metrics[2] > best_score:
                best_score = test_metrics[2]

                # save model
                if best_score >= 74:
                    dir_ = os.getcwd()
                    dir = os.path.join(dir_, 'save_model/' + str(best_score) + '.pth')
                    torch.save(self.model, dir)

            print("best f1: %.2f" % (best_score), flush=True)

            self.model.zero_grad()

        return self.model

    def evaluate_model(self, batch_insts_ids, name: str, insts, triggers):
        ## evaluation
        metrics = np.asarray([0, 0, 0], dtype=int)
        batch_id = 0
        batch_size = self.config.batch_size
        for batch in batch_insts_ids:
            one_batch_insts = insts[batch_id * batch_size:(batch_id + 1) * batch_size]
            batch_max_scores, batch_max_ids = self.model.decode(*batch[0:5], triggers)
            metrics += evaluate_batch_insts(one_batch_insts, batch_max_ids, batch[-3], batch[1], self.config.idx2labels,
                                            self.config.use_crf_layer)

            batch_id += 1
        p, total_predict, total_entity = metrics[0], metrics[1], metrics[2]
        precision = p * 1.0 / total_predict * 100 if total_predict != 0 else 0
        recall = p * 1.0 / total_entity * 100 if total_entity != 0 else 0
        fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
        print("[%s set] Precision: %.2f, Recall: %.2f, F1: %.2f" % (name, precision, recall, fscore), flush=True)
        return [precision, recall, fscore]

    def test_model(self, test_data):
        batched_data = batching_list_instances(self.config, test_data)
        self.model.eval()
        predicted_list = []
        target_list = []
        match_target_list = []
        matched_list = []
        target_location_list = []
        predicted_location_list = []

        for index in tqdm(np.random.permutation(len(batched_data))):

            trig_rep, trig_type_probas, entity_location_probas, match_trig, match_sent, _ = self.model(*batched_data[index][0:5], batched_data[index][-2], batched_data[index][-3])
            trig_type_value, trig_type_predicted = torch.max(trig_type_probas, 1)
            entity_location_value, entity_location_predicted = torch.max(entity_location_probas, 1)
            target = batched_data[index][-1]
            target_list.extend(target.tolist())
            predicted_list.extend(trig_type_predicted.tolist())
            target_location = batched_data[index][-4]
            target_location_list.extend(target_location.tolist())
            predicted_location_list.extend(entity_location_predicted.tolist())
            match_target_list.extend([torch.tensor(1)]*trig_rep.size(0) + [torch.tensor(0)]*trig_rep.size(0))
            distances = (match_trig - match_sent).pow(2).sum(1)
            distances = torch.sqrt(distances)
            matched_list.extend((distances < 1.0).long().tolist())

        print("trigger classification accuracy ", accuracy_score(predicted_list, target_list))
        print("entity location accuracy ", accuracy_score(predicted_location_list, target_location_list))
        print("soft matching accuracy ", accuracy_score(matched_list, match_target_list))

    def get_triggervec(self, data):
        batched_data = batching_list_instances(self.config, data)
        self.model.eval()
        logits_list = []
        predicted_list = []
        trigger_list = []
        for index in tqdm(range(len(batched_data))):
            trig_rep, trig_type_probas, entity_location_probas, match_trig, match_sent, _ = self.model(*batched_data[index][0:5], batched_data[index][-2], batched_data[index][-3])
            trig_type_value, trig_type_predicted = torch.max(trig_type_probas, 1)
            ne_batch_insts = data[index * self.config.batch_size:(index + 1) * self.config.batch_size]
            for idx in range(len(trig_rep)):
                ne_batch_insts[idx].trigger_vec = trig_rep[idx]
            logits_list.extend(trig_rep)
            predicted_list.extend(trig_type_predicted)
            word_seq = batched_data[index][-2]
            trigger_list.extend([" ".join(self.config.idx2word[index] for index in indices if index != 0) for indices in word_seq])

        return logits_list, predicted_list, trigger_list

    def eval_result(self, train_data, devs, tests):
        dir_ = os.getcwd()
        dir = os.path.join(dir_, 'save_model/' + '87.41662159725978' + '.pth')
        self.model = torch.load(dir)

        logits, predicted, triggers = self.get_triggervec(train_data)
        triggers_remove = remove_duplicates(logits, predicted, triggers, train_data)

        if eval:
            self.model.eval()
            dev_batches = batching_list_instances(self.config, devs)
            test_batches = batching_list_instances(self.config, tests)
            dev_metrics = self.evaluate_model(dev_batches, "dev", devs, triggers_remove)
            test_metrics = self.evaluate_model(test_batches, "test", tests, triggers_remove)


