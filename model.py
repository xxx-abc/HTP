from functools import total_ordering
import numpy as np
import torch
import sys
import torch
import torch.nn as nn


FLOAT_MIN = -sys.float_info.max

class GCN(torch.nn.Module):
    
    def __init__(self, dims, head_num, dev) -> None:
        super(GCN, self).__init__()

        self.dim = dims
        self.dev = dev
        self.head_size = self.dim // head_num
        self.conv1 = nn.Conv1d(self.dim, self.dim, kernel_size=1)
        self.conv1.weight.data = nn.init.xavier_uniform_(self.conv1.weight.data, gain=nn.init.calculate_gain('relu'))
        
        self.conv2 = nn.Conv1d(self.dim, self.dim, kernel_size=1)
        self.conv2.weight.data = nn.init.xavier_uniform_(self.conv1.weight.data, gain=nn.init.calculate_gain('relu'))


        self.time_weight = nn.Linear(self.dim, self.dim)
        self.time_weight2 = nn.Linear(self.dim, self.dim)
        self.W = nn.Linear(self.dim, self.dim)
        self.LN = torch.nn.LayerNorm(self.dim, eps=1e-8)

    def forward(self, seqs, attention_mask, time_matrices):
        
        # relu 作用在于去掉0
        a_ = self.conv1(seqs.permute(0, 2, 1)).permute(0, 2, 1)
        b_ = self.conv2(seqs.permute(0, 2, 1)).permute(0, 2, 1)

        a = torch.cat(torch.split(a_, self.head_size, dim=2), dim=0)
        b = torch.cat(torch.split(b_, self.head_size, dim=2), dim=0)
        time_information = torch.cat(torch.split(time_matrices, self.head_size, dim=3), dim=0)


        att = torch.matmul(a, b.permute(0, 2, 1))
        a_2 = torch.norm(a, dim=-1).reshape(-1, a.shape[1], 1)
        # time_information = time_matrices
        # time_information = self.time_weight(time_matrices)
        b_t = b.unsqueeze(1) + time_information   # B * N * N * d
        # b_2 = torch.norm(b, dim=-1).reshape(-1, a.shape[1], 1)
        b_t_2 = torch.norm(b_t, dim=-1).reshape(-1, a.shape[1], a.shape[1])


        att = att + time_information.matmul(a.unsqueeze(-1)).squeeze(-1)
        att_2 = a_2 * b_t_2
        # att_2 = torch.matmul(a_2, b_2.permute(0, 2, 1)) 

        raw_graph = att / (att_2 + 0.000001)  # B * N * N
        paddings = torch.zeros(raw_graph.shape)  # -1e23 # float('-inf')
        paddings = paddings.to(self.dev)
        attn_mask = attention_mask.unsqueeze(0).expand(raw_graph.shape[0], -1, -1) 
        raw_graph = torch.where(attn_mask, paddings, raw_graph)  

        _, indices = raw_graph.topk(k=3, dim=-1)
        mask = torch.zeros(raw_graph.shape).to(self.dev)
        # 改进算法
        for i in range(raw_graph.shape[1]):
            mask[torch.arange(raw_graph.shape[0]).view(-1, 1), i, indices[:, i, :]] = 1.
            mask[torch.arange(raw_graph.shape[0]).view(-1, 1), indices[:, i, :], i] = 1.
        mask.requires_grad = False
        
        sparse_graph = raw_graph * mask
        # 做mask
        sparse_graph = torch.where(attn_mask, paddings, sparse_graph)  # enforcing causality  # B * N * 1 * N 

        seqs_ = torch.cat(torch.split(self.W(seqs), self.head_size, dim=2), dim=0)

        outputs = sparse_graph.matmul(seqs_)
        # outputs = outputs + sparse_graph.unsqueeze(2).matmul(time_information).reshape(outputs.shape)   
        time_interval_outputs = sparse_graph.unsqueeze(2).matmul(time_information).reshape(outputs.shape)
        outputs = torch.cat(torch.split(outputs, seqs.shape[0], dim=0), dim=2)
        time_interval_outputs = torch.cat(torch.split(time_interval_outputs, seqs.shape[0], dim=0), dim=2)

        return self.LN(outputs),time_interval_outputs



class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate): # wried, why fusion X 2?

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class TimeAwareMultiHeadAttention(torch.nn.Module):
    # required homebrewed mha layer for Ti/SASRec experiments
    def __init__(self, hidden_size, head_num, dropout_rate, dev):
        super(TimeAwareMultiHeadAttention, self).__init__()
        self.Q_w = torch.nn.Linear(hidden_size, hidden_size)
        self.K_w = torch.nn.Linear(hidden_size, hidden_size)
        self.V_w = torch.nn.Linear(hidden_size, hidden_size)

        self.WA_1 = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.WA_2 = torch.nn.Linear(hidden_size, hidden_size, bias=False)

        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.softmax = torch.nn.Softmax(dim=-1)

        self.hidden_size = hidden_size
        self.head_num = head_num
        self.head_size = hidden_size // head_num
        self.dropout_rate = dropout_rate
        self.dev = dev

    def forward(self, item_embs, time_mask, attn_mask, time_matrix):


        queries, keys, value = item_embs, item_embs, item_embs

        Q, K, V = self.Q_w(queries), self.K_w(keys), self.V_w(keys)
        time_matrix_K = self.WA_1(time_matrix)
        time_matrix_V = self.WA_2(time_matrix)
        # head dim * batch dim for parallelization (h*N, T, C/h)
        Q_ = torch.cat(torch.split(Q, self.head_size, dim=2), dim=0)
        K_ = torch.cat(torch.split(K, self.head_size, dim=2), dim=0)
        V_ = torch.cat(torch.split(V, self.head_size, dim=2), dim=0)

        time_matrix_K_ = torch.cat(torch.split(time_matrix_K, self.head_size, dim=3), dim=0)
        time_matrix_V_ = torch.cat(torch.split(time_matrix_V, self.head_size, dim=3), dim=0)

        # batched channel wise matmul to gen attention weights
        attn_weights = Q_.matmul(torch.transpose(K_, 1, 2))
        attn_weights = attn_weights + time_matrix_K_.matmul(Q_.unsqueeze(-1)).squeeze(-1)

        # seq length adaptive scaling
        attn_weights = attn_weights / (K_.shape[-1] ** 0.5)

        time_mask = time_mask.unsqueeze(-1).repeat(self.head_num, 1, 1)
        time_mask = time_mask.expand(-1, -1, attn_weights.shape[-1])
        attn_mask = attn_mask.unsqueeze(0).expand(attn_weights.shape[0], -1, -1)

        paddings = torch.ones(attn_weights.shape) *  (-2**32+1) # -1e23 # float('-inf')
        paddings = paddings.to(self.dev)
        attn_weights = torch.where(time_mask, paddings, attn_weights) # True:pick padding
        attn_weights = torch.where(attn_mask, paddings, attn_weights) # enforcing causality

        attn_weights = self.softmax(attn_weights)

        attn_weights = self.dropout(attn_weights)
        outputs = attn_weights.matmul(V_)
        outputs = outputs + attn_weights.unsqueeze(2).matmul(time_matrix_V_).reshape(outputs.shape)
        # (num_head * N, T, C / num_head) -> (N, T, C)
        outputs = torch.cat(torch.split(outputs, Q.shape[0], dim=0), dim=2) # div batch_size
        return outputs


class HTP(torch.nn.Module):
    def __init__(self, user_num, item_num, yearnum, monthnum, daynum, args, item_time_matirx):
        super(HTP, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.year_num = yearnum
        self.day_num = daynum
        self.month_num = monthnum
    
        self.abs_num_head = args.abs_num_heads
        self.abs_head_size = args.hidden_units // self.abs_num_head
        self.ritm_head_size = args.hidden_units // args.ritm_num_heads
        self.args = args

        self.dev = args.device
        self.item_time_matrix = item_time_matirx.to(self.dev)
        self.item_emb = torch.nn.Embedding(self.item_num, args.hidden_units, padding_idx=0)
        self.year_emb = torch.nn.Embedding(self.year_num, args.hidden_units, padding_idx=0)
        self.month_emb = torch.nn.Embedding(self.month_num, args.hidden_units, padding_idx=0)
        self.day_emb = torch.nn.Embedding(self.day_num, args.hidden_units, padding_idx=0)


        self.item_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.year_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.month_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.day_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        
        # position encoding
        self.abs_pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)
        self.abs_pos_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        # rel->Self-Attention block
        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            new_attn_layer = TimeAwareMultiHeadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate,
                                                            args.device)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

        # RTIM->GRU(x)->F_u
        self.GRU = torch.nn.GRU(input_size=args.hidden_units, hidden_size=args.hidden_units,
                                num_layers=1, batch_first=True)
        self.softmax = torch.nn.Softmax(dim=-1)

        # ATM->Q, K, V's transformation matrices
        self.Q_w = torch.nn.Linear(args.hidden_units, args.hidden_units)
        self.K_w = torch.nn.Linear(args.hidden_units, args.hidden_units)
        self.V_w = torch.nn.Linear(args.hidden_units, args.hidden_units)

        self.dim = args.hidden_units
        
        self.GCN_block = torch.nn.ModuleList()
        for _ in range(args.num_blocks):
            gcn = GCN(self.dim, args.num_heads, args.device)
            self.GCN_block.append(gcn)

        
        self.W_t = torch.nn.Linear(self.abs_head_size, 1)

        self.W_interval = torch.nn.Linear(self.abs_head_size, self.abs_head_size)


        self.f_i = torch.nn.Linear(self.dim, self.dim)
        self.f_r = torch.nn.Linear(self.dim, self.dim)
        self.f_a = torch.nn.Linear(self.dim, self.dim)
    def seq2feats(self, user_ids, log_seqs, year, month, day):
        # item embedding
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs = seqs * self.item_emb.embedding_dim ** 0.5
        seqs = self.item_emb_dropout(seqs)

        # position encoding
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        positions = torch.LongTensor(positions).to(self.dev)
        abs_pos_embs = self.abs_pos_emb(positions)
        abs_pos_embs = self.abs_pos_emb_dropout(abs_pos_embs)

        seqs_pos = seqs + abs_pos_embs
        
        # time embedding
        year_embs = self.year_emb(torch.LongTensor(year).to(self.dev))
        month_embs = self.month_emb(torch.LongTensor(month).to(self.dev))
        day_embs = self.day_emb(torch.LongTensor(day).to(self.dev))

        year_embs = self.year_emb_dropout(year_embs)
        month_embs = self.month_emb_dropout(month_embs)
        day_embs = self.day_emb_dropout(day_embs)
        
        time_embs = year_embs + month_embs + day_embs
        # time_embs = year_embs + day_embs

        # history time
        history_time_embs = time_embs[:, :self.args.maxlen]       # B * maxlen * d
        # target time
        perdiction_time_embs = time_embs[:, 1:self.args.maxlen+1]   # B * maxlen * d

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)  # B * len
        seqs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim

        tl = seqs.shape[1]  # time dim len for enforce causality  # maxlen
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))  # N * N 
        # compute time interval matrix
        src_time_embs = history_time_embs.unsqueeze(1)  # B * 1 * N * dim
        dst_time_embs = history_time_embs.unsqueeze(2)  # B * N * 1 * dim
        time_matrices = src_time_embs - dst_time_embs

        E_rel, item_time_interval = self.ITIM(seqs_pos, timeline_mask, attention_mask, time_matrices)
        # ATM module
        E_abs, _ = self.absolut_time_process(seqs_pos, log_seqs, perdiction_time_embs, attention_mask, timeline_mask.unsqueeze(-1))
        # return self.last_layernorm(E_abs)
        # RTIM module
        
        E_recom = self.RITM_two(perdiction_time_embs, history_time_embs, item_time_interval, attention_mask, E_rel)
        # E_recom = self.RITM(perdiction_time_embs, history_time_embs, item_time_interval, E_abs, attention_mask, seqs)
        # E_recom = self.last_layernorm(E_recom)  # 去掉这个效果会好一点
        # E_rel, _ = self.GRU(E_rel)
        # E_rel = E_rel * ~timeline_mask.unsqueeze(-1)
        # Fusion
        log_feats = E_recom + self.last_layernorm(E_abs) + self.last_layernorm(E_rel)
        
        return log_feats

    def forward(self, user_ids, log_seqs, year, month, day, pos_seqs, neg_seqs): # for training
        log_feats = self.seq2feats(user_ids, log_seqs, year, month, day )

        pos_seqs = torch.LongTensor(pos_seqs).to(self.dev)

        pos_embs = self.item_emb(pos_seqs) # B *  N * d

        neg_seqs = torch.LongTensor(neg_seqs).to(self.dev)
        neg_embs = self.item_emb(neg_seqs)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)


        return pos_logits, neg_logits  # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices, year, month, day): # for inference

        log_feats = self.seq2feats(user_ids, log_seqs, year, month, day,)

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits # preds # (U, I)
    # TODO: ITIM module
    def relative_time_process(self, seqs, timeline_mask, attention_mask, time_matrices):

        for i in range(len(self.attention_layers)):

            Q = self.attention_layernorms[i](seqs)  # PyTorch mha requires time first fmt
            mha_outputs = self.attention_layers[i](Q,
                                                   timeline_mask, attention_mask, time_matrices)
            seqs = Q + mha_outputs

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)

        return log_feats
    # TODO: RTIM module
    def perdiction_time_process(self, per_time_embs, history_time_embs, item_embs, Fu, attention_mask):
        src_time_embs = per_time_embs.unsqueeze(1)  # B * 1 * N * dim
        dst_time_embs = history_time_embs.unsqueeze(2)  # B * N * 1 * dim
        time_embs = (src_time_embs - dst_time_embs).sum(-1)  # B * N * N * d

        paddings = torch.ones(time_embs.shape) * (-2 ** 32 + 1)  # -1e23 # float('-inf')
        paddings = paddings.to(self.dev)
        attn_weights = torch.where(attention_mask, paddings, time_embs)  # enforcing causality

        time_attention = self.softmax(attn_weights)
        time_attention = time_attention * ~attention_mask  # B * N * N

        intent_attention = torch.matmul(Fu, item_embs.permute(0, 2, 1))  # B* N * N

        attn_weights = torch.where(attention_mask, paddings, intent_attention)   # enforcing causality
        intent_attention = self.softmax(attn_weights)

        attention = time_attention * intent_attention
        embs = torch.matmul(attention, item_embs)

        return embs
    # TODO: ATM module
    def absolut_time_process(self, seqs, log_seqs, per_time_embs, attention_mask, timeline_mask):
        train = True
        if log_seqs.shape[0]==1:
            train=False

        # year_embs = torch.zeros(self.year_emb.weight.shape).to(self.dev)
        year_embs = self.year_emb.weight
        month_embs = self.month_emb.weight
        # month_embs = torch.zeros(self.month_emb.weight.shape).to(self.dev)
        day_embs = self.day_emb.weight

        time_embs = torch.cat((year_embs, month_embs, day_embs), dim=0)
        item_time_embs = torch.sparse.mm(self.item_time_matrix, time_embs)
        if train:
            k = item_time_embs[log_seqs]
        else:
            k = item_time_embs[log_seqs].unsqueeze(0)


        Q, K, V = per_time_embs, k, seqs
        # Q, K, V = self.Q_w(per_time_embs), self.K_w(k), self.V_w(seqs)
        Q_ = torch.cat(torch.split(Q, self.abs_head_size, dim=2), dim=0)
        K_ = torch.cat(torch.split(K, self.abs_head_size, dim=2), dim=0)
        V_ = torch.cat(torch.split(V, self.abs_head_size, dim=2), dim=0)

        
        
        attn_weights = Q_.matmul(torch.transpose(K_, 1, 2))
        attn_weights = attn_weights / (K_.shape[-1] ** 0.5)
        paddings = torch.ones(attn_weights.shape) * (-2 ** 32 + 1)  # -1e23 # float('-inf')
        paddings = paddings.to(self.dev)
        attn_mask = attention_mask.unsqueeze(0).expand(attn_weights.shape[0], -1, -1)
        attn_weights = torch.where(attn_mask, paddings, attn_weights)  # enforcing causality
        attn_weights = self.softmax(attn_weights)  # code as below invalids pytorch backward rules
        outputs = attn_weights.matmul(V_)
        outputs = torch.cat(torch.split(outputs, Q.shape[0], dim=0), dim=2)  # div batch_size

        return outputs*~timeline_mask, item_time_embs

    def ITIM(self, seqs, timeline_mask, attention_mask, time_matrices):
        Q = seqs
        for i in range(len(self.GCN_block)):

            temp_Q, interval = self.GCN_block[i](Q, attention_mask, time_matrices)
            Q = Q + temp_Q
            Q *= ~timeline_mask.unsqueeze(-1)

        E_rel = self.last_layernorm(Q)
        return E_rel, interval

    def RITM(self, per_time_embs, history_time_embs, item_time_interval, attention_mask, seqs):
        src_time_embs = per_time_embs.unsqueeze(1)  # B * 1 * N * dim
        dst_time_embs = history_time_embs.unsqueeze(2)  # B * N * 1 * dim
          
        per_time_interval = src_time_embs - dst_time_embs # B * N * N * d
        # 时间间隔注意力
        # time_embs = (src_time_embs - dst_time_embs).sum(-1)
        time_embs = self.W_t(per_time_interval).reshape(src_time_embs.shape[0], src_time_embs.shape[2], src_time_embs.shape[2])
        # time_embs = torch.exp(-torch.sigmoid((src_time_embs - dst_time_embs).sum(-1)))  # B * N * N
        paddings = torch.ones(time_embs.shape) * (-2 ** 32 + 1)  # -1e23 # float('-inf')
        paddings = paddings.to(self.dev)
        attn_weights = torch.where(attention_mask, paddings, time_embs)  # enforcing causality
        time_attention = self.softmax(attn_weights)
        time_attention = time_attention * ~attention_mask  # B * N * N

        # 意图注意力
        # intent_attention = torch.matmul(Fu, item_embs.permute(0, 2, 1))  # B* N * N
        interval_attention = per_time_interval.matmul(self.W_interval(item_time_interval).unsqueeze(-1)).squeeze(-1)
        # intent_attention = intent_attention - interval_attention
        attn_weights = torch.where(attention_mask, paddings, interval_attention)   # enforcing causality
        interval_attention = self.softmax(attn_weights)
        interval_attention = interval_attention * ~attention_mask

        attention = time_attention * interval_attention
        embs = torch.matmul(attention, seqs)
        return embs

    def RITM_two(self, per_time_embs, history_time_embs, item_time_interval, attention_mask, seqs):
        src_time_embs = per_time_embs.unsqueeze(1)  # B * 1 * N * dim
        dst_time_embs = history_time_embs.unsqueeze(2)  # B * N * 1 * dim
          
        per_time_interval_ = src_time_embs - dst_time_embs # B * N * N * d

        per_time_interval = torch.cat(torch.split(per_time_interval_, self.ritm_head_size, dim=3), dim=0)
        # 时间间隔注意力
        # time_embs = (src_time_embs - dst_time_embs).sum(-1)
        time_embs = self.W_t(per_time_interval).reshape(per_time_interval.shape[0], src_time_embs.shape[2], src_time_embs.shape[2])
        # time_embs = torch.exp(-torch.sigmoid((src_time_embs - dst_time_embs).sum(-1)))  # B * N * N
        paddings = torch.ones(time_embs.shape) * (-2 ** 32 + 1)  # -1e23 # float('-inf')
        paddings = paddings.to(self.dev)
        attn_mask = attention_mask.unsqueeze(0).expand(time_embs.shape[0], -1, -1)
        attn_weights = torch.where(attn_mask, paddings, time_embs)  # enforcing causality
        time_attention = self.softmax(attn_weights)
        time_attention = time_attention * ~attention_mask  # B * N * N

        # 意图注意力
        # intent_attention = torch.matmul(Fu, item_embs.permute(0, 2, 1))  # B* N * N
        item_time_interval = torch.cat(torch.split(item_time_interval, self.ritm_head_size, dim=2), dim=0)
        interval_attention = per_time_interval.matmul(self.W_interval(item_time_interval).unsqueeze(-1)).squeeze(-1)
        # intent_attention = intent_attention - interval_attention
        attn_weights = torch.where(attn_mask, paddings, interval_attention)   # enforcing causality
        interval_attention = self.softmax(attn_weights)

        attention = time_attention * interval_attention
        seqs_ = torch.cat(torch.split(seqs, self.ritm_head_size, dim=2), dim=0)
        embs = torch.matmul(attention, seqs_)
        embs = torch.cat(torch.split(embs, seqs.shape[0], dim=0), dim=2)
        return embs
    