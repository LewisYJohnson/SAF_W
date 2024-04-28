import torch
import torch.nn as nn
# from utils.general_util import convert_masks


class PointNetAttention(nn.Module):
    """
        Attention layer
    """

    def __init__(self, hidden_dimensions, first_base, **kwargs):
        super(PointNetAttention, self).__init__()
        self.W1 = nn.Linear(first_base * hidden_dimensions, hidden_dimensions, bias=False)
        self.W2 = nn.Linear(hidden_dimensions, hidden_dimensions, bias=False)
        self.V = nn.Linear(hidden_dimensions, 1, bias=False)

    def forward(self, encoder_outputs, dec_output, mask=None):
        w1_e = self.W1(encoder_outputs)  # encoder_outputs shape=(batch_size, 句子长度/时间跨度, 词向量维度)，这里转换最后一个维度  # 输入2, 20, 42, 1536
        w2_d = self.W2(dec_output)  # dec_output shape=(batch_size, 句子长度/时间跨度, 词向量维度)，这里转换最后一个维度  # 输入2, 20, 42, 768
        tanh_output = torch.tanh(w1_e + w2_d)  # 将编码器和解码器信息组合，再使用双曲正切激活，形状不变
        v_dot_tanh = self.V(tanh_output)  # v_dot_tanh shape=(batch_size, 句子长度/时间跨度, 1)，因为对词向量进行选择，因此这里必须控制最后一个维度是1
        if mask is not None:
            mask = torch.clamp(mask, 0, 1)  # 限制值在0-1之间
            mask = mask.unsqueeze(-1)
            v_dot_tanh += (mask * -1e9)
        attention_weights = v_dot_tanh.softmax(1)  # attention_weights shape=(batch_size, 句子长度/时间跨度, 1) 三维张量时softmax给第二个维度
        attention_weights = attention_weights.squeeze(-1)
        return attention_weights


class Decoder(nn.Module):
    """
        Decoder class for PointerLayer
    """

    def __init__(self, hidden_dimensions, decoder_bert, **kwargs):
        super(Decoder, self).__init__()
        # self.lstm = nn.LSTM(hidden_dimensions, hidden_dimensions, batch_first=True)
        self.decoder = decoder_bert

    def forward(self, x, last_hidden):
        # dec_output, (state_h, state_c) = self.lstm(x, hidden_states)  # 这里的x是(batch_size, 1, 词向量维度)
        conbine_hidden = torch.cat([last_hidden, x], 1)
        # output_shape = (x.shape[0], x.shape[1], x.shape[3])
        # conbine_hidden = conbine_hidden.reshape(x.shape[0] * x.shape[1], 2, x.shape[3])
        dec_output = self.decoder(conbine_hidden)['last_hidden_state'][:, -1, :]
        # return dec_output, [state_h, state_c]  # 返回三个(batch_size, 词向量维度)向量
        # dec_output = dec_output.reshape(output_shape)
        return dec_output  # 返回三个(batch_size, 词向量维度)向量

    def get_initial_state(self, inputs):
        return None


class MTVR_for_Entity(nn.Module):
    def __init__(self, basic_dimensions, hidden_dimensions, decoder_bert, length_info=7):
        super(MTVR_for_Entity, self).__init__()
        self.attention = PointNetAttention(hidden_dimensions, first_base=1)
        self.decoder = Decoder(hidden_dimensions, decoder_bert, name="decoder")
        self.length_info = length_info
        self.seq_dense = nn.Linear(basic_dimensions, hidden_dimensions)

    def forward(self, x, pooled=None, context_masks=None, entity_masks=None):
        """
        使用pooled作为基础指针，同时生成一个全0的向量用来记录这样的信息，在指针网络中进行循环。
        :param x: 句子经过编码之后的样本 2 105 42 768 b e s h
        :param pooled: 经过最大池化之后的结果，用于指定第一个指针，这个池化方案也可以替换为平均池化 b e h 2 106 768
        :param context_masks: 掩码，记录文本长度的掩码，防止选错区域

        """
        # pooled = convert_masks(pooled, from_value=-1e30, to_value=0)
        pre = torch.zeros_like(pooled).to(x.device)
        cur = pooled
        context_masks = 1 - context_masks
        # context_masks = context_masks.unsqueeze(1).repeat(1, x.shape[1], 1)  # 如果是三维的话无需扩展
        one_hot_base_iteration = torch.zeros_like(context_masks).to(x.device).float()
        # one_hot_base_return = torch.zeros_like(self.mask_input).to(self.src_seq.device).float()
        real_prob_base_return = torch.zeros_like(context_masks).to(x.device).float()
        one_hot_base_return_container = []
        for i in range(self.length_info):
            pre, cur, one_hot_base_iteration, one_hot_base_return, real_prob_base_return = \
                self.step(pre, cur, one_hot_base_iteration, real_prob_base_return, x, context_masks)
            one_hot_base_return_container.append(one_hot_base_return)
        one_hot_outputs = torch.max(torch.cat([x.unsqueeze(1) for x in one_hot_base_return_container], 1), -1)[1]  # 2->1,
        real_prob = real_prob_base_return.unsqueeze(-1)
        selected_seq = real_prob * x
        return selected_seq, one_hot_outputs

    def step(self, pre, cur, prob_iter, real_prob_base_return, x, context_masks):
        h = self.decoder(cur.unsqueeze(1), pre.unsqueeze(1))  # 三维张量时由2改1
        dec_seq = h.unsqueeze(1).repeat(1, x.shape[1], 1)
        probs = self.attention(dec_seq, x, mask=context_masks + prob_iter)
        # 增加生成过程
        probs_zeros = torch.zeros_like(probs).to(x.device)
        kth = torch.max(probs, dim=-1)[1].unsqueeze(-1)
        probs_zeros.scatter_(dim=-1, index=kth, value=1)  # one-hot扩张向量
        # 对比方案1: 极化softmax(可求导并增强了错误结果的修正幅度,因此称为极化)
        selected_seq = x * (probs + probs_zeros).unsqueeze(-1)  # 总会有某一组向量较大
        selected_seq = selected_seq.max(dim=1)[0]  # 2->1
        current_prob = (probs + probs_zeros) + real_prob_base_return
        one_hot_outputs = torch.ge(probs, probs_zeros.float()).reshape(probs_zeros.shape).float()
        one_hot_outputs = torch.ones_like(one_hot_outputs) - one_hot_outputs

        return h, selected_seq, one_hot_outputs + prob_iter, one_hot_outputs, current_prob
