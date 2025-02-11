import torch
from torch import nn

from anchor_free import anchor_free_helper
from modules.models import build_base_model


class SelfAttention(nn.Module):
    def __init__(self, input_size=1024, output_size=1024, freq=10000, heads=8, pos_enc=None):
        """ The basic (multi-head) Attention 'cell' containing the learnable parameters of Q, K and V

        :param int input_size: Feature input size of Q, K, V.
        :param int output_size: Feature -hidden- size of Q, K, V.
        :param int freq: The frequency of the sinusoidal positional encoding.
        :param int heads: Number of heads for the attention module.
        :param str | None pos_enc: The type of the positional encoding [supported: Absolute, Relative].
        """
        super(SelfAttention, self).__init__()

        self.permitted_encodings = ["absolute", "relative"]
        if pos_enc is not None:
            pos_enc = pos_enc.lower()
            assert pos_enc in self.permitted_encodings, f"Supported encodings: {*self.permitted_encodings,}"

        self.input_size = input_size
        self.output_size = output_size
        self.heads = heads
        self.pos_enc = pos_enc
        self.freq = freq
        self.Wk, self.Wq, self.Wv = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for _ in range(self.heads):
            self.Wk.append(nn.Linear(in_features=input_size, out_features=output_size//heads, bias=False))
            self.Wq.append(nn.Linear(in_features=input_size, out_features=output_size//heads, bias=False))
            self.Wv.append(nn.Linear(in_features=input_size, out_features=output_size//heads, bias=False))
        self.out = nn.Linear(in_features=output_size, out_features=input_size, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.drop = nn.Dropout(p=0.5)

    def getAbsolutePosition(self, T):
        """Calculate the sinusoidal positional encoding based on the absolute position of each considered frame.
        Based on 'Attention is all you need' paper (https://arxiv.org/abs/1706.03762)

        :param int T: Number of frames contained in Q, K and V
        :return: Tensor with shape [T, T]
        """
        freq = self.freq
        d = self.input_size

        pos = torch.tensor([k for k in range(T)], device=self.out.weight.device)
        i = torch.tensor([k for k in range(T//2)], device=self.out.weight.device)

        # Reshape tensors each pos_k for each i indices
        pos = pos.reshape(pos.shape[0], 1)
        pos = pos.repeat_interleave(i.shape[0], dim=1)
        i = i.repeat(pos.shape[0], 1)

        AP = torch.zeros(T, T, device=self.out.weight.device)
        AP[pos, 2*i] = torch.sin(pos / freq ** ((2 * i) / d))
        AP[pos, 2*i+1] = torch.cos(pos / freq ** ((2 * i) / d))
        return AP

    def getRelativePosition(self, T):
        """Calculate the sinusoidal positional encoding based on the relative position of each considered frame.
        r_pos calculations as here: https://theaisummer.com/positional-embeddings/

        :param int T: Number of frames contained in Q, K and V
        :return: Tensor with shape [T, T]
        """
        freq = self.freq
        d = 2 * T
        min_rpos = -(T - 1)

        i = torch.tensor([k for k in range(T)], device=self.out.weight.device)
        j = torch.tensor([k for k in range(T)], device=self.out.weight.device)

        # Reshape tensors each i for each j indices
        i = i.reshape(i.shape[0], 1)
        i = i.repeat_interleave(i.shape[0], dim=1)
        j = j.repeat(i.shape[0], 1)

        # Calculate the relative positions
        r_pos = j - i - min_rpos

        RP = torch.zeros(T, T, device=self.out.weight.device)
        idx = torch.tensor([k for k in range(T//2)], device=self.out.weight.device)
        RP[:, 2*idx] = torch.sin(r_pos[:, 2*idx] / freq ** ((i[:, 2*idx] + j[:, 2*idx]) / d))
        RP[:, 2*idx+1] = torch.cos(r_pos[:, 2*idx+1] / freq ** ((i[:, 2*idx+1] + j[:, 2*idx+1]) / d))
        return RP

    def forward(self, x, mask=None):
        """ Compute the weighted frame features, based on either the global or local (multi-head) attention mechanism.

        :param torch.tensor x: Frame features with shape [BS, seq, dim]
        
        """
        bs = x.shape[0]
        n = x.shape[1]  # sequence length
        dim = x.shape[2]
        dim_head = dim // self.heads

        outputs = []
        for head in range(self.heads):
            K = self.Wk[head](x)
            Q = self.Wq[head](x)
            V = self.Wv[head](x)

            local_dim = V.shape[-1]

            # Q *= 0.06                       # scale factor VASNet
            # Q /= np.sqrt(self.output_size)  # scale factor (i.e 1 / sqrt(d_k) )
            # energies = torch.matmul(Q, K.transpose(1, 0))
            energies = torch.matmul(Q, K.transpose(1,2))
            # for batch training
            if self.pos_enc is not None:
                if self.pos_enc == "absolute":
                    AP = self.getAbsolutePosition(T=energies.shape[2])
                    energies = energies + AP
                elif self.pos_enc == "relative":
                    RP = self.getRelativePosition(T=energies.shape[0])
                    energies = energies + RP

            if mask is not None:
                mask2 = mask.unsqueeze(-1)
                mask2_t = mask2.transpose(2,1)
                attention_mask = torch.matmul(mask2.float(), mask2_t.float()).bool()
                energies[~attention_mask] = -1e9 #float('-Inf')

            att_weights = self.softmax(energies)
            _att_weights = self.drop(att_weights)

            y = torch.matmul(V.transpose(1,2), _att_weights).transpose(1,2)

            # Save the current head output
            outputs.append(y)
        y = self.out(torch.cat(outputs, dim=2))
        return y, att_weights.clone()  # for now we don't deal with the weights (probably max or avg pooling)


class DSNetAF(nn.Module):
    def __init__(self, base_model, num_feature, num_hidden, num_head):
        super().__init__()
        self.base_model = SelfAttention(input_size=num_feature, output_size=num_feature,
                                       freq=10000, pos_enc="absolute", heads=num_head)
        self.layer_norm = nn.LayerNorm(num_feature)

        self.fc1 = nn.Sequential(
            nn.Linear(num_feature, num_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.LayerNorm(num_hidden)
        )
        self.fc_cls = nn.Linear(num_hidden, 1)
        self.fc_loc = nn.Linear(num_hidden, 2)
        self.fc_ctr = nn.Linear(num_hidden, 1)

    def forward(self, x, mask=None):
        _, seq_len, _ = x.shape

        # bs = x.shape[0]
        # m = x.shape[2] # Feature size 1024
        # x = x.view(bs, -1, m)

        out, _ = self.base_model(x, mask=mask)

        out = out + x
        out = self.layer_norm(out)

        out = self.fc1(out)
        # print("self.fc_cls(out).sigmoid()")
        # print(self.fc_cls(out).sigmoid())
        # print(self.fc_cls(out).sigmoid().shape)
        pred_cls = self.fc_cls(out).sigmoid().view(-1, seq_len, 1)
        # print("pred_cls")
        # print(pred_cls)
        # print(pred_cls.shape)

        # print("self.fc_loc(out).exp()")
        # print(self.fc_loc(out).exp())
        # print(self.fc_loc(out).exp().shape)
        pred_loc = self.fc_loc(out).exp()
        # print("pred_loc")
        # print(pred_loc)
        # print(pred_loc.shape)

        # print("self.fc_ctr(out).sigmoid()")
        # print(self.fc_ctr(out).sigmoid())
        # print(self.fc_ctr(out).sigmoid().shape)
        pred_ctr = self.fc_ctr(out).sigmoid()

        return pred_cls, pred_loc, pred_ctr

    def predict(self, seq):
        pred_cls, pred_loc, pred_ctr = self(seq)

        pred_cls *= pred_ctr
        pred_cls /= pred_cls.max() + 1e-8

        pred_cls = pred_cls.cpu().numpy()
        pred_loc = pred_loc.cpu().numpy()
        
        pred_loc = pred_loc.squeeze(0)
        pred_bboxes = anchor_free_helper.offset2bbox(pred_loc)
        return pred_cls, pred_loc, pred_ctr, pred_bboxes
