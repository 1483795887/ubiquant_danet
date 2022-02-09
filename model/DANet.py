import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import model.sparsemax as sparsemax

def initialize_glu(module, input_dim, output_dim):
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(input_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    return

class GBN(torch.nn.Module):
    """
    Ghost Batch Normalization
    https://arxiv.org/abs/1705.08741
    """
    def __init__(self, input_dim, virtual_batch_size=512):
        super(GBN, self).__init__()
        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(self.input_dim)

    def forward(self, x):
        if self.training == True:
            chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)
            res = [self.bn(x_) for x_ in chunks]
            return torch.cat(res, dim=0)
        else:
            return self.bn(x)

class LearnableLocality(nn.Module):

    def __init__(self, input_dim, k):
        super(LearnableLocality, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.rand(k, input_dim)))
        self.smax = sparsemax.Entmax15(dim=-1)

    def forward(self, x):
        mask = self.smax(self.weight)
        masked_x = torch.einsum('nd,bd->bnd', mask, x)  # [B, k, D]
        return masked_x

class AbstractLayer(nn.Module):
    def __init__(self, base_input_dim, base_output_dim, k, virtual_batch_size, bias=True):
        super(AbstractLayer, self).__init__()
        self.masker = LearnableLocality(input_dim=base_input_dim, k=k)
        self.fc = nn.Conv1d(base_input_dim * k, 2 * k * base_output_dim, kernel_size=1, groups=k, bias=bias)
        initialize_glu(self.fc, input_dim=base_input_dim * k, output_dim=2 * k * base_output_dim)
        self.bn = GBN(2 * base_output_dim * k, virtual_batch_size)
        self.k = k
        self.base_output_dim = base_output_dim

    def forward(self, x):
        b = x.size(0)
        x = self.masker(x)  # [B, D] -> [B, k, D]
        x = self.fc(x.view(b, -1, 1))  # [B, k, D] -> [B, k * D, 1] -> [B, k * (2 * D'), 1]
        x = self.bn(x)
        chunks = x.chunk(self.k, 1)  # k * [B, 2 * D', 1]
        x = sum([F.relu(torch.sigmoid(x_[:, :self.base_output_dim, :]) * x_[:, self.base_output_dim:, :]) for x_ in chunks])  # k * [B, D', 1] -> [B, D', 1]
        return x.squeeze(-1)


class BasicBlock(nn.Module):
    def __init__(self, input_dim, base_outdim, k, virtual_batch_size, fix_input_dim, drop_rate):
        super(BasicBlock, self).__init__()
        self.conv1 = AbstractLayer(input_dim, base_outdim // 2, k, virtual_batch_size)
        self.conv2 = AbstractLayer(base_outdim // 2, base_outdim, k, virtual_batch_size)

        self.downsample = nn.Sequential(
            nn.Dropout(drop_rate),
            AbstractLayer(fix_input_dim, base_outdim, k, virtual_batch_size)
        )

    def forward(self, x, pre_out=None):
        if pre_out == None:
            pre_out = x
        out = self.conv1(pre_out)
        out = self.conv2(out)
        identity = self.downsample(x)
        out += identity
        return F.leaky_relu(out, 0.01)


class DANet(nn.Module):
    def __init__(self, input_dim, num_classes, layer_num, base_outdim, k,
                 virtual_batch_size, drop_rate=0.1, cat_idxs=[],
                 cat_dims=[], cat_emb_dim=1, ):
        super(DANet, self).__init__()
        params = {'base_outdim': base_outdim, 'k': k, 'virtual_batch_size': virtual_batch_size,
                  'fix_input_dim': input_dim, 'drop_rate': drop_rate}
        self.lay_num = layer_num
        self.layer = nn.ModuleList()

        self.embedder = EmbeddingGenerator(input_dim, cat_dims, cat_idxs,
                                           cat_emb_dim)
        self.post_embed_dim = self.embedder.post_embed_dim

        self.init_layer = BasicBlock(self.post_embed_dim, **params)

        for i in range((layer_num // 2) - 1):
            self.layer.append(BasicBlock(base_outdim, **params))
        self.drop = nn.Dropout(0.1)

        self.fc = nn.Sequential(nn.Linear(base_outdim, 256),
                                nn.ReLU(inplace=True),
                                nn.Linear(256, 512),
                                nn.ReLU(inplace=True),
                                nn.Linear(512, num_classes))

    def forward(self, x):
        x = self.embedder(x)
        out = self.init_layer(x)
        for i in range(len(self.layer)):
            out = self.layer[i](x, out)
        out = self.drop(out)
        out = self.fc(out)
        return out

class EmbeddingGenerator(torch.nn.Module):
    """
    Classical embeddings generator
    """

    def __init__(self, input_dim, cat_dims, cat_idxs, cat_emb_dim):
        """This is an embedding module for an entire set of features

        Parameters
        ----------
        input_dim : int
            Number of features coming as input (number of columns)
        cat_dims : list of int
            Number of modalities for each categorial features
            If the list is empty, no embeddings will be done
        cat_idxs : list of int
            Positional index for each categorical features in inputs
        cat_emb_dim : int or list of int
            Embedding dimension for each categorical features
            If int, the same embedding dimension will be used for all categorical features
        """
        super(EmbeddingGenerator, self).__init__()
        if cat_dims == [] and cat_idxs == []:
            self.skip_embedding = True
            self.post_embed_dim = input_dim
            return
        elif (cat_dims == []) ^ (cat_idxs == []):
            if cat_dims == []:
                msg = "If cat_idxs is non-empty, cat_dims must be defined as a list of same length."
            else:
                msg = "If cat_dims is non-empty, cat_idxs must be defined as a list of same length."
            raise ValueError(msg)
        elif len(cat_dims) != len(cat_idxs):
            msg = "The lists cat_dims and cat_idxs must have the same length."
            raise ValueError(msg)

        self.skip_embedding = False
        if isinstance(cat_emb_dim, int):
            self.cat_emb_dims = [cat_emb_dim] * len(cat_idxs)
        else:
            self.cat_emb_dims = cat_emb_dim

        # check that all embeddings are provided
        if len(self.cat_emb_dims) != len(cat_dims):
            msg = f"""cat_emb_dim and cat_dims must be lists of same length, got {len(self.cat_emb_dims)}
                      and {len(cat_dims)}"""
            raise ValueError(msg)
        self.post_embed_dim = int(
            input_dim + np.sum(self.cat_emb_dims) - len(self.cat_emb_dims)
        )

        self.embeddings = torch.nn.ModuleList()

        # Sort dims by cat_idx
        sorted_idxs = np.argsort(cat_idxs)
        cat_dims = [cat_dims[i] for i in sorted_idxs]
        self.cat_emb_dims = [self.cat_emb_dims[i] for i in sorted_idxs]

        for cat_dim, emb_dim in zip(cat_dims, self.cat_emb_dims):
            self.embeddings.append(torch.nn.Embedding(cat_dim, emb_dim))

        # record continuous indices
        self.continuous_idx = torch.ones(input_dim, dtype=torch.bool)
        self.continuous_idx[cat_idxs] = 0

    def forward(self, x):
        """
        Apply embeddings to inputs
        Inputs should be (batch_size, input_dim)
        Outputs will be of size (batch_size, self.post_embed_dim)
        """
        if self.skip_embedding:
            # no embeddings required
            return x

        cols = []
        cat_feat_counter = 0
        for feat_init_idx, is_continuous in enumerate(self.continuous_idx):
            # Enumerate through continuous idx boolean mask to apply embeddings
            if is_continuous:
                cols.append(x[:, feat_init_idx].float().view(-1, 1))
            else:
                cols.append(
                    self.embeddings[cat_feat_counter](
                        x[:, feat_init_idx].long())
                )
                cat_feat_counter += 1
        # concat
        post_embeddings = torch.cat(cols, dim=1)
        return post_embeddings