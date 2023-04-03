import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
torch.set_printoptions(threshold=torch.inf)


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.hidden_size = hidden_size
        self.eps = eps

        self.weight = nn.Parameter(torch.Tensor(hidden_size))
        self.bias = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def forward(self, inputs):
        """Layer Normalization.

        This module applies Layer Normalization, with rescaling and shift,
        only on the last dimension. See Lecture 07 (I), slide 23.

        Parameters
        ----------
        inputs (`torch.FloatTensor` of shape `(*dims, hidden_size)`)
            The input tensor. This tensor can have an arbitrary number N of
            dimensions, as long as `inputs.shape[N-1] == hidden_size`. The
            leading N - 1 dimensions `dims` can be arbitrary.

        Returns
        -------
        outputs (`torch.FloatTensor` of shape `(*dims, hidden_size)`)
            The output tensor, having the same shape as `inputs`.
        """
         
        # mean and variance of the last dimension
        mean = inputs.mean(-1, keepdim=True)

        # variance should be biased
        variance = inputs.var(-1, keepdim=True, unbiased=False)

        # normalize the inputs
        normalized_inputs = (inputs - mean) / torch.sqrt(variance + self.eps)

        # scale and shift
        outputs = self.weight * normalized_inputs + self.bias

        return outputs

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)


class MultiHeadedAttention(nn.Module):
    def __init__(self, head_size, num_heads, sequence_length):
        super(MultiHeadedAttention, self).__init__()
        self.head_size = head_size
        self.num_heads = num_heads
        self.sequence_length = sequence_length

        self.linear_q = nn.Linear(num_heads * head_size, num_heads * head_size)
        self.linear_k = nn.Linear(num_heads * head_size, num_heads * head_size)
        self.linear_v = nn.Linear(num_heads * head_size, num_heads * head_size)
        self.linear_o = nn.Linear(num_heads * head_size, num_heads * head_size) 

        self.Q_bias = nn.Parameter(torch.Tensor(num_heads * head_size))
        self.K_bias = nn.Parameter(torch.Tensor(num_heads * head_size))
        self.V_bias = nn.Parameter(torch.Tensor(num_heads * head_size))
        self.O_bias = nn.Parameter(torch.Tensor(num_heads * head_size))

    def get_attention_weights(self, queries, keys, mask=None):
        """Compute the attention weights.

        This computes the attention weights for all the sequences and all the
        heads in the batch. For a single sequence and a single head (for
        simplicity), if Q are the queries (matrix of size `(sequence_length, head_size)`),
        and K are the keys (matrix of size `(sequence_length, head_size)`), then
        the attention weights are computed as

            weights = softmax(Q * K^{T} / sqrt(head_size))

        Here "*" is the matrix multiplication. See Lecture 06, slides 19-24.

        Parameters
        ----------
        queries (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the queries for all the positions in the sequences
            and all the heads.

        keys (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the keys for all the positions in the sequences
            and all the heads. 
           
        mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The masked tensor containing the location of padding in the sequences.

        Returns
        -------
        attention_weights (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, sequence_length)`)
            Tensor containing the attention weights for all the heads and all
            the sequences in the batch.
        """
        # 1. multiply the queries by the keys
        S = torch.matmul(queries, keys.transpose(-1, -2))/math.sqrt(self.head_size)
        # 2. apply the mask
        if mask is not None:
            S = S.masked_fill(mask == 0, -torch.inf)
        # 3. compute the attention weights
        attention_weights = F.softmax(S, dim=-1)
        return attention_weights

        
    def apply_attention(self, queries, keys, values, mask=None):
        """Apply the attention.

        This computes the output of the attention, for all the sequences and
        all the heads in the batch. For a single sequence and a single head
        (for simplicity), if Q are the queries (matrix of size `(sequence_length, head_size)`),
        K are the keys (matrix of size `(sequence_length, head_size)`), and V are
        the values (matrix of size `(sequence_length, head_size)`), then the ouput
        of the attention is given by

            weights = softmax(Q * K^{T} / sqrt(head_size))
            attended_values = weights * V
            outputs = concat(attended_values)

        Here "*" is the matrix multiplication, and "concat" is the operation
        that concatenates the attended values of all the heads (see the
        `merge_heads` function). See Lecture 06, slides 19-24.

        Parameters
        ----------
        queries (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the queries for all the positions in the sequences
            and all the heads. 

        keys (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the keys for all the positions in the sequences
            and all the heads. 

        values (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the values for all the positions in the sequences
            and all the heads. 
        
        mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The masked tensor containing the location of padding in the sequences.
            
        Returns
        -------
        outputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * head_size)`)
            Tensor containing the concatenated outputs of the attention for all
            the sequences in the batch, and all positions in each sequence. 
        """
        print(mask)
        #1. compute the attention weights
        attention_weights = self.get_attention_weights(queries, keys, mask)
        #2. apply the attention weights to the values
        attended_values = torch.matmul(attention_weights, values)
        #3. concatenate the attended values of all the heads
        outputs = self.merge_heads(attended_values)

        return outputs

    def split_heads(self, tensor):
        """Split the head vectors.

        This function splits the head vectors that have been concatenated (e.g.
        through the `merge_heads` function) into a separate dimension. This
        function also transposes the `sequence_length` and `num_heads` axes.
        It only reshapes and transposes the input tensor, and it does not
        apply any further transformation to the tensor. The function `split_heads`
        is the inverse of the function `merge_heads`.

        Parameters
        ----------
        tensor (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * dim)`)
            Input tensor containing the concatenated head vectors (each having
            a size `dim`, which can be arbitrary).

        Returns
        -------
        output (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, dim)`)
            Reshaped and transposed tensor containing the separated head
            vectors. Here `dim` is the same dimension as the one in the
            definition of the input `tensor` above.
        """

        bs, seq_len = tensor.size(0), tensor.size(1)
        tensor = tensor.view(bs, seq_len, self.num_heads, self.head_size)
        tensor = tensor.transpose(1, 2)
        return tensor
        
    def merge_heads(self, tensor):
        """Merge the head vectors.

        This function concatenates the head vectors in a single vector. This
        function also transposes the `sequence_length` and the newly created
        "merged" dimension. It only reshapes and transposes the input tensor,
        and it does not apply any further transformation to the tensor. The
        function `merge_heads` is the inverse of the function `split_heads`.

        Parameters
        ----------
        tensor (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, dim)`)
            Input tensor containing the separated head vectors (each having
            a size `dim`, which can be arbitrary).

        Returns
        -------
        output (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * dim)`)
            Reshaped and transposed tensor containing the concatenated head
            vectors. Here `dim` is the same dimension as the one in the
            definition of the input `tensor` above.
        """

        tensor = tensor.transpose(1, 2)
        bs, seq_len = tensor.size(0), tensor.size(2)
        tensor = tensor.contiguous().view(bs, seq_len, self.num_heads * self.head_size)
        return tensor

    def forward(self, hidden_states, mask=None):
        """Multi-headed attention.

        This applies the multi-headed attention on the input tensors `hidden_states`.
        For a single sequence (for simplicity), if X are the hidden states from
        the previous layer (a matrix of size `(sequence_length, num_heads * head_size)`
        containing the concatenated head vectors), then the output of multi-headed
        attention is given by

            Q = X * W_{Q} + b_{Q}        # Queries
            K = X * W_{K} + b_{K}        # Keys
            V = X * W_{V} + b_{V}        # Values

            Y = attention(Q, K, V)       # Attended values (concatenated for all heads)
            outputs = Y * W_{Y} + b_{Y}  # Linear projection

        Here "*" is the matrix multiplication.

        Parameters
        ----------
        hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * head_size)`)
            Input tensor containing the concatenated head vectors for all the
            sequences in the batch, and all positions in each sequence. This
            is, for example, the tensor returned by the previous layer.
            
        
        mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The masked tensor containing the location of padding in the sequences.

        Returns
        -------
        output (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * head_size)`)
            Tensor containing the output of multi-headed attention for all the
            sequences in the batch, and all positions in each sequence.
        """

        print(mask)
        #1. apply linear transformation to the input tensor
        queries = self.linear_q(hidden_states) + self.Q_bias
        keys = self.linear_k(hidden_states) + self.K_bias
        values = self.linear_v(hidden_states) + self.V_bias
        #2. split the head vectors
        queries = self.split_heads(queries)
        keys = self.split_heads(keys)
        values = self.split_heads(values)
        #3. apply the attention mechanism
        outputs = self.apply_attention(queries, keys, values, mask) 
        #4. W_o * Y + b_o
        outputs = self.linear_o(outputs) + self.O_bias 

class PostNormAttentionBlock(nn.Module):
    
    def __init__(self, embed_dim, hidden_dim, num_heads, sequence_length, dropout=0.30):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network 
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()
        
        
        self.layer_norm_1 = LayerNorm(embed_dim)
        self.attn = MultiHeadedAttention(embed_dim//num_heads, num_heads,sequence_length)
        self.layer_norm_2 = LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        
    def forward(self, x, mask=None):
       
        attention_outputs = self.attn(x, mask)
        attention_outputs = self.layer_norm_1(x + attention_outputs)
        outputs = self.linear(attention_outputs)

        outputs = self.layer_norm_2(outputs + attention_outputs)
        return outputs

class PreNormAttentionBlock(nn.Module):
    
    def __init__(self, embed_dim, hidden_dim, num_heads,sequence_length, dropout=0.0):
        """A decoder layer.

        This module combines a Multi-headed Attention module and an MLP to
        create a layer of the transformer, with normalization and skip-connections.
        See Lecture 06, slide 33.

        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network 
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            sequence_length - Length of the sequence
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()
        
        self.layer_norm_1 = LayerNorm(embed_dim)
        self.attn = MultiHeadedAttention(embed_dim//num_heads, num_heads,sequence_length)
        self.layer_norm_2 = LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        
    def forward(self, x, mask=None):
        x1_norm = self.layer_norm_1(x)
        x1_attention_outputs = self.attn(x1_norm, mask)
        x2 = x + x1_attention_outputs
        x2_norm = self.layer_norm_2(x2)
        x2_linear_outputs = self.linear(x2_norm)
        outputs = x2 + x2_linear_outputs
        return outputs


class Transformer(nn.Module):
    
    def __init__(self, vocabulary_size=30522, embed_dim=256, hidden_dim=256, num_heads=1,
            num_layers=2, block='prenorm', dropout=0.3):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            block - Type of attention block
            dropout - Amount of dropout to apply in the feed-forward network and 
                      on the input encoding
        """
        super().__init__()
        
        #Adding the cls token to the sequnence 
        self.sequence_length= 1 + 256
        # Layers/Networks
        self.embedding = nn.Embedding(vocabulary_size, embed_dim)
        if block =='prenorm':
          self.transformer = nn.ModuleList([PreNormAttentionBlock(embed_dim, hidden_dim, num_heads,self.sequence_length, dropout=dropout) for _ in range(num_layers)])
        else:
          self.transformer = nn.ModuleList([PostNormAttentionBlock(embed_dim, hidden_dim, num_heads,self.sequence_length, dropout=dropout) for _ in range(num_layers)])
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
        )
        self.dropout = nn.Dropout(dropout)
        
        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1,self.sequence_length,embed_dim))
   
    def forward(self, x, mask=None):
        """Transformer

        This is a small version of  Transformer

        Parameters
        ----------
        x - (`torch.LongTensor` of shape `(batch_size, sequence length)`)
            The input tensor containing text.
        
        mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The masked tensor containing the location of padding in the sequences.

        Returns
        -------
        output (`torch.FloatTensor` of shape `(batch_size, embed_dim)`)
            A tensor containing the output from the mlp_head.
        """
        # Preprocess input
        
        x = self.embedding(x)
        B, T, _ = x.shape

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:,:T+1]
        #Add dropout and then the transformer
        x = self.dropout(x)
        for layer in self.transformer:
            x = layer(x, mask)
        
        #Take the cls token representation and send it to mlp_head
        x = self.mlp_head(x[:,0])
        return x
         
 
