import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_printoptions(threshold=torch.inf)

class GRU(nn.Module):
    def __init__(
            self,
            input_size, 
            hidden_size
            ):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.w_ir = nn.Parameter(torch.empty(hidden_size, input_size))
        self.w_iz = nn.Parameter(torch.empty(hidden_size, input_size))
        self.w_in = nn.Parameter(torch.empty(hidden_size, input_size))

        self.b_ir = nn.Parameter(torch.empty(hidden_size))
        self.b_iz = nn.Parameter(torch.empty(hidden_size))
        self.b_in = nn.Parameter(torch.empty(hidden_size))

        self.w_hr = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.w_hz = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.w_hn = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.b_hr = nn.Parameter(torch.empty(hidden_size))
        self.b_hz = nn.Parameter(torch.empty(hidden_size))
        self.b_hn = nn.Parameter(torch.empty(hidden_size))
        for param in self.parameters():
            nn.init.uniform_(param, a=-(1/hidden_size)**0.5, b=(1/hidden_size)**0.5)

    def GRUcell(self, inputs, hidden_states):

        r = torch.sigmoid( 
            torch.matmul(inputs, self.w_ir.t()) + self.b_ir +
            torch.matmul(hidden_states, self.w_hr.t()) + self.b_hr
        )   

        z = torch.sigmoid(
            torch.matmul(inputs, self.w_iz.t()) + self.b_iz +
            torch.matmul(hidden_states, self.w_hz.t()) + self.b_hz
        )

        n = torch.tanh(
            torch.matmul(inputs, self.w_in.t()) + self.b_in +
            r * (torch.matmul(hidden_states, self.w_hn.t()) + self.b_hn)
        )

        # run on GPU
        # on CPU using torch.float64.
        # Note however that your implementation must also work with torch.float32 inputs and on
        # a CUDA enabled device.
        if z.device.type == 'cuda': 
            one = torch.tensor(1, dtype=torch.float32).cuda()
        else:
            one = torch.tensor(1, dtype=torch.float64)

        hidden_states = (one - z) * n + z * hidden_states

        return hidden_states

    def forward(self, inputs, hidden_states):
            """GRU.
            
            This is a Gated Recurrent Unit that implements the following equations:
            \begin{array}{ll}
            r = \sigma(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\
            z = \sigma(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) \\
            n = \tanh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn})) \\
            h' = (1 - z) * n + z * h
            \end{array}

            Parameters
            ----------
            inputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, input_size)`)
            The input tensor containing the embedded sequences. input_size corresponds to embedding size.
            
            hidden_states (`torch.FloatTensor` of shape `(1, batch_size, hidden_size)`)
            The (initial) hidden state.
            
            Returns
            -------
            outputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`)
            A feature tensor encoding the input sentence. 
            
            hidden_states (`torch.FloatTensor` of shape `(1, batch_size, hidden_size)`)
            The final hidden state. 
            """

            outputs = []
            hidden_states = hidden_states.squeeze(0)
            for i in range(inputs.shape[1]):
                hidden_states = self.GRUcell(inputs[:, i, :], hidden_states)
                outputs.append(hidden_states)
            
            outputs = torch.stack(outputs, dim=1)
            hidden_states = hidden_states.unsqueeze(0)

            return outputs, hidden_states


         
        


class Attn(nn.Module):
    def __init__(
        self,
        hidden_size=256,
        dropout=0.0 # note, this is an extrenous argument
        ):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size

        self.W = nn.Linear(hidden_size*2, hidden_size)
        self.V = nn.Linear(hidden_size, hidden_size) # in the forwards, after multiplying
                                                     # do a torch.sum(..., keepdim=True), its a linear operation

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, inputs, hidden_states, mask = None):
        """Soft Attention mechanism.

        This is a one layer MLP network that implements Soft (i.e. Bahdanau) Attention with masking
        Parameters
        ----------
        inputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`)
            The input tensor containing the embedded sequences.

        hidden_states (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)
            The (initial) hidden state.

        mask ( optional `torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The masked tensor containing the location of padding in the sequences.

        Returns
        -------
        outputs (`torch.FloatTensor` of shape `(batch_size,  hidden_size)`)
        yuchen: I removed sequence_length,
            A feature tensor encoding the input sentence with attention applied.

        x_attn (`torch.FloatTensor` of shape `(batch_size, sequence_length, 1)`)
            The attention vector.
        """
        # exchange 1 and 0 within mask 
        mask = 1 - mask

        encooder_outputs = inputs
        decoder_hidden = hidden_states[-1].unsqueeze(1).repeat(1, encooder_outputs.shape[1], 1)

        # compute attention weights
        x = self.tanh(self.W(torch.cat((encooder_outputs, decoder_hidden), dim=2)))
        x = self.V(x)
        x = torch.sum(x, dim=2, keepdim=True)

        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(2), -torch.inf)
        
        x_attn = self.softmax(x)


        outputs = torch.sum(encooder_outputs * x_attn, dim=1)
        print(outputs)
        print(x_attn)

        return outputs, x_attn



class Encoder(nn.Module):
    def __init__(
        self,
        vocabulary_size=30522,
        embedding_size=256,
        hidden_size=256,
        num_layers=1,
        dropout=0.0
        ):
        super(Encoder, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(
            vocabulary_size, embedding_size, padding_idx=0,
        )

        self.dropout = nn.Dropout(p=dropout)
        self.rnn = nn.GRU(
            embedding_size, hidden_size, num_layers=num_layers, bidirectional=True,
            batch_first=True
        )

    def forward(self, inputs, hidden_states):
        """GRU Encoder.

        This is a Bidirectional Gated Recurrent Unit Encoder network
        Parameters
        ----------
        inputs (`torch.FloatTensor` of shape `(batch_size, sequence_length)`)
            The input tensor containing the token sequences.

        hidden_states(`torch.FloatTensor` of shape `(num_layers*2, batch_size, hidden_size)`)
            The (initial) hidden state for the bidrectional GRU.
            
        Returns
        -------
        outputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`)
            A feature tensor encoding the input sentence. 

        hidden_states (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)
            The final hidden state. 
        """
        #1. retrieve the embedding of the inputs
        #2. apply dropout to the embeddings
        #3. apply the GRU to the embeddings
        #4. return the outputs and the hidden states
        # ==========================

        embeddings = self.embedding(inputs)
        embeddings = self.dropout(embeddings)
        outputs, hidden_states = self.rnn(embeddings, hidden_states)
        # sum bidirectional outputs
        outputs = (outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:])

        # sum bidirectional hidden states
        hidden_states = (hidden_states[:self.num_layers] + hidden_states[self.num_layers:])
        return outputs, hidden_states

    def initial_states(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device
        shape = (self.num_layers*2, batch_size, self.hidden_size)
        # The initial state is a constant here, and is not a learnable parameter
        h_0 = torch.zeros(shape, dtype=torch.float, device=device)
        return h_0

class DecoderAttn(nn.Module):
    def __init__(
        self,
        vocabulary_size=30522,
        embedding_size=256,
        hidden_size=256,
        num_layers=1,
        dropout=0.0, 
        ):

        super(DecoderAttn, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p=dropout)

        self.rnn = nn.GRU(
            hidden_size, hidden_size, num_layers=num_layers, batch_first=True 
        )
        
        self.mlp_attn = Attn(hidden_size, dropout)

    def forward(self, inputs, hidden_states, mask=None):
        """GRU Decoder network with Soft attention

        This is a Unidirectional Gated Recurrent Unit Decoder network
        Parameters
        ----------
        inputs (`torch.LongTensor` of shape `(batch_size, sequence_length, hidden_size)`)
            The input tensor containing the encoded input sequence.

        hidden_states(`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)
            The (initial) hidden state for the unidrectional GRU.

        mask ( optional `torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The masked tensor containing the location of padding in the sequences.

        Returns
        -------
        outputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`)
            A feature tensor encoding the input sentence. 

        hidden_states (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)
            The final hidden state. 
        """

        encoder_outputs = inputs
        decoder_hidden = hidden_states 
        # dropout on encoder outputs
        encoder_outputs = self.dropout(encoder_outputs)
        # context vector
        context_vector, _ = self.mlp_attn(encoder_outputs, decoder_hidden, mask)
        # fed to decoder
        outputs, hidden_states = self.rnn(context_vector, decoder_hidden)

        return outputs, hidden_states

        
        
class EncoderDecoder(nn.Module):

    def __init__(
        self,
        vocabulary_size=30522,
        embedding_size=256,
        hidden_size=256,
        num_layers=1,
        dropout = 0.0,
        encoder_only=False
        ):
        super(EncoderDecoder, self).__init__()
        self.encoder_only = encoder_only
        self.encoder = Encoder(vocabulary_size, embedding_size, hidden_size,
                num_layers, dropout=dropout)
        if not encoder_only:
          self.decoder = DecoderAttn(vocabulary_size, embedding_size, hidden_size, num_layers, dropout=dropout)
        
    def forward(self, inputs, mask=None):
        """GRU Encoder-Decoder network with Soft attention.

        This is a Gated Recurrent Unit network for Sentiment Analysis. This
        module returns a decoded feature for classification. 
        Parameters
        ----------
        inputs (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The input tensor containing the token sequences.

        mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The masked tensor containing the location of padding in the sequences.

        Returns
        -------
        x (`torch.FloatTensor` of shape `(batch_size, hidden_size)`)
            A feature tensor encoding the input sentence. 

        hidden_states (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)
            The final hidden state. 
        """
        # print mask if mask is not None
        print(mask)
        hidden_states = self.encoder.initial_states(inputs.shape[0])
        x, hidden_states = self.encoder(inputs, hidden_states)
        if self.encoder_only:
          x = x[:, 0]
          return x, hidden_states
        x, hidden_states = self.decoder(x, hidden_states, mask)
        x = x[:, 0]
        return x, hidden_states
