import torch
import torch.nn as nn
import zipfile
import numpy as np

class BaseModel(nn.Module):
    def __init__(self, args, vocab, tag_size):
        super(BaseModel, self).__init__()
        self.args = args
        self.vocab = vocab
        self.tag_size = tag_size

    def save(self, path):
        # Save model
        print(f'Saving model to {path}')
        ckpt = {
            'args': self.args,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(ckpt, path)

    def load(self, path):
        # Load model
        print(f'Loading model from {path}')
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.vocab = ckpt['vocab']
        self.args = ckpt['args']
        self.load_state_dict(ckpt['state_dict'])


def load_embedding(vocab, emb_file, emb_size):
    """
    Read embeddings for words in the vocabulary from the emb_file (e.g., GloVe, FastText).
    Args:
        vocab: (Vocab), a word vocabulary
        emb_file: (string), the path to the embdding file for loading
        emb_size: (int), the embedding size (e.g., 300, 100) depending on emb_file
    Return:
        emb: (np.array), embedding matrix of size (|vocab|, emb_size) 
    """
    scale = 0.05
    emb = np.random.uniform(-scale, scale, size=(len(vocab), emb_size)).astype(np.float32)

    if hasattr(vocab, "pad_id"):
        emb[vocab.pad_id] = 0.0

    with open(emb_file, "r", encoding = "utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            if len(parts) == 2:
                continue

            if len(parts) != emb_size + 1:
                continue
            word = parts[0]

            if word in vocab.word2id:
                vector = np.asarray(parts[1:], dtype=np.float32)
                emb[vocab.word2id[word]] = vector
    return emb

class DanModel(BaseModel):
    def __init__(self, args, vocab, tag_size):
        super(DanModel, self).__init__(args, vocab, tag_size)
        self.define_model_parameters()
        self.init_model_parameters()

        # Use pre-trained word embeddings if emb_file exists
        if args.emb_file is not None:
            self.copy_embedding_from_numpy()

    def define_model_parameters(self):
        """
        Define the model's parameters, e.g., embedding layer, feedforward layer.
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """

        #Embedding layer
        self.emb = nn.Embedding(len(self.vocab), self.args.emb_size, padding_idx=self.vocab.pad_id)
        self.emb_drop = nn.Dropout(self.args.emb_drop)

        #Feedforward layers
        self.ff_layers = nn.ModuleList()
        in_dim = self.args.emb_size
        for _ in range(self.args.hid_layer):
            self.ff_layers.append(nn.Linear(in_dim, self.args.hid_size))
            in_dim = self.args.hid_size

        self.hid_drop = nn.Dropout(self.args.hid_drop)
        self.fc2 = nn.Linear(in_dim, self.tag_size)

        #Activation
        if self.tag_size == 2:
            self.activation = nn.ReLU()
        else:
            self.activation = nn.GELU()

    def init_model_parameters(self):
        """
        Initialize the model's parameters by uniform sampling from a range [-v, v], e.g., v=0.08
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        nn.init.uniform_(self.emb.weight, -0.08, 0.08)
        with torch.no_grad():
            self.emb.weight[self.vocab.pad_id].fill_(0)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def copy_embedding_from_numpy(self):
        """
        Load pre-trained word embeddings from numpy.array to nn.embedding
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        emb_numpy = load_embedding(self.vocab, self.args.emb_file, self.args.emb_size)
        emb_tensor = torch.from_numpy(emb_numpy)

        with torch.no_grad():
            self.emb.weight.copy_(emb_tensor)
            self.emb.weight[self.vocab.pad_id].fill_(0)

    def forward(self, x):
        """
        Compute the unnormalized scores for P(Y|X) before the softmax function.
        E.g., feature: h = f(x)
              scores: scores = w * h + b
              P(Y|X) = softmax(scores)  
        Args:
            x: (torch.LongTensor), [batch_size, seq_length]
        Return:
            scores: (torch.FloatTensor), [batch_size, ntags]
        """
        emb = self.emb(x)
        emb = self.emb_drop(emb)

        mask = (x != self.vocab.pad_id).unsqueeze(-1).to(emb.dtype)
        emb = emb * mask

        avg_emb = emb.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        h = avg_emb
        for layer in self.ff_layers:
            h = self.activation(layer(h))
            h = self.hid_drop(h)

        scores = self.fc2(h)
        return scores