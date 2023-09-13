import tensorgrad


class Embedding(tensorgrad.nn.Module):

    def __init__(self, in_features, seq_len, embedding_dim):
        super().__init__()
        self.in_features = in_features
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim

        self.proj = tensorgrad.nn.Linear(self.in_features, self.embedding_dim)
        self.position = tensorgrad.nn.Embedding(
            num_embeddings=self.seq_len,
            embedding_dim=self.embedding_dim,
        )
    
    def forward(self, x):
        x_proj = self.proj(x)
        x_pos = tensorgrad.tensor.arange(x.shape[1], device=x.device)
        x_pos = self.position(x_pos)
        x = x_proj + x_pos
        return x


class Encoder(tensorgrad.nn.Module):
    
    def __init__(self, num_heads, embedding_dim, ffn_dim):
        super().__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.ffn_dim = ffn_dim

        self.attn = tensorgrad.nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=self.num_heads)
        self.ffn = tensorgrad.nn.Sequential(
            tensorgrad.nn.Linear(self.embedding_dim, self.ffn_dim),
            tensorgrad.nn.ReLU(),
            tensorgrad.nn.Linear(self.ffn_dim, self.embedding_dim),
        )
        self.norm = tensorgrad.nn.LayerNorm(self.embedding_dim)
    
    def forward(self, x):
        x_attn = self.attn(x, x, x)
        x = x + x_attn
        x = self.norm(x)
        x_ffn = self.ffn(x)
        x = x + x_ffn
        x = self.norm(x)
        return x


class ViT(tensorgrad.nn.Module):

    def __init__(
        self,
        in_features,
        seq_len,
        embedding_dim,
        ffn_dim=None,
        num_heads=2,
        num_layers=1,
        num_classes=10,
    ):
        super().__init__()
        self.in_features = in_features
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ffn_dim = ffn_dim or self.embedding_dim
        self.num_classes = num_classes

        self.embedding = Embedding(
            in_features=self.in_features,
            seq_len=self.seq_len,
            embedding_dim=self.embedding_dim,
        )
        encoders = [
            Encoder(
                num_heads=self.num_heads,
                embedding_dim=self.embedding_dim,
                ffn_dim=self.ffn_dim,
            )
            for _ in range(self.num_layers)
        ]
        self.encoder = tensorgrad.nn.Sequential(*encoders)
        self.dropout = tensorgrad.nn.Dropout(0.1)
        self.head = tensorgrad.nn.Linear(self.embedding_dim, self.num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        x = self.head(x)
        return x
