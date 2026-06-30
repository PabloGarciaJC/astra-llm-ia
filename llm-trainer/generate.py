import os
import sys
import torch
import torch.nn as nn
from torch.nn import functional as F

MODEL_PATH  = 'output/model.pt'
GEN_LEN     = int(os.getenv('GEN_LEN', 300))
TEMPERATURE = float(os.getenv('TEMPERATURE', 0.8))

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ==============================
# ARQUITECTURA (debe coincidir con train.py)
# ==============================
class Head(nn.Module):
    def __init__(self, n_embed, head_size, block_size, dropout):
        super().__init__()
        self.key   = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        att = (q @ k.transpose(-2, -1)) * (C ** -0.5)
        att = att.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        return att @ self.value(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embed, num_heads, head_size, block_size, dropout):
        super().__init__()
        self.heads   = nn.ModuleList([Head(n_embed, head_size, block_size, dropout) for _ in range(num_heads)])
        self.proj    = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))


class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embed // n_head
        self.att = MultiHeadAttention(n_embed, n_head, head_size, block_size, dropout)
        self.ff  = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.att(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class LLM(nn.Module):
    def __init__(self, vocab_size, n_embed, n_head, n_layer, block_size, dropout):
        super().__init__()
        self.block_size = block_size
        self.token_emb  = nn.Embedding(vocab_size, n_embed)
        self.pos_emb    = nn.Embedding(block_size, n_embed)
        self.blocks     = nn.Sequential(*[Block(n_embed, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln         = nn.LayerNorm(n_embed)
        self.head       = nn.Linear(n_embed, vocab_size)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            B, T = idx_cond.shape
            tok = self.token_emb(idx_cond)
            pos = self.pos_emb(torch.arange(T, device=idx.device))
            x   = self.ln(self.blocks(tok + pos))
            logits = self.head(x)[:, -1, :] / temperature
            probs  = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx


# ==============================
# GENERACIÓN
# ==============================
def main():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] No se encontró el modelo en: {MODEL_PATH}")
        print("  Entrena primero con: make train")
        sys.exit(1)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    cfg        = checkpoint['config']
    stoi       = checkpoint['stoi']
    itos       = checkpoint['itos']
    vocab_size = checkpoint['vocab_size']

    model = LLM(
        vocab_size  = vocab_size,
        n_embed     = cfg['n_embed'],
        n_head      = cfg['n_head'],
        n_layer     = cfg['n_layer'],
        block_size  = cfg['block_size'],
        dropout     = cfg['dropout'],
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    prompt = os.getenv('PROMPT', '')
    if not prompt:
        print("Escribe un prompt para generar texto (Enter para empezar desde cero):")
        prompt = input("> ").strip()

    print("\n" + "=" * 50)

    if prompt:
        valid_chars = [c for c in prompt if c in stoi]
        if not valid_chars:
            print("[WARN] El prompt contiene caracteres fuera del vocabulario. Empezando desde cero.")
            context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
        else:
            encoded = [stoi[c] for c in valid_chars]
            context = torch.tensor([encoded], dtype=torch.long, device=DEVICE)
        print(f"Prompt: {prompt}\n")
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)

    generated = model.generate(context, max_new_tokens=GEN_LEN, temperature=TEMPERATURE)
    text = ''.join(itos[t] for t in generated[0].tolist())
    print(text)
    print("=" * 50)


if __name__ == '__main__':
    main()
