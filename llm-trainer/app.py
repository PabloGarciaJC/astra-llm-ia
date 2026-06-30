import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from flask import Flask, request, jsonify, render_template

MODEL_PATH = 'output/model.pt'
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

app = Flask(__name__)

# ==============================
# ARQUITECTURA
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
# CARGAR MODELO AL ARRANCAR
# ==============================
model     = None
stoi      = None
itos      = None

def load_model():
    global model, stoi, itos
    if not os.path.exists(MODEL_PATH):
        return False
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
    return True


# ==============================
# RUTAS
# ==============================
@app.after_request
def no_cache(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    return response


@app.route('/')
def index():
    model_ready = model is not None
    return render_template('index.html', model_ready=model_ready)


@app.route('/generate', methods=['POST'])
def generate():
    if model is None:
        return jsonify({'error': 'Modelo no cargado. Ejecuta make train primero.'}), 503

    data        = request.get_json()
    prompt      = data.get('prompt', '').strip()
    gen_len     = int(data.get('gen_len', 300))
    temperature = float(data.get('temperature', 0.8))

    if prompt:
        valid_chars = [c for c in prompt if c in stoi]
        if valid_chars:
            encoded = [stoi[c] for c in valid_chars]
            context = torch.tensor([encoded], dtype=torch.long, device=DEVICE)
        else:
            context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)

    generated = model.generate(context, max_new_tokens=gen_len, temperature=temperature)
    text = ''.join(itos[t] for t in generated[0].tolist())

    return jsonify({'text': text})


KILL_SW_JS = """
self.addEventListener('install', (e) => {
  e.waitUntil(self.skipWaiting());
});
self.addEventListener('activate', (e) => {
  e.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(keys.map(k => caches.delete(k)));
    await self.clients.claim();
    const clients = await self.clients.matchAll({ type: 'window', includeUncontrolled: true });
    clients.forEach(c => c.navigate(c.url));
    await self.registration.unregister();
  })());
});
self.addEventListener('fetch', (e) => {
  e.respondWith(fetch(e.request).catch(() => new Response('', { status: 503 })));
});
"""

UNREGISTER_AND_RELOAD_JS = """
(async () => {
  if ('serviceWorker' in navigator) {
    const regs = await navigator.serviceWorker.getRegistrations();
    await Promise.all(regs.map(r => r.unregister()));
  }
  if ('caches' in window) {
    const keys = await caches.keys();
    await Promise.all(keys.map(k => caches.delete(k)));
  }
  window.location.replace('/');
})();
"""

UNREGISTER_HTML = """<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>Limpiando caché...</title>
<style>body{{background:#0f1117;color:#e2e8f0;font-family:sans-serif;
display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0;}}
.box{{text-align:center;}}.box p{{color:#64748b;margin-top:.5rem;}}</style>
</head><body><div class="box"><h2>Limpiando caché...</h2>
<p>Serás redirigido en un momento.</p></div>
<script>{script}</script></body></html>""".format(script=UNREGISTER_AND_RELOAD_JS)


@app.route('/service-worker.js')
@app.route('/sw.js')
def service_worker():
    from flask import Response
    resp = Response(KILL_SW_JS, mimetype='application/javascript')
    resp.headers['Service-Worker-Allowed'] = '/'
    resp.headers['Cache-Control'] = 'no-store'
    return resp


@app.route('/error')
@app.route('/error/<path:subpath>')
def error_page(subpath=''):
    from flask import Response
    return Response(UNREGISTER_HTML, status=200, mimetype='text/html')


@app.route('/static/loader.js')
def loader_js():
    from flask import Response
    resp = Response(UNREGISTER_AND_RELOAD_JS, mimetype='application/javascript')
    resp.headers['Cache-Control'] = 'no-store'
    return resp


@app.route('/_app/version.json')
def app_version():
    from flask import Response
    resp = Response('{}', mimetype='application/json')
    resp.headers['Clear-Site-Data'] = '"cache", "storage"'
    return resp


@app.route('/_app/<path:subpath>')
def app_assets(subpath):
    from flask import Response
    resp = Response('', status=404)
    resp.headers['Clear-Site-Data'] = '"cache", "storage"'
    return resp


@app.route('/status')
def status():
    return jsonify({'model_loaded': model is not None, 'device': DEVICE})


if __name__ == '__main__':
    print("Cargando modelo...")
    if load_model():
        print("Modelo cargado. Servidor en http://localhost:8080")
    else:
        print("AVISO: No se encontró model.pt — ejecuta 'make train' primero.")
    app.run(host='0.0.0.0', port=8080, debug=False)
