import json
import torch
import torch.nn.functional as F
import tiktoken

from model import TinyGPT

with open('config.json') as f:
    cfg = json.load(f)

HIDDEN      = cfg['hidden']
LAYERS      = cfg['layers']
HEADS       = cfg['heads']
CONTEXT_LEN = cfg['context_len']
CHECKPOINT  = cfg['checkpoint']


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def main():
    device    = get_device()
    tokenizer = tiktoken.get_encoding('gpt2')

    model = TinyGPT(
        vocab_size  = tokenizer.n_vocab,
        hidden      = HIDDEN,
        layers      = LAYERS,
        heads       = HEADS,
        context_len = CONTEXT_LEN,
    ).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()
    print(f'模型已加载：{CHECKPOINT}')

    prompts = [
        'Once upon a time',
        'The little girl',
        'One sunny day',
        'Two thousand years ago'
    ]

    for prompt in prompts:
        ids = tokenizer.encode(prompt)
        inp = torch.tensor([ids], device=device)
        out = model.generate(inp, max_new_tokens=80, temperature=0.8, top_k=40)
        print(f'\nPrompt: {prompt}')
        print(tokenizer.decode(out[0].tolist()))
        print('-' * 60)


if __name__ == '__main__':
    main()
