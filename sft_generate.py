import json
import torch
import tiktoken
from model import TinyGPT

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"

with open('config.json') as f:
    cfg = json.load(f)


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
        hidden      = cfg['hidden'],
        layers      = cfg['layers'],
        heads       = cfg['heads'],
        context_len = cfg['context_len'],
    ).to(device)
    model.load_state_dict(torch.load('tiny_gpt_sft.pt', map_location=device))
    model.eval()
    print('SFT 模型已加载，输入 q 退出\n')

    while True:
        instruction = input('Instruction: ').strip()
        if instruction.lower() == 'q':
            break
        prompt  = PROMPT_TEMPLATE.format(instruction=instruction)
        ids     = tokenizer.encode(prompt)
        inp     = torch.tensor([ids], device=device)
        out     = model.generate(inp, max_new_tokens=120, temperature=0.8, top_k=40)
        response_ids = out[0].tolist()[len(ids):]
        print(f'\nResponse: {tokenizer.decode(response_ids)}\n')


if __name__ == '__main__':
    main()
