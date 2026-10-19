import re
import json
import glob
import os
import random
import pandas as pd
from modelscope.hub.snapshot_download import snapshot_download

TEMPLATES = [
    "Tell me a story about {topic}.",
    "Write a short story about {topic}.",
    "I want to hear a story about {topic}.",
    "Make up a story where the main character is {topic}.",
    "Can you write a bedtime story about {topic}?",
    "Create a story featuring {topic}.",
    "Write a children's story about {topic}.",
    "Once there was {topic}. Tell me their story.",
]

N_STORIES = 5000
OUTPUT    = 'sft_dataset.json'


def extract_topic(text):
    first_sent = text.split('.')[0]
    names = re.findall(r'\b[A-Z][a-z]+\b', first_sent)
    if names:
        return names[0]
    nouns = re.findall(r'\b(?:a|an)\s+(\w+)\b', first_sent, re.IGNORECASE)
    if nouns:
        return nouns[0]
    return None


def main():
    dataset_dir = snapshot_download(
        'AI-ModelScope/TinyStories',
        repo_type='dataset',
        allow_file_pattern='*train*.parquet',
    )
    files = sorted(glob.glob(os.path.join(dataset_dir, '**/*.parquet'), recursive=True))

    data = []
    for f in files:
        if len(data) >= N_STORIES:
            break
        df = pd.read_parquet(f, columns=['text'])
        for text in df['text']:
            if len(data) >= N_STORIES:
                break
            topic = extract_topic(str(text))
            if not topic:
                continue
            instruction = random.choice(TEMPLATES).format(topic=topic)
            data.append({'instruction': instruction, 'response': str(text)})

    with open(OUTPUT, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'生成 {len(data)} 条指令数据 → {OUTPUT}')


if __name__ == '__main__':
    main()
