import torch

def preprocessing(tokenizer, sentence1, sentence2):
    text = '[SEP]'.join([sentence1, sentence2])
    inputs = tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True)
    data = [inputs['input_ids']]
    data = torch.LongTensor(data)
    return data