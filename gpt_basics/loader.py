import torch

def load_data(filename):
    ##method to load the data from the file and return it as a string
    with open(filename,'r') as f:
        data = f.read()
        data = data.split('<|endoftext|>')
        print(f"Number of stories: {len(data)}")
        text = "\n\n".join(data)
        print(f"Length of text: {len(text)} characters")
    return text


def get_batch(train_data, val_data, seq_len, batch_size, device,split='train'):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y