binfile = '/data/spey3125/mari970/KoBERT-NER/model_0418/pytorch_model.bin'
with open(binfile, 'rb') as f:
    data = f.read()
    for b in data:
        print(b)