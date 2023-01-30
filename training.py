from torch.utils.data import DataLoader
import torch
import sys
from transformer.model import *
from transformer.tools import *

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if use_cuda else "cpu")

    # data_dir = '/data'
    data_dir = sys.argv[1]
    # file_name = 'train'
    file_name = sys.argv[2]
    val_file_name = sys.argv[3]
    # file_ext = ['.x', '.y']
    file_ext = [sys.argv[4], sys.argv[5]]
    train_set = load_dataset(data_dir, file_name, file_ext)

    valid_set = load_dataset(data_dir, val_file_name, file_ext, train_set)

    # batch_size = 64
    batch_size = int(sys.argv[6])

    # get the vocab
    src_vocab = train_set.src_vocab
    trg_vocab = train_set.trg_vocab
    src_vocab_size = src_vocab.__len__()
    trg_vocab_size = trg_vocab.__len__()

    # load data
    print("Creating Data Loader")
    train_data_loader = DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=8)

    valid_data_loader = DataLoader(
        dataset=valid_set, batch_size=batch_size, shuffle=False, num_workers=8)
    print("Done")
    # training pipeline
    epochs = int(sys.argv[7])  # The number of epochs
    best_model = None
    criterion = torch.nn.CrossEntropyLoss()
    model = TransformerModel(src_vocab_size, trg_vocab_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # start training
    print("Start Training")
    for epoch in range(1, epochs + 1):
        train(640, model, train_data_loader, valid_data_loader, optimizer, epoch, criterion, trg_vocab,DEVICE, k=10, clip_rate=0.1)