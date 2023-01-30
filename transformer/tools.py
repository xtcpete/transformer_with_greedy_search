import os
from tqdm import tqdm

import torch
from torch.utils.data import Dataset


class Vocabulary:
    """
    Initialize vocabulary class for text data
    """
    def __init__(self, pad_token="<pad>", unk_token='<unk>', eos_token='<eos>', sos_token='<sos>'):
        self.id_to_string = {}
        self.string_to_id = {}

        # add the default pad token
        self.id_to_string[0] = pad_token
        self.string_to_id[pad_token] = 0

        # add the default unknown token
        self.id_to_string[1] = unk_token
        self.string_to_id[unk_token] = 1

        # add the default unknown token
        self.id_to_string[2] = eos_token
        self.string_to_id[eos_token] = 2

        # add the default unknown token
        self.id_to_string[3] = sos_token
        self.string_to_id[sos_token] = 3

        # shortcut access
        self.pad_id = 0
        self.unk_id = 1
        self.eos_id = 2
        self.sos_id = 3

    def __len__(self):
        return len(self.id_to_string)

    def add_new_word(self, string):
        self.string_to_id[string] = len(self.string_to_id)
        self.id_to_string[len(self.id_to_string)] = string

    # Given a string, return ID
    # if extend_vocab is True, add the new word
    def get_idx(self, string, extend_vocab=False):
        if string in self.string_to_id:
            return self.string_to_id[string]
        elif extend_vocab:  # add the new word
            self.add_new_word(string)
            return self.string_to_id[string]
        else:
            return self.unk_id


# Read the raw txt file and generate a 1D pytorch tensor
# containing the whole text mapped to sequence of token ID,
# and a vocab file
class ParallelTextDataset(Dataset):

    def __init__(self, src_file_path, trg_file_path, src_vocab=None,
                 trg_vocab=None, extend_vocab=False, device='cuda'):
        (self.data, self.src_vocab, self.trg_vocab,
         self.src_max_seq_length, self.tgt_max_seq_length) = self.parallel_text_to_data(
            src_file_path, trg_file_path, src_vocab, trg_vocab, extend_vocab, device)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def parallel_text_to_data(self, src_file, tgt_file, src_vocab=None, tgt_vocab=None,
                              extend_vocab=False, device='cuda'):
        # Convert paired src/tgt texts into torch.tensor data.
        # All sequences are padded to the length of the longest sequence
        # of the respective file.

        assert os.path.exists(src_file)
        assert os.path.exists(tgt_file)

        if src_vocab is None:
            src_vocab = Vocabulary()

        if tgt_vocab is None:
            tgt_vocab = Vocabulary()

        data_list = []
        # Check the max length, if needed construct vocab file.
        src_max = 0
        with open(src_file, 'r') as text:
            for line in text:
                tokens = list(line)
                length = len(tokens)
                if src_max < length:
                    src_max = length

        tgt_max = 0
        with open(tgt_file, 'r') as text:
            for line in text:
                tokens = list(line)
                length = len(tokens)
                if tgt_max < length:
                    tgt_max = length
        tgt_max += 2  # add for begin/end tokens

        src_pad_idx = src_vocab.pad_id
        tgt_pad_idx = tgt_vocab.pad_id

        tgt_eos_idx = tgt_vocab.eos_id
        tgt_sos_idx = tgt_vocab.sos_id

        # Construct data
        src_list = []
        print(f"Loading source file from: {src_file}")
        with open(src_file, 'r') as text:
            for line in tqdm(text):
                seq = []
                tokens = list(line)
                for token in tokens:
                    seq.append(src_vocab.get_idx(token, extend_vocab=extend_vocab))
                var_len = len(seq)
                var_seq = torch.tensor(seq, device=device, dtype=torch.int64)
                # padding
                new_seq = var_seq.data.new(src_max).fill_(src_pad_idx)
                new_seq[:var_len] = var_seq
                src_list.append(new_seq)

        tgt_list = []
        print(f"Loading target file from: {tgt_file}")
        with open(tgt_file, 'r') as text:
            for line in tqdm(text):
                seq = []
                tokens = list(line)
                # append a start token
                seq.append(tgt_sos_idx)
                for token in tokens:
                    seq.append(tgt_vocab.get_idx(token, extend_vocab=extend_vocab))
                # append an end token
                seq.append(tgt_eos_idx)

                var_len = len(seq)
                var_seq = torch.tensor(seq, device=device, dtype=torch.int64)

                # padding
                new_seq = var_seq.data.new(tgt_max).fill_(tgt_pad_idx)
                new_seq[:var_len] = var_seq
                tgt_list.append(new_seq)

        # src_file and tgt_file are assumed to be aligned.
        assert len(src_list) == len(tgt_list)
        for i in range(len(src_list)):
            data_list.append((src_list[i], tgt_list[i]))

        print("Done.")

        return data_list, src_vocab, tgt_vocab, src_max, tgt_max


def load_dataset(dataset_dir, file_name, file_ext, train_set = None):
    """
    :param dataset_dir: directory where data is located
    :param file_name: file name of the data file
    :param file_ext: a list of file extensions of data and target.
                     First element is extension for data, second element is extension for label
    :param train_set: provide train_set if loading validation set
    :return: a ParallelTextDataset class
    """

    src_file_path = f"{dataset_dir}/{file_name}{file_ext[0]}"
    trg_file_path = f"{dataset_dir}/{file_name}{file_ext[1]}"

    if train_set:
        src_vocab = train_set.src_vocab
        trg_vocab = train_set.trg_vocab

        dataset = ParallelTextDataset(
            src_file_path, trg_file_path, src_vocab=src_vocab, trg_vocab=trg_vocab,
            extend_vocab=False)

    else:
        dataset = ParallelTextDataset(src_file_path, trg_file_path, extend_vocab=True)

    return dataset


def get_accu(y_true, y_pred):
    """
    Function for calculating the accuracy
    :param y_true: label
    :param y_pred: predicted output
    :return: accuracy
    """
    # y tensor shape in (sequence_length, batch_size)
    bol = (y_pred == y_true).all(dim=1)

    correct = torch.sum(bol)

    accu = correct / y_true.shape[0]

    return accu


def evaluate(eval_model, valid_data_loader, criterion, trg_vocab, DEVICE):
    """
    :param eval_model: torch model for evaluation
    :param valid_data_loader: data loader for validation data
    :param criterion: loss criterion
    :param trg_vocab: target vocab
    :return: validation loss and validation accuracy
    """
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    total_accu = 0.
    ntokens = len(trg_vocab.id_to_string)

    tb = len(valid_data_loader)
    with torch.no_grad():
        for X, y in valid_data_loader:
            X = X.permute(1, 0)

            y_input = y[:, :-1]
            y_expected = y[:, 1:]
            y_input = y_input.permute(1, 0)
            # get the output from the model
            output = eval_model(X, y_input)
            output = output.permute(1, 2, 0)
            total_loss += criterion(output, y_expected)
            predicted = eval_model.greedy_search(X, y_input).to(DEVICE)
            total_accu += get_accu(y_expected, predicted)

    loss = total_loss / tb
    accu = total_accu / tb

    print('Validation | loss {:5.2f} | accu {:8.2f}%'.format(loss, accu * 100))

    return loss, accu


def train(log_interval, model, train_data_loader, valid_data_loader, optimizer, epoch, criterion, trg_vocab,
          DEVICE, losses=[], accuracy=[], valid_accu=[], valid_loss=[], k=10, clip_rate=0.5):
    model.train()
    total_loss = 0.
    total_accu = 0.

    val_accu = 0

    N_count = 0
    ntokens = len(trg_vocab.id_to_string)
    tb = len(train_data_loader)
    with tqdm(train_data_loader, unit="batch") as tepoch:
        for batch_idx, (X, y) in enumerate(tepoch):
            X = X.permute(1, 0)

            y_input = y[:, :-1]
            y_expected = y[:, 1:]
            y_input = y_input.permute(1, 0)
            # get the output from the model
            output = model(X, y_input)

            predicted = model.greedy_search(X, y_input).to(DEVICE)
            # calculate the loss
            output = output.permute(1, 2, 0)
            loss = criterion(output, y_expected)
            loss.backward()

            # gradient accumulation
            if ((batch_idx + 1) % k == 0 or (batch_idx + 1 == tb)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_rate)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()

            losses.append(loss.item())
            accu = get_accu(y_expected, predicted)
            total_accu += accu.item()
            accuracy.append(accu.item())

            if (batch_idx + 1) % log_interval == 0 and batch_idx > 0:
                cur_loss = total_loss / log_interval
                cur_accu = total_accu / log_interval

                tepoch.set_postfix(loss=cur_loss, accuracy=100. * cur_accu)
                """print('Training | epoch {:3d} | {:5d}/{:5d} batch | loss {:5.2f} | accu {:8.2f}%'.format(
                    epoch, batch_idx + 1, tb, cur_loss, cur_accu * 100))"""
                val_loss, val_accu = evaluate(model, valid_data_loader, criterion,
                                              trg_vocab)  # evaluate using valid data set

                valid_accu.append(val_accu.item())
                valid_loss.append(val_loss.item())

                total_loss = 0
                total_accu = 0

            elif (batch_idx + 1) == tb:
                cur_loss = total_loss / log_interval
                cur_accu = total_accu / log_interval
                tepoch.set_postfix(loss=cur_loss, accuracy=100. * cur_accu)
                """ print('Training | epoch {:3d} done | {:5d}/{:5d} batch | loss {:5.2f} | accu {:8.2f}%'.format(
                    epoch, batch_idx + 1, tb, cur_loss, cur_accu * 100))"""

                val_loss, val_accu = evaluate(model, valid_data_loader, criterion, trg_vocab)

                valid_accu.append(val_accu.item())
                valid_loss.append(val_loss.item())

                total_loss = 0
                total_accu = 0

            if val_accu > 0.9:  # stop training if we get validation accuracy larger than 0.9
                # stop training if get validation accuracy greater than 0.9
                print("Training Done | Validation Accuracy: ", val_accu.item() * 100)
                return

