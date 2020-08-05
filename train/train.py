import argparse
import json
import os
import pickle
import sys
import sagemaker_containers
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data
import torch.nn as nn

from model import LSTMRegressor

def model_fn(model_dir):
    """
    Load PyTorch model from model_dir
    """
    
    print("Start loading model...")
    
    # first load the model creation parameters
    model_params = {}
    model_params_path = os.path.join(model_dir, "model_params.pth")
    with open(model_params_path, 'rb') as mp_f:
        model_params = torch.load(mp_f)
        
    print("model parameters loaded: {}".format(model_params))
    
    # determine whether to use cpu/gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # create model
    model = LSTMRegressor(model_params['embedding_dim'], model_params['hidden_dim'], model_params['vocab_size'])
    
    # load the stored state of the model
    model_state_path = os.path.join(model_dir, "model_state.pth")
    with open(model_state_path, 'rb') as ms_f:
        model.load_state_dict(torch.load(ms_f), strict=False)
        
    # load word_dict
    word_dict_path = os.path.join(model_dir, "word_dict.pickle")
    with open(word_dict_path, 'rb') as wd_f:
        model.word_dict = pickle.load(wd_f)
    
    model.to(device).eval()
    
    print("Finish loading model...")
    
    return model

def _get_train_data_loader(batch_size, training_dir, training_data_file):
    """
    Return a training data loader for training purpose
    """
    
    train_data = pd.read_csv(os.path.join(training_dir, training_data_file), header=None, names=None)
    
    train_y = torch.from_numpy(train_data[[0]].values).float().squeeze()
    train_X = torch.from_numpy(train_data.drop([0], axis=1).values).long()
    
    train_ds = torch.utils.data.TensorDataset(train_X, train_y)
    
    return torch.utils.data.DataLoader(train_ds, batch_size=batch_size)

def _get_valid_data_loader(batch_size, valid_dir, valid_data_file):
    """
    Return a training data loader for training purpose
    """
    
    valid_data = pd.read_csv(os.path.join(valid_dir, valid_data_file), header=None, names=None)
    
    valid_y = torch.from_numpy(valid_data[[0]].values).float().squeeze()
    valid_X = torch.from_numpy(valid_data.drop([0], axis=1).values).long()
    
    valid_ds = torch.utils.data.TensorDataset(valid_X, valid_y)
    
    return torch.utils.data.DataLoader(valid_ds, batch_size=batch_size)

def train(model, train_loader, valid_loader, epochs, optimizer, loss_fn, device):
    """
    This function trains the LSTM with a training data loader from batches of training data
    """
    total_loss = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            
            batch_X, batch_y = batch
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            output = model(batch_X)
            loss = loss_fn(output, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.data.item()
        print("Epoch: {}, Loss: {}".format(epoch, total_loss / len(train_loader)))
    print("Loss: {}".format(total_loss / len(train_loader)))
    
    total_valid_loss = 0
    for valid_batch in valid_loader:
        valid_batch_X, valid_batch_y = valid_batch
        valid_batch_X = valid_batch_X.to(device)
        valid_batch_y = valid_batch_y.to(device)
        
        valid_output = model(valid_batch_X)
        valid_loss = loss_fn(valid_output, valid_batch_y)
        total_valid_loss += valid_loss.data.item()
    
    print("Validation Loss: {}".format(total_valid_loss / len(valid_loader)))     

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--embedding_dim', type=int, default=32, metavar='N',
                        help='size of the word embeddings (default: 32)')
    parser.add_argument('--hidden_dim', type=int, default=100, metavar='N',
                        help='size of the hidden dimension (default: 100)')
    parser.add_argument('--vocab_size', type=int, default=10000, metavar='N',
                        help='size of the vocabulary (default: 10000)')
    parser.add_argument('--num_layer', type=int, default=1, metavar='N',
                        help='layer of lstm')
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--training_data_file', type=str, default='google_train.csv')
    parser.add_argument('--valid_data_file', type=str, default='google_valid.csv')
    parser.add_argument('--word_dict_file', type=str, default='google_dict.pickle')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(args.seed)
    
    # load training data
    train_loader = _get_train_data_loader(args.batch_size, args.data_dir, args.training_data_file)
    
    valid_loader = _get_valid_data_loader(args.batch_size, args.data_dir, args.valid_data_file)
    
    # create model
    model = LSTMRegressor(args.embedding_dim, args.hidden_dim, args.vocab_size, args.num_layer).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss()
    
    train(model, train_loader, valid_loader, args.epochs, optimizer, loss_fn, device)
   
    # save model and related parameters
    model_params_path = os.path.join(args.model_dir, 'model_params.pth')
    with open(model_params_path, 'wb') as mp_f:
        model_params = {
            'embedding_dim': args.embedding_dim,
            'hidden_dim': args.hidden_dim,
            'vocab_size': args.vocab_size,
        }
        torch.save(model_params, mp_f)
        
    model_state_path = os.path.join(args.model_dir, 'model_state.pth')
    with open(model_state_path, 'wb') as ms_f:
        torch.save(model.cpu().state_dict(), ms_f)
        
    word_dict_path = os.path.join(args.model_dir, 'word_dict.pickle')
    with open(word_dict_path, 'wb') as wd_f:
        pickle.dump(model.word_dict, wd_f)