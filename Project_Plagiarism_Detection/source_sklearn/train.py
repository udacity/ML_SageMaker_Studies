from __future__ import print_function

import argparse
import os
import pandas as pd
import sys
import json

# pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

# from sklearn.externals import joblib
from model import SimpleNet


# Provided model load function
def model_fn(model_dir):
    """Load model from the model_dir. This is the same model that is saved
    in the main if statement.
    """
    print("Loading model.")
    
    # load using joblib
#     model = joblib.load(os.path.join(model_dir, "model.joblib"))
    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNet(model_info['input_dim'], 
                      model_info['hidden_dim'], 
                      model_info['output_dim'])

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))
        
    print("Done loading model.")
    
    return model.to(device)

# Provided model saving functions
def save_model(model, model_dir):
    print("Saving the model.")
    path = os.path.join(model_dir, 'model.pth')
    # save state dictionary
    torch.save(model.cpu().state_dict(), path)

def save_model_params(model, model_dir):
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'input_dim': args.input_dim,
            'hidden_dim': args.hidden_dim,
            'output_dim': args.output_dim
        }
        torch.save(model_info, f)
        
# Provided train function
def train(model, train_loader, epochs, optimizer, criterion, device):
    """
    This is the training method that is called by the PyTorch training script. The parameters
    passed are as follows:
    model        - The PyTorch model that we wish to train.
    train_loader - The PyTorch DataLoader that should be used during training.
    epochs       - The total number of epochs to train for.
    optimizer    - The optimizer to use during training.
    criterion    - The loss function used for training. 
    device       - Where the model and data should be loaded (gpu or cpu).
    """
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            # prep data
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad() # zero accumulated gradients
            # get output of SimpleNet
            output = model(data)
            # calculate loss and perform backprop
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item()
        
        # print loss stats
        print("Epoch: {}, Loss: {}".format(epoch, total_loss / len(train_loader)))

    # save trained model, after all epochs
    save_model(model, args.model_dir)


## TODO: Complete the main code
if __name__ == '__main__':
    
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    # Model parameters
    parser.add_argument('--input_dim', type=int, default=6, metavar='IN',
                        help='number of input features to model (default: 6)')
    parser.add_argument('--hidden_dim', type=int, default=10, metavar='H',
                        help='hidden dim of model (default: 10)')
    parser.add_argument('--output_dim', type=int, default=1, metavar='OUT',
                        help='output dim of model (default: 1)')
    
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    
    
    # args holds all passed-in arguments
    args = parser.parse_args()

    # Read in csv training file
    training_dir = args.data_dir
    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)

    # Labels are in the first column
#     train_y = train_data.iloc[:,0]
#     train_x = train_data.iloc[:,1:]
    
    # labels are first column
    train_y = torch.from_numpy(train_data.iloc[:,0].values).float().squeeze()
    train_x = torch.from_numpy(train_data.iloc[:,1:].values).float()
    train_ds = torch.utils.data.TensorDataset(train_x, train_y)
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    

    ## Define a model 
    model = SimpleNet(args.input_dim, args.hidden_dim, args.output_dim).to(device)
    
    # Given: save the parameters used to construct the model
    save_model_params(model, args.model_dir)
    
    ## Define an optimizer and loss function for training
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()
    
    # Trains the model (given line of code, which calls the above training function)
    # This function *also* saves the model state dictionary
    train(model, train_loader, args.epochs, optimizer, criterion, device)
    
    ## --- End of your code  --- ##
    

    # Save the trained model
#     joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))