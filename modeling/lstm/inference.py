import torch
from modeling.lstm.models import NLINet
import os
from utils import load_json

if __name__ == '__main__':
    config = load_json('dataset/d-atomic/config_nli_model.json')
    training_params = load_json('dataset/d-atomic/params.json')

    nli_net = torch.load_state_dict(torch.load('dataset/d-atomic/infersent.pth'))