import torch
import torch.nn as nn
from torchsummary import summary
import os
from option import args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(os.path.join(args.model_dir, args.model_name), map_location=device)

summary(model, input_size=(1,530, 960),batch_size=64)