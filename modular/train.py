import os
import torch
from torch import nn
import data_setup, utils, model_builder, engine

from torchvision import transforms


NUM_EPOCHS = 5
BATCH_SIZE = 16
HIDDEN_UNITS = 10
LEARNING_RATE = 0.01


device = "cuda" if torch.cuda.is_available() else "cpu"


train_dir = "mini_food/pizza_steak_sushi/train"
test_dir = "mini_food/pizza_steak_sushi/test"

data_transforms = transforms.Compose(
    [transforms.Resize(size=(64, 64)), transforms.ToTensor()]
)


train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir,
    test_dir,
    data_transforms,
    BATCH_SIZE,
)

model = model_builder.MiniFoodCNN(
    input_shape=3, hidden_units=HIDDEN_UNITS, output_shape=len(class_names)
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)


engine.train(
    model, optimizer, loss_fn, train_dataloader, test_dataloader, NUM_EPOCHS, device
)

utils.save_model(model, target_dir="models", model_name="going_modular_model.pth")
