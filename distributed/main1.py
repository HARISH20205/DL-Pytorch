import torch
from torch import nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from pathlib import Path
import os
from helper_functions import accuracy_fn


# print(torch.cuda.is_available())
# print(torch.distributed.is_available())
# print(torch.distributed.is_nccl_available())

file_path = Path("mini_food")
image_path = file_path /"pizza_steak_sushi"
train_dir = image_path / "train"
test_dir = image_path / "test"

BATCH_SIZE = 16
NUM_WORKERS = os.cpu_count() - 2
LOCAL_RANK = int(os.environ.get('LOCAL_RANK', 0))
WORLD_SIZE = int(os.environ.get('WORLD_SIZE',2))
WORLD_RANK = int(os.environ.get('RANK', 0))



data_transform = transforms.Compose(
    [
        transforms.Resize(size=(64,64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ]
)

train_data = datasets.ImageFolder(
    root=train_dir,
    transform=data_transform,
    target_transform=None
)

test_data = datasets.ImageFolder(
    root=test_dir,
    transform = data_transform
)


class_names,class_dict = train_data.classes,train_data.class_to_idx





train_sampler = DistributedSampler(
    train_data,
    num_replicas=WORLD_SIZE,
    rank=WORLD_RANK,
    shuffle=True
)

train_dataloader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    sampler=train_sampler,
    pin_memory=True
)

test_sampler = DistributedSampler(
    test_data,
    num_replicas=WORLD_SIZE,
    rank=WORLD_RANK,
    shuffle=False
)

test_dataloader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    sampler=test_sampler,
    pin_memory=True
)

device = "cuda" if torch.cuda.is_available() else "cpu"



class MiniFoodCNN(nn.Module):
    def __init__(self,input_shape,hidden_units,output_shape):
        super().__init__()
        
        self.conv_block_1 = nn.Sequential(
            
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=2,
                stride=1,
                padding=1
            ),
            
            nn.ReLU(),
            
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=2,
                stride=1,
                padding=1
            ),
            
            nn.ReLU(),
            
            nn.MaxPool2d(
                kernel_size=2
            )
        )
        
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                padding=1,
                stride=1
            ),
            
            nn.ReLU(),
            
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=4,
                padding=1,
                stride=1
            ),
            
            nn.ReLU(),
            
            nn.MaxPool2d(
                kernel_size=2
            )
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*16*16,
                      out_features=output_shape)
        )        
    def forward(self,x):
        x = x.to(device=device)
        
        x = self.conv_block_1(x)
        # print(x.shape)
        
        x = self.conv_block_2(x)
        # print(x.shape)
        
        x = self.classifier(x)
        
        return x
    



def setup_ddp():
    """Initialize the distributed process group safely."""
    dist.init_process_group(backend='nccl')

    cuda_count = torch.cuda.device_count()
    print(f"cuda_count: {cuda_count} LOCAL_RANK: {LOCAL_RANK} WORLD_SIZE: {WORLD_SIZE}")

    if cuda_count == 0:
        raise RuntimeError("No CUDA devices found but NCCL backend requested.")
    
    if cuda_count == 1:
        torch.cuda.set_device(0) #used this for local testing when only one gpu is available
    else:
        torch.cuda.set_device(LOCAL_RANK)



def cleanup_ddp():
    """Clean up the distributed process group"""
    dist.destroy_process_group()

def train_step(model, optimizer, loss_fn, train_dataloader, device, epoch):

    print("Starting training...")
    

    train_loss, train_acc = 0, 0


    # for DDP, we need to use sampler and set the epoch for shuffling, its for makeing  each replica gets different data each epoch
    train_dataloader.sampler.set_epoch(epoch)
    
    model.train()
    
    for batch, (X, y) in enumerate(train_dataloader):
        X = X.to(device)
        y = y.to(device)
        
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        
        train_loss += loss.item()
        train_acc += accuracy_fn(y, y_pred.argmax(dim=1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)
    
    if WORLD_RANK == 0:
        print(f" Train Loss: {train_loss:.4f} || Train Accuracy {train_acc:.2f}%")

    
    return train_loss, train_acc

def eval_model(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               accuracy_fn,device):
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            
            X,y =X.to(device),y.to(device)
            y_pred = model(X)
            
            
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, 
                                y_pred=y_pred.argmax(dim=1)) 
        
        
        loss /= len(data_loader)
        acc /= len(data_loader)

    return {"model_name": model.__class__.__name__, 
            "model_loss": loss.item(),
            "model_acc": acc}


def test_step(model, loss_fn, test_dataloader, device):
    test_loss, test_acc = 0.0, 0
    
    model.eval()
    
    with torch.inference_mode():
        for X_test, y_test in test_dataloader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            test_pred = model(X_test)
            
            test_loss += loss_fn(test_pred, y_test).item()
            test_acc += accuracy_fn(y_test, test_pred.argmax(dim=1))
        
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
    
    if WORLD_RANK == 0:
        print(f" Test Loss: {test_loss:.4f} || Test Accuracy {test_acc:.2f}%\n")

    return test_loss, test_acc

def main():
    setup_ddp()
    print(f"Running DDP on rank {WORLD_RANK}.")
    print(f"Local rank: {LOCAL_RANK} || World size: {WORLD_SIZE}")

    # device = torch.device(f'cuda:{LOCAL_RANK}')
    device = torch.device("cuda:0") #for local testing when only one gpu is available, uncomment this line and comment the above line
    
    model = MiniFoodCNN(input_shape=3, hidden_units=10, output_shape=3)
    model = model.to(device)
    
    model = DDP(model, device_ids=[LOCAL_RANK])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    epochs = 10
    for epoch in range(epochs):
        
        train_loss, train_acc = train_step(
            model, optimizer, loss_fn, train_dataloader, device, epoch
        )
        
        test_loss, test_acc = test_step(
            model, loss_fn, test_dataloader, device
        )

        if WORLD_RANK == 0:
            eval = eval_model(model,test_dataloader,loss_fn,accuracy_fn,device)     
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 30)
            print(f" Train Loss: {train_loss:.4f} || Train Accuracy {train_acc:.2f}%")
            print(f" Test Loss: {test_loss:.4f} || Test Accuracy {test_acc:.2f}%")
            print(f" Eval Loss: {eval['model_loss']:.4f} || Eval Acc: {eval['model_acc']:.2f}%")
            

    
    cleanup_ddp()

if __name__ == "__main__":
    main()