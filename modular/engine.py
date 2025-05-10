import torch
from tqdm.auto import tqdm
from utils import accuracy_fn

# from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter()

torch.manual_seed(42)


def train_step(model, optimizer, loss_fn, train_dataloader, device):

    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(train_dataloader):

        X = X.to(device)
        y = y.to(device)

        model.train()

        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss

        train_acc += accuracy_fn(y, y_pred.argmax(dim=1))

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)
    # print(f" Train Loss: {train_loss:.4f} || Train Accuracy {train_acc:.2f}%")
    return train_loss, train_acc


def test_step(model, loss_fn, test_dataloader, device):
    test_loss, test_acc = 0.0, 0

    model.eval()

    with torch.inference_mode():
        for X_test, y_test in test_dataloader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            test_pred = model(X_test)

            test_loss += loss_fn(test_pred, y_test)

            test_acc += accuracy_fn(y_test, test_pred.argmax(dim=1))

        test_loss /= len(test_dataloader)

        test_acc /= len(test_dataloader)

    # print(f" Test Loss: {test_acc:.4f} || Test Accuracy {test_acc:.2f}%\n")

    return test_loss, test_acc


def train(
    model, optimizer, loss_fn, train_dataloader, test_dataloader, epochs, device, writer
):
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    model.to(device)

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model, optimizer, loss_fn, train_dataloader, device
        )

        test_loss, test_acc = test_step(model, loss_fn, test_dataloader, device)
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )
        results["train_loss"].append(
            train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss
        )
        results["train_acc"].append(
            train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc
        )
        results["test_loss"].append(
            test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss
        )
        results["test_acc"].append(
            test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc
        )
        if writer:
            writer.add_scalars(
                main_tag="Loss",
                tag_scalar_dict={"train_loss": train_loss, "test_loss": test_loss},
                global_step=epoch,
            )
            writer.add_scalars(
                main_tag="Accuracy",
                tag_scalar_dict={"train_acc": train_acc, "test_acc": test_acc},
                global_step=epoch,
            )

            writer.add_graph(
                model=model,
                input_to_model=torch.randn(1, 3, 384, 384).to(device=device),
            )
        else:
            pass
    if writer:
        writer.close
    return results
