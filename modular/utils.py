import torch
from pathlib import Path


def save_model(model: torch.nn.Module, target_dir: str, model_name: str):

    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pt") or model_name.endswith("pth")

    model_save_path = target_dir_path / model_name

    print(f"[INFO] Saving model to: {model_save_path}")

    torch.save(model.state_dict(), model_save_path)


def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc
