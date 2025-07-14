import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .models import ClassificationLoss, load_model, save_model
from .utils import load_data
from .utils import compute_accuracy 


def train(
    exp_dir: str = "logs",
    model_name: str = "linear",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    train_data = load_data("classification_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("classification_data/val", shuffle=False)

    # create loss function and optimizer
    loss_func = ClassificationLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0
    metrics = {"train_acc": [], "val_acc": []}

    # training loop
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        for key in metrics:
            metrics[key].clear()

        model.train()

        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            # TODO: implement training step
            optimizer.zero_grad()
            logits = model(img)
            loss = loss_func(logits, label)
            loss.backward()
            optimizer.step()

            acc = compute_accuracy(logits, label)
            logger.add_scalar("train_loss", loss.item(), global_step)
            metrics["train_acc"].append(acc.item())


            global_step += 1

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()

            for img, label in val_data:
                img, label = img.to(device), label.to(device)

                # TODO: compute validation accuracy
                logits = model(img)
                acc = compute_accuracy(logits, label)
                metrics["val_acc"].append(acc.item())

        # log average train and val accuracy to tensorboard
        epoch_train_acc = torch.as_tensor(metrics["train_acc"]).mean()
        epoch_val_acc = torch.as_tensor(metrics["val_acc"]).mean()

        # raise NotImplementedError("Logging not implemented")
        logger.add_scalar("train_accuracy", epoch_train_acc.item(), epoch)
        logger.add_scalar("val_accuracy", epoch_val_acc.item(), epoch)

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_acc={epoch_train_acc:.4f} "
                f"val_acc={epoch_val_acc:.4f}"
            )

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--batch_size", type=int, default=128)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))

# LinearClassifier - default params
# Model size: 0.28 MB
# Epoch  1 / 50: train_acc=0.6781 val_acc=0.7142
# Epoch 10 / 50: train_acc=0.8224 val_acc=0.7405
# Epoch 20 / 50: train_acc=0.8468 val_acc=0.7401
# Epoch 30 / 50: train_acc=0.8619 val_acc=0.7698
# Epoch 40 / 50: train_acc=0.8773 val_acc=0.7469
# Epoch 50 / 50: train_acc=0.8760 val_acc=0.7405
# Model saved to logs/linear_0713_003714/linear.th

# MLPClassifier - default params
# Model size: 9.07 MB
# Epoch  1 / 50: train_acc=0.6951 val_acc=0.7375
# Epoch 10 / 50: train_acc=0.9359 val_acc=0.7982
# Epoch 20 / 50: train_acc=0.9663 val_acc=0.7977
# Epoch 30 / 50: train_acc=0.9779 val_acc=0.8022
# Epoch 40 / 50: train_acc=0.9906 val_acc=0.8141
# Epoch 50 / 50: train_acc=0.9880 val_acc=0.8004
# Model saved to logs/mlp_0713_213046/mlp.th

# MLPClassifierDeep - lr == 2e-3, num_layers=10, hidden_dim=128
# Model size: 6.57 MB
# Epoch  1 / 50: train_acc=0.4510 val_acc=0.6474
# Epoch 10 / 50: train_acc=0.8908 val_acc=0.8121
# Epoch 20 / 50: train_acc=0.9330 val_acc=0.8051
# Epoch 30 / 50: train_acc=0.9537 val_acc=0.8162
# Epoch 40 / 50: train_acc=0.9585 val_acc=0.7809
# Epoch 50 / 50: train_acc=0.9741 val_acc=0.8077
# Model saved to logs/mlp_deep_0714_035330/mlp_deep.th

# python3 -m homework.train --model_name mlp_deep --num_epoch 50 --lr 1e-3 --batch_size 128