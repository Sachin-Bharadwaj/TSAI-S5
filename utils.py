import torch
from torchvision import datasets, transforms
import torch.optim as optim
from matplotlib import pyplot as plt
from tqdm import tqdm


def get_tfms(type: str) -> transforms:
    if type == "train":
        # Train data transformations
        tfm = transforms.Compose(
            [
                transforms.RandomApply(
                    [
                        transforms.CenterCrop(22),
                    ],
                    p=0.1,
                ),
                transforms.Resize((28, 28)),
                transforms.RandomRotation((-15.0, 15.0), fill=0),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
    else:
        # Test data transformations
        tfm = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
    return tfm


def get_ds(type: str, tfms, download=True):
    if type == "train":
        ds = datasets.MNIST("../data", train=True, download=download, transform=tfms)
    else:
        ds = datasets.MNIST("../data", train=False, download=download, transform=tfms)
    return ds


def get_dls(train_ds, test_ds, **kwargs):
    train_dl = torch.utils.data.DataLoader(train_ds, **kwargs)
    test_dl = torch.utils.data.DataLoader(test_ds, **kwargs)
    return train_dl, test_dl


def GetCorrectPredCount(pPrediction, pLabels):
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()


def train_step(model, device, train_loader, optimizer, criterion, train_acc, train_losses):
    model.train()
    pbar = tqdm(train_loader)

    train_loss = 0
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Predict
        pred = model(data)

        # Calculate loss
        loss = criterion(pred, target)
        train_loss += loss.item() * len(data)
        # pdb.set_trace()

        # Backpropagation
        loss.backward()
        optimizer.step()

        correct += GetCorrectPredCount(pred, target)
        processed += len(data)

        pbar.set_description(
            desc=f"Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}"
        )

    train_acc.append(100 * correct / processed)
    train_losses.append(train_loss / len(train_loader))
    return train_acc, train_losses


def test_step(model, device, test_loader, criterion, test_acc, test_losses):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss

            correct += GetCorrectPredCount(output, target)

    test_loss /= len(test_loader.dataset)
    test_acc.append(100.0 * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return test_acc, test_losses

def plot_acc_loss(train_losses, train_acc, test_losses, test_acc):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")

def show_dls_samples(dataloader):
    batch_data, batch_label = next(iter(dataloader))

    fig = plt.figure()

    for i in range(12):
        plt.subplot(3, 4, i + 1)
        plt.tight_layout()
        plt.imshow(batch_data[i].squeeze(0), cmap='gray')
        plt.title(batch_label[i].item())
        plt.xticks([])
        plt.yticks([])

def get_optimizer(params, lr=4e-3, momentum=0.9):
    return optim.SGD(params, lr=lr, momentum=momentum)

def get_scheduler(optimizer, step_size=15, gamma=0.1, verbose=True):
    return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, verbose=verbose)

def getlossCriterion():
    return torch.nn.CrossEntropyLoss(reduction='mean')
