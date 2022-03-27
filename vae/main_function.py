import torch
from tqdm import tqdm


def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    train_loss = 0

    t = tqdm(train_loader)

    for batch_idx, data in enumerate(t, 1):
        optimizer.zero_grad()
        data = data.to(device)
        output, mu, logvar = model(data)
        loss = model.loss(output, data, mu, logvar)
        loss.backward()
        optimizer.step()

        cpu_loss = loss.item()
        train_loss += cpu_loss

        if batch_idx % log_interval == 0:
            t.set_description(f'Train Epoch: {epoch} Loss: {cpu_loss:.6f}')

    train_loss /= len(train_loader)
    print('Train set Average loss:', train_loss)
    return train_loss


def test(model, device, test_loader, epoch, return_images=0, log_interval=1):
    model.eval()
    test_loss = 0

    # two np arrays of images
    original_images = []
    recreated_images = []

    t = tqdm(test_loader)

    with torch.no_grad():
        for batch_idx, data in enumerate(t, 1):
            data = data.to(device)
            output, mu, logvar = model(data)
            loss = model.loss(output, data, mu, logvar)

            cpu_loss = loss.item()
            test_loss += cpu_loss

            if return_images > 0 and len(original_images) < return_images:
                original_images.append(data[0].cpu())
                recreated_images.append(output[0].cpu())

            if batch_idx % log_interval == 0:
                t.set_description(f'Test Epoch: {epoch} Loss: {cpu_loss:.6f}')

    test_loss /= len(test_loader)
    print('Test set Average loss:', test_loss)

    if return_images > 0:
        return test_loss, original_images, recreated_images

    return test_loss
