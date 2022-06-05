
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T

transforms = T.Compose([
    T.Resize(226),                     
    T.RandomCrop(224),                 
    T.ToTensor(),                               
    T.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
])


def show_image(img, title=None, pause_time=5, save_path=None):
    """Imshow for Tensor."""
    img[0] = img[0] * 0.229
    img[1] = img[1] * 0.224 
    img[2] = img[2] * 0.225 
    img[0] += 0.485 
    img[1] += 0.456 
    img[2] += 0.406
    
    img = img.numpy().transpose((1, 2, 0))
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    if not save_path:
        plt.savefig('test.png')

    plt.pause(pause_time)  # pause a bit so that plots are updated



def save_checkpoint(state, filename="checkpoint.pth.tar"):
    # print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step


def calculate_centroid(alpha):
    alpha = alpha.reshape(7, 7)
    m, n = alpha.shape
    centroid = np.zeros((1, 2))
    for i in range(m):
        for j in range(n):
            centroid += alpha[i, j] * np.array([i, j])
    return centroid

def maxpool_centroid(alpha):
    alpha = alpha.reshape(7, 7)
    m, n = alpha.shape
    max_cen = [(0, 0), -1]
    for i in range(m):
        for j in range(n):
            if alpha[i, j] > max_cen[1]:
                max_cen = [(i, j), alpha[i, j]] 
    return max_cen[0]