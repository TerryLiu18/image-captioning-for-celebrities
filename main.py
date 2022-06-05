import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.transforms as T

from config import DATA_LOCATION, TEST_LOCATION, print_every
from torch.utils.data import DataLoader, Dataset
from data_loader import FlickrDataset, TestDataset, get_data_loader, get_test_data
from utils import transforms, load_checkpoint, save_checkpoint, show_image
from model import EncoderDecoder
from face_recognition import get_name

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



BATCH_SIZE = 256
NUM_WORKER = 1

transforms = T.Compose([
    T.Resize(226),                     
    T.RandomCrop(224),                 
    T.ToTensor(),                               
    T.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
])

# testing the dataset class
dataset =  FlickrDataset(
    root_dir = DATA_LOCATION + "/Images",
    caption_file = DATA_LOCATION + "/captions.txt",
    transform=transforms
)

# writing the dataloader
data_loader = get_data_loader(
    dataset=dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKER,
    shuffle=True,
)

# testdataset = TestDataset(
#     root_dir=TEST_LOCATION,
#     transform=transforms
# )

# test_loader = get_test_loader(
#     dataset=testdataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     num_workers=NUM_WORKER,
# )

embed_size = 300
vocab_size = len(dataset.vocab)
attention_dim = 256
encoder_dim = 2048
decoder_dim = 512
learning_rate = 3e-4


# init model
model = EncoderDecoder(
    embed_size=300,
    vocab_size = len(dataset.vocab),
    attention_dim=256,
    encoder_dim=2048,
    decoder_dim=512
).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

load_model = True
need_train = False
num_epochs = 0

# generate caption
def get_caps_from(features_tensors):
    #generate the caption
    model.eval()
    with torch.no_grad():
        features = model.encoder(features_tensors.to(device))
        caps, alphas = model.decoder.generate_caption(features, vocab=dataset.vocab)
        caption = ' '.join(caps[:-1])
        show_image(features_tensors[0],title=caption)
    return caps, alphas

#Show attention
def plot_attention(img, result, attention_plot):
    #untransform
    img[0] = img[0] * 0.229
    img[1] = img[1] * 0.224 
    img[2] = img[2] * 0.225 
    img[0] += 0.485 
    img[1] += 0.456 
    img[2] += 0.406
    
    img = img.numpy().transpose((1, 2, 0))
    temp_image = img

    fig = plt.figure(figsize=(15, 15))

    len_result = len(result)
    # print("type(attention_plot[0]):",type(attention_plot[0]))
    # print(attention_plot.shape)
    for l in range(len_result - 2):
        temp_att = attention_plot[l].reshape(7,7)
        
        ax = fig.add_subplot(len_result//2,len_result//2, l+1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.7, extent=img.get_extent())
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    print('start training!')
    if load_model and os.path.exists("checkpoint.pth.tar"):
        step = load_checkpoint(torch.load("checkpoint.pth.tar"), model, optimizer)
        print(f'step {step} model loaded')
    else:
        step = 0

    if need_train:
        for epoch in range(1, num_epochs + 1):
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step
            }
            save_checkpoint(checkpoint)

            for idx, data in enumerate(data_loader):
                image, captions = data[0].to(device), data[1].to(device)

                optimizer.zero_grad()
                outputs, attentions = model(image, captions)

                # Calculate the batch loss.
                targets = captions[:, 1:]
                loss = criterion(outputs.view(-1, vocab_size), targets.reshape(-1))
                loss.backward() # Backward pass.
                optimizer.step() # Update the parameters in the optimizer.

                step += 1

                if (idx + 1) % print_every == 0:
                    print("Epoch: {} loss: {:.5f}".format(epoch, loss.item()))

    img, name_list = get_test_data()
    # display image

    transforms2 = T.Compose([
        T.Resize(226),                     
        T.RandomCrop(224),                 
        T.ToTensor(),                               
        T.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
    ])
    img = transforms2(img)

    # dataiter = iter(test_loader)
    # images, name_list = next(dataiter)

    # img = images[0].detach().clone()
    # img1 = images[0].detach().clone()
    caps, alphas = get_caps_from(img.unsqueeze(0))
    print(f"this is {name_list}")
    print(caps)






