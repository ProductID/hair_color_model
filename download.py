# In this file, we define download_model
# It runs during container build time to get model weights built intorch.device("cpu")to the container

# In this example: A Huggingface BERT model

#from transformers import pipeline
import torch


def download_model():
    # do a dry run of loading the model, which will download weights
    device = torch.device( 0 if torch.cuda.is_available() else 1)
    # try:
    #     # device = torch.device(torch.cuda.is_available())
    #     # device = torch.device( 0 if torch.cuda.is_available() else 1)
    # except:
    #     device = torch.device("cpu")
    # # device = torch.device( 0 if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load('Barbershop/pretrained_models/ffhq.pt')
    checkpoint_sh = torch.load('Barbershop/pretrained_models/seg.pth')


if __name__ == "__main__":
    download_model()
