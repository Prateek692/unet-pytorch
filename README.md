# Image Segmentation using U-Net PyTorch 

## This is an image segmentation network implemented using U-Net in PyTorch. The repository consists two python scripts, unet_finetune.py and unet_inference.py.

In unet_finetune, the preprocessing steps and training code is given. This can be used to train the Unet model and save the model in a __.pth file__ for future use. The __segmentation-models-pytorch__ python library is used to load the U-Net model. Resnet50 is used as the encoder along with pre-trained weights of Imagenet.  

In unet_inference, the saved model is loaded and tested on the test dataset.
