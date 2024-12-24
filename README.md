# Image Segmentation using U-Net PyTorch 

## Here we have implemented an image segmentation network using U-Net. The repository consists of unet_finetune.py, unet_inference.py, and img_seg_annots.py.

In unet_finetune, the preprocessing steps and training code is given. This can be used to train the Unet model and save the model in a __.pth file__ for future use. The __segmentation-models-pytorch__ python library is used to load the U-Net model. Resnet50 is used as the encoder along with pre-trained weights of Imagenet.  

In unet_inference, the saved model is loaded and tested on the test dataset.

In img_seg_annots, the segmentation labels are created for model training.
