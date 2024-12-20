import torch
import segmentation_models_pytorch as smp
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

# Enable CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 4
original_height = 1216
original_width = 1936
print(device)

def pad_to_divisible(image, divisor=32):
    height, width = image.shape[1:3]
    new_height = ((height + divisor - 1) // divisor) * divisor
    new_width = ((width + divisor - 1) // divisor) * divisor
    pad_height = new_height - height
    pad_width = new_width - width

    padded_image = torch.nn.functional.pad(image, (0, pad_width, 0, pad_height), mode='constant', value=0)

    return padded_image


def preprocessing(image_path):
    image = Image.open(image_path)

    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    input_tensor = preprocess(image)
    # input_tensor = pad_to_divisible(input_tensor)
    input_tensor = input_tensor.unsqueeze(0)

    return input_tensor

# def postprocessing(image):

#     postprocess = transforms.Compose([
#         transforms.Resize((1216, 1936))
#     ])

#     image = postprocess(image)
#     return image

# class_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]

# def postprocessing(image):

#     postprocess = transforms.Compose([
#         transforms.Resize((1216, 1936))
#     ])
    
#     image = postprocess(image)
#     return image

# def get_colored_segmentation_image(seg_arr, n_classes, colors=class_colors):

#     seg_arr = seg_arr.squeeze(0).cpu().numpy()
#     # seg_arr = np.argmax(seg_arr, axis=0)

#     # output_height = seg_arr.shape[0]
#     # output_width = seg_arr.shape[1]

#     # seg_img = np.zeros((output_height, output_width, 3))

#     rgb_overlay = np.zeros((*seg_arr[0].shape, 3), dtype=float)

#     for i in range(n_classes):
        
#         rgb_overlay += seg_arr[i][:, :, np.newaxis] * colors[i]

#         # seg_img[:, :, 0] += ((seg_arr_c)*(colors[c][0])).astype('uint8')
#         # seg_img[:, :, 1] += ((seg_arr_c)*(colors[c][1])).astype('uint8')
#         # seg_img[:, :, 2] += ((seg_arr_c)*(colors[c][2])).astype('uint8')
#     rgb_overlay = np.clip(rgb_overlay, 0, 1)

#     return rgb_overlay

# def visualize_segmentation(seg_arr, inp_img=None, n_classes=None,
#                            colors=class_colors, class_names=None,
#                            overlay_img=False, show_legends=False,
#                            prediction_width=None, prediction_height=None):

#     if n_classes is None:
#         n_classes = np.max(seg_arr)

#     seg_img = get_colored_segmentation_image(seg_arr, n_classes, colors=colors)

#     if inp_img is not None:
#         original_h = inp_img.shape[0]
#         original_w = inp_img.shape[1]
#         seg_img = cv2.resize(seg_img, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

#     if (prediction_height is not None) and (prediction_width is not None):
#         seg_img = cv2.resize(seg_img, (prediction_width, prediction_height), interpolation=cv2.INTER_NEAREST)
#         if inp_img is not None:
#             inp_img = cv2.resize(inp_img, (prediction_width, prediction_height))

#     # if overlay_img:
#     #     assert inp_img is not None
#     #     seg_img = overlay_seg_image(inp_img, seg_img)

#     # if show_legends:
#     #     assert class_names is not None
#     #     legend_img = get_legends(class_names, colors=colors)

#     #     seg_img = concat_lenends(seg_img, legend_img)

#     return seg_img

def visualize_class_channels(tensor):
    # Remove batch dimension and convert to numpy
    # print(tensor.shape)
    # resize_operation = transforms.Resize((1216, 1936))
    # tensor = resize_operation(tensor)
    tensor = tensor.squeeze(0).cpu().numpy()
    # print(tensor.shape)
    # tensor = np.resize(tensor, (3,1216,1936))
    
    # Normalize each channel to [0, 1]
    normalized_channels = []
    for channel in tensor:
        channel = (channel - channel.min()) / (channel.max() - channel.min())
        normalized_channels.append(channel)
    
    # Create an RGB image by combining channels
    # You can experiment with different color mappings
    rgb_image = np.zeros((*normalized_channels[0].shape, 3), dtype=float)
    # print(rgb_image.shape)
    # Assign different colors to different channels
    # Ensure you have at least 3 channels, or repeat channels if fewer
    color_map = [
        [0, 0, 0],   # Red for first channel
        [1, 0, 0],   # Green for second channel
        [0, 1, 0],   # Blue for third channel
        [0, 0, 1]    # Yellow for fourth channel
    ]
    
    for i, channel in enumerate(normalized_channels):
        # if i < len(color_map):
        # print(i)
        rgb_image += channel[:, :, np.newaxis] * color_map[i]
    
    # Clip values to ensure they are in [0, 1]
    # print("unique pix vals before clip", np.unique(rgb_image))
    # rgb_image = np.clip(rgb_image, 0, 1)
    # print("unique pix vals after clip", np.unique(rgb_image))
    return rgb_image
    # Plot the image
    # plt.figure(figsize=(10, 8))
    # plt.imshow(rgb_image)
    # plt.title('Multi-Channel Segmentation Overlay')
    # plt.axis('off')
    # plt.show()

# def crop_to_original_size(output, original_size):
#     return output[:original_size[0], :original_size[1]]

# Load the model with pre-trained weights
model = smp.Unet('resnet50', encoder_weights='imagenet', classes=num_classes, activation='softmax', in_channels=1, encoder_depth=5)
model.load_state_dict(torch.load("unet_finetune_test.pth", weights_only=True))
model.eval()  # Set the model to evaluation mode
model.to(device)

# Load and preprocess the image
input_dir = r"training_data/train_1/images"
output_dir = r"training_data/train_1/outputs"

imgs = os.listdir(input_dir)

for i in imgs: 
    # img = Image.open(os.path.join(input_dir, i)).convert("RGB")
    input_tensor = preprocessing(os.path.join(input_dir, i))
    # input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
    input_tensor = input_tensor.to(device)

    with torch.no_grad():  # Disable gradient calculation for inference
        output = model(input_tensor)  # Get model predictions
        # print(output.shape)
    print(torch.unique(output))
    # output = output.cpu().numpy()
    output_mask = visualize_class_channels(output)
    # print(np.shape(output_mask))
    # output_mask = postprocessing(output_mask)
    # print(np.shape(output_mask))
    # print(np.max(output_mask))
    # print(np.min(output_mask))
    # output_mask = Image.fromarray(output_mask.astype(np.uint8))
    # output_mask = crop_to_original_size(output_mask, (original_height, original_width))    
    # output_mask = output_mask
    # output_mask = cv2.resize(output_mask, (1936,1216), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(output_dir, i), output_mask*63)
    # output_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    # Convert mask to an image and save it
    # output_image = Image.fromarray(output_mask.astype(np.uint8))  # Scale mask to [0, 255]
    # output_image.save(os.path.join(output_dir, i))
    # print("Saved mask: ", os.path.join(output_dir, i))