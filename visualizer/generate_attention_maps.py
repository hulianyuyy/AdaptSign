#Ref: https://blog.csdn.net/weixin_41735859/article/details/106474768
import numpy as np
import os
import glob
import cv2
from utils import video_augmentation
from slr_network import SLRModel
import torch
from collections import OrderedDict
import utils
import torchvision.transforms as T
import json
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shutil
from visualizer import get_local
get_local.activate()

# The tutorial is https://zhuanlan.zhihu.com/p/398408338, with a demo at https://nbviewer.org/github/luo3300612/Visualizer/blob/main/demo.ipynb
# This file holds the implementation to show the attention maps in a vit. You should prepare 
# 1. a model file, with the 'get_local' descriptor before the forward function to get the values of the forwarded attention maps ;
# 2. this file. It has three modes: 
#    (1) generate attention maps for a signle image defined by with a single grid defined by grid_index;
#    (2) generate attention maps for all images with a single grid defined by grid_index;
#    (3) generate attention maps for all images with all grid-to-grid attention maps.
# Below are the hyper-parameters you should choose before running

#model_weights = './work_dir/baseline_ViT-B32_dist_25_adapter_parallel/_best_model.pt'
model_weights = '../FrozenCLIP/work_dir/baseline_ViT-B32_dist_25/_best_model.pt'
select_id = 1 # The video selected to show. 539: 31October_2009_Saturday_tagesschau_default-8.  0: 01April_2010_Thursday_heute_default-1
# 1: 01August_2011_Monday_heute_default-6, 2: 01December_2011_Thursday_heute_default-3
grid_size = 7 # 7 for ViT-B/32, 14 for ViT-B/16
gpu_id = 0  # the gpu to run this file
image_id = 14  # the index of selected image to show in mode 1 & 2
layer_id = 1  # attention maps of which layer to genearte. 0-11 for ViT-B/32 & ViT-B/16
grid_index = 26  # the index of which grid to show. 0 for cls token, 10-11 around head, 24 around right hand, 26 around left hand
mode = 3 # Three modes of this file, introduced as above.
work_dir = './attention_maps/'  # the path to store attention maps

if os.path.exists(work_dir):
    shutil.rmtree(work_dir)
    os.makedirs(work_dir)
else:
    os.makedirs(work_dir)
#head_id = 1
#name = '01April_2010_Thursday_heute_default-1'

# Load data and apply transformation
dataset = 'phoenix2014'
prefix = './dataset/phoenix2014/phoenix-2014-multisigner'
dict_path = f'./preprocess/{dataset}/gloss_dict.npy'
gloss_dict = np.load(dict_path, allow_pickle=True).item()
inputs_list = np.load(f"./preprocess/{dataset}/dev_info.npy", allow_pickle=True).item()
name = inputs_list[select_id]['fileid']
print(f'Generating attention weights for {grid_index}th grid of {image_id}th image for video {name}')

img_folder = os.path.join(prefix, "features/fullFrame-256x256px/" + inputs_list[select_id]['folder']) if 'phoenix' in dataset else os.path.join(prefix, "features/fullFrame-256x256px/" + inputs_list[select_id]['folder'] + "/*.jpg")
img_list = sorted(glob.glob(img_folder))
img_list = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in img_list]
label_list = []
for phase in inputs_list[select_id]['label'].split(" "):
    if phase == '':
        continue
    if phase in gloss_dict.keys():
        label_list.append(gloss_dict[phase][0])
transform = video_augmentation.Compose([
                video_augmentation.CenterCrop(224),
                video_augmentation.Resize(1.0),
                video_augmentation.ToTensor(),
            ])
vid, label = transform(img_list, label_list, None)
vid = vid.float() / 127.5 - 1
vid = vid.unsqueeze(0)
#vid = vid[:,image_id:image_id+1] #btchw

left_pad = 0
last_stride = 1
total_stride = 1
kernel_sizes = ['K5', "P2", 'K5', "P2"]
for layer_idx, ks in enumerate(kernel_sizes):
    if ks[0] == 'K':
        left_pad = left_pad * last_stride 
        left_pad += int((int(ks[1])-1)/2)
    elif ks[0] == 'P':
        last_stride = int(ks[1])
        total_stride = total_stride * last_stride

max_len = vid.size(1)
video_length = torch.LongTensor([np.ceil(vid.size(1) / total_stride) * total_stride + 2*left_pad ])
right_pad = int(np.ceil(max_len / total_stride)) * total_stride - max_len + left_pad
max_len = max_len + left_pad + right_pad
vid = torch.cat(
    (
        vid[0,0][None].expand(left_pad, -1, -1, -1),
        vid[0],
        vid[0,-1][None].expand(max_len - vid.size(1) - left_pad, -1, -1, -1),
    )
    , dim=0).unsqueeze(0)

device = utils.GpuDataParallel()
device.set_device(gpu_id)
# Define model and load state-dict
model = SLRModel( num_classes=1296, c2d_type='ViT-B/32', conv_type=2, use_bn=1, gloss_dict=gloss_dict,
            loss_weights={'ConvCTC': 1.0, 'SeqCTC': 1.0, 'Dist': 25.0},   )
state_dict = torch.load(model_weights)['model_state_dict']
state_dict = OrderedDict([(k.replace('.module', ''), v) for k, v in state_dict.items()])
model.load_state_dict(state_dict, strict=True)

model = model.to(device.output_device)
model.cuda()

model.eval()

print(f'Input {len(img_list)} images')
print('Video size : ' + str(vid.shape) +'  after padding.')
vid = device.data_to_device(vid)
vid_lgt = device.data_to_device(video_length)
label = device.data_to_device([torch.LongTensor(label)])
label_lgt = device.data_to_device(torch.LongTensor([len(label_list)]))
ret_dict = model(vid, vid_lgt, label=label, label_lgt=label_lgt)

cache = get_local.cache
attention_maps = cache['ResidualAttentionBlock.forward']
grid_num = int(attention_maps[0].shape[1])  # Include cls
print(f'there are layers of {len(attention_maps)} attention maps')
print(f'there are {attention_maps[0].shape[0]} frames with spatial indice {attention_maps[0].shape[1]} *{attention_maps[0].shape[2]} in each layer')

del model
del vid
del vid_lgt
del label
del label_lgt
del ret_dict
del cache
torch.cuda.empty_cache()

#print(attention_maps[layer_id][image_id + left_pad, grid_index])
def grid_show(to_shows, cols):
    rows = (len(to_shows)-1) // cols + 1
    it = iter(to_shows)
    fig, axs = plt.subplots(rows, cols, figsize=(rows*8.5, cols*2))
    for i in range(rows):
        for j in range(cols):
            try:
                image, title = next(it)
            except StopIteration:
                image = np.zeros_like(to_shows[0][0])
                title = 'pad'
            axs[i, j].imshow(image)
            axs[i, j].set_title(title)
            axs[i, j].set_yticks([])
            axs[i, j].set_xticks([])
    #plt.show()

def visualize_head(att_map):
    ax = plt.gca()
    # Plot the heatmap
    im = ax.imshow(att_map)
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    #plt.show()
    
def visualize_heads(att_map, cols):
    to_shows = []
    att_map = att_map.squeeze()
    for i in range(att_map.shape[0]):
        to_shows.append((att_map[i], f'Head {i}'))
    average_att_map = att_map.mean(axis=0)
    to_shows.append((average_att_map, 'Head Average'))
    grid_show(to_shows, cols=cols)

def gray2rgb(image):
    return np.repeat(image[...,np.newaxis],3,2)
    
def cls_padding(image, mask, cls_weight, grid_size):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
        
    image = np.array(image)

    H, W = image.shape[:2]
    delta_H = int(H/grid_size[0])
    delta_W = int(W/grid_size[1])
    
    padding_w = delta_W
    padding_h = H
    padding = np.ones_like(image) * 255
    padding = padding[:padding_h, :padding_w]
    
    padded_image = np.hstack((padding,image))
    padded_image = Image.fromarray(padded_image)
    draw = ImageDraw.Draw(padded_image)
    draw.text((int(delta_W/4),int(delta_H/4)),'CLS', fill=(0,0,0)) # PIL.Image.size = (W,H) not (H,W)

    mask = mask / max(np.max(mask),cls_weight)
    cls_weight = cls_weight / max(np.max(mask),cls_weight)
    
    if len(padding.shape) == 3:
        padding = padding[:,:,0]
        padding[:,:] = np.min(mask)
    mask_to_pad = np.ones((1,1)) * cls_weight
    mask_to_pad = Image.fromarray(mask_to_pad)
    mask_to_pad = mask_to_pad.resize((delta_W, delta_H))
    mask_to_pad = np.array(mask_to_pad)

    padding[:delta_H,  :delta_W] = mask_to_pad
    padded_mask = np.hstack((padding, mask))
    padded_mask = padded_mask
    
    meta_mask = np.zeros((padded_mask.shape[0], padded_mask.shape[1],4))
    meta_mask[delta_H:,0: delta_W, :] = 1 
    
    return padded_image, padded_mask, meta_mask
    

def visualize_grid_to_grid_with_cls(att_map, grid_index, image, grid_size=14, alpha=0.6, save_name=None):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
    
    attention_map = att_map[grid_index]
    cls_weight = attention_map[0]
    
    mask = attention_map[1:].reshape(grid_size[0], grid_size[1])
    mask = Image.fromarray(mask).resize((image.size))
    
    padded_image ,padded_mask, meta_mask = cls_padding(image, mask, cls_weight, grid_size)
    
    if grid_index != 0: # adjust grid_index since we pad our image
        grid_index = grid_index + (grid_index-1) // grid_size[1]
        
    grid_image = highlight_grid(padded_image, [grid_index], (grid_size[0], grid_size[1]+1))
    
    fig, ax = plt.subplots(1, 2, figsize=(10,7))
    fig.tight_layout()
    
    ax[0].imshow(grid_image)
    ax[0].axis('off')
    
    #ax[1].imshow(grid_image)  # draw a rectangle representing the reference point
    ax[1].imshow(padded_image) # don't draw a rectangle representing the reference point
    ax[1].imshow(padded_mask, alpha=alpha, cmap='rainbow')
    ax[1].imshow(meta_mask)
    ax[1].axis('off')
    #plt.show()
    if save_name == None:
        plt.savefig('attention_maps_vit_b_32.png')
    else:
        plt.savefig(save_name)
    plt.close()
    

def visualize_grid_to_grid(att_map, grid_index, image, grid_size=14, alpha=0.6):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
    
    H,W = att_map.shape
    with_cls_token = False
      
    grid_image = highlight_grid(image, [grid_index], grid_size)
    
    mask = att_map[grid_index].reshape(grid_size[0], grid_size[1])
    mask = Image.fromarray(mask).resize((image.size))
    
    fig, ax = plt.subplots(1, 2, figsize=(10,7))
    fig.tight_layout()
    
    ax[0].imshow(grid_image)
    ax[0].axis('off')
    
    ax[1].imshow(grid_image)
    ax[1].imshow(mask/np.max(mask), alpha=alpha, cmap='rainbow')
    ax[1].axis('off')
    #plt.show()
    
def highlight_grid(image, grid_indexes, grid_size=14):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
    
    W, H = image.size
    h = H / grid_size[0]
    w = W / grid_size[1]
    image = image.copy()
    for grid_index in grid_indexes:
        x, y = np.unravel_index(grid_index, (grid_size[0], grid_size[1]))
        a= ImageDraw.ImageDraw(image)
        a.rectangle([(y*w,x*h),(y*w+w,x*h+h)],fill =None,outline ='red',width =2)
    return image
    
if mode==1:
    #show attention map for a single image
    image = cv2.cvtColor(img_list[image_id], cv2.COLOR_RGB2BGR)
    image = Image.fromarray(np.uint8(image))
    visualize_grid_to_grid_with_cls(attention_maps[layer_id][image_id + left_pad,:,:], grid_index, image, grid_size=grid_size) #btchw -> bchw
elif mode == 2:
    # show attention maps for all images with a single grid defined by grid_index
    for image_index in range(len(img_list)):
        image = cv2.cvtColor(img_list[image_index], cv2.COLOR_RGB2BGR)
        image = Image.fromarray(np.uint8(image))
    #visualize_grid_to_grid_with_cls(-attention_maps[layer_id][image_index + left_pad,:,:]+attention_maps[layer_id][image_index + left_pad,:,:].mean(), grid_index, image, grid_size=grid_size, save_name= work_dir + f'vit_b32_image_{image_id}_frame_{image_index}_layer_{layer_id}_grid_{grid_index}.png')
    visualize_grid_to_grid_with_cls(attention_maps[layer_id][image_index + left_pad,:,:], grid_index, image, grid_size=grid_size, save_name= work_dir + f'vit_b32_image_{image_id}_frame_{image_index}_layer_{layer_id}_grid_{grid_index}.png')
elif mode == 3:
    # show attention maps for all images with all grid-to-grid attention maps. The resulted attention maps are stored into different subdirs by grid indice
    for image_index in range(len(img_list)):
        image = cv2.cvtColor(img_list[image_index], cv2.COLOR_RGB2BGR)
        image = Image.fromarray(np.uint8(image))
        for grid_id in range(grid_num):
            target_grid_dir = work_dir + f'grid_{grid_id}/'
            if not os.path.exists(target_grid_dir):
                os.makedirs(target_grid_dir)
            visualize_grid_to_grid_with_cls(attention_maps[layer_id][image_index + left_pad,:,:], grid_id, image, grid_size=grid_size, save_name= target_grid_dir + f'vit_b32_image_{image_id}_layer_{layer_id}_frame_{image_index}.png')
else:
    raise ValueError(' only mode = 1, 2, 3 is supported.')