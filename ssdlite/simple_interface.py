import torch
import os

from PIL import Image
from torchvision.utils import draw_bounding_boxes
from numpy.random import randint
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
from torchvision.transforms._presets import ObjectDetection

# set relative path
ds_path = 'VOC2028/'

# create model with head for 3 classes(0 - background, 1 - head, 2 - helmet)
#base model 
model_loaded = ssdlite320_mobilenet_v3_large( weights='DEFAULT',
                                             score_tresh=0.1
                                             )
# model with needed head
m_3class = ssdlite320_mobilenet_v3_large( num_classes=3,
                                          weights_backbone=None,
                                          score_tresh=0.1,
                                        )
# change head
model_loaded.head = m_3class.head

# ordinary path to best model
best_path = 'models/' + 'SSDLiteMobNetFreezBackbone_3class_best(49ep).pt'

# ask path to model
model_path = input('Input path of model to load:')

# check existence of path
if not os.path.exists(model_path):
    print('Wrong path. The best default model will be loaded:', best_path)
    model_path = best_path

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

checkpoint = torch.load(model_path, map_location=torch.device(DEVICE))
model_loaded.load_state_dict(checkpoint['model_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
model_loaded.eval()

print('Model trained', epoch, 'epoches')
print(f'Boxes loss: {round(loss["bbox_loss"].item(),4)}, class loss: {round(loss["cls_loss"].item(),4)}')

#preparation to model inference
convert_to_tensor = ObjectDetection()
model_loaded.score_thresh = 0.6
labels_dict={'1': 'head',
             '2': 'helmet'}
colors_dict={'1': 'red',
             '2': 'green'}

# get path to image in infinite circle
flag = True
while flag:
    print()
    img_path = input('Input full path to image for prediction (to exi input 0):')
    if img_path == '0':
        flag = False
        break
    if not os.path.exists(img_path):
        lst = os.listdir('short_test/')
        img_path = 'short_test/' + lst[randint(0, len(lst))]
        print('Wrong path. The random image from standard folder will be loaded:', img_path)

    img = Image.open(img_path)
    tensor_img = convert_to_tensor(img)
    prediction = model_loaded([tensor_img])[0]
    
    labels = [labels_dict[str(label.item())] + ': ' + \
          str(round(prediction['scores'][idx].item(), 2)) \
          for idx, label in enumerate(prediction['labels'])]
    colors = [colors_dict[str(label.item())] for label in prediction['labels']]
    
    #constructing image with boxes
    box = draw_bounding_boxes(pil_to_tensor(img), # for original image case
                              boxes=prediction['boxes'],
                              labels=labels,
                              colors=colors,
                              width=3)
    
    im = to_pil_image(box.detach())
    im.show()
    print(prediction)