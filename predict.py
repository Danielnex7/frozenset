import torch
import os
import cv2
import matplotlib
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image
from torchvision.utils import draw_bounding_boxes
from numpy.random import randint
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
from torchvision.transforms._presets import ObjectDetection

matplotlib.use('TkAgg')

def model_chooser():
    # Show choose
    print('Choose model:')
    print('1 - yolo8m')
    print('2 - yolo7n')
    print('3 - yolo5s')
    print('4 - ssdlite')
    print('0 - for exit')

    # ask model
    try:
        model_number = int(input('Input number of model:'))
        if 0 <= model_number < 5:
            print('Your choice: ', model_number)
            return model_number
        else:
            raise ValueError
    except ValueError:
        print('Wrong input. Only numbers from 0 to 4 are available! Try again!')
        return -1

# load model (general choice of loader). Simple YOLO loader
def load_model(num_model):
    if num_model == 1:
        return YOLO(os.path.join('models', 'yolov8m.pt'))
    elif num_model == 2:
        return load_yolo7()
    elif num_model == 3:
        return YOLO(os.path.join('models', 'yolov5s.pt'))
    elif num_model == 4:
        return load_ssdlite()

# additional code for load other non-ultralytics models
def load_yolo7():
    pass

def load_ssdlite():
    # base pretrained model as base template
    model = ssdlite320_mobilenet_v3_large( weights='DEFAULT', score_tresh=0.1)
    # model with needed head
    m_3class = ssdlite320_mobilenet_v3_large( num_classes=3,
                                             weights_backbone=None,
                                             score_tresh=0.1,
                                             )
    # change head for 3 classes: 0 - background (required), 1 - head, 2 - helmet
    model.head = m_3class.head
    # construct path to model
    path_to_model = os.path.join('models', 'ssdlite.pt')
    # get current device
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # load model dict file
    checkpoint = torch.load(path_to_model, map_location=DEVICE)#torch.device(DEVICE))
    # load state dictionary 
    model.load_state_dict(checkpoint['model_state_dict'])
    # set confidence for prediction
    model.score_tresh = 0.4
    # transfer pretrained model to eval state
    model.eval()
    return model


# image chooser
def choose_image():
    flag = True
    while flag:
        print('Input number to choose:')
        print('0 - exit programm')
        print('1 - choose another model')
        print('2 - choose random image from test_images dir')
        img_path = input('Or input full path to image:')

        if os.path.exists(img_path): # if path is correct
            return img_path
        else:
            if img_path == '0':
                return 0
            elif img_path == '1':
                return 1
            elif img_path == '2':
                test_img_lst = os.listdir('test_images')
                img_path = os.path.join('test_images', test_img_lst[randint(0, len(test_img_lst))])
                print(f'Path to image: {img_path}')
                return img_path
            else:
                print('Incorrect input. Try again!')
    

# make prediction depending on model (don't optimized for yolo 1-3 for flexibility)
def make_prediction(img_path, model, num_model):
    # choose prediction method for different models
    if num_model == 1:
        return model.predict(source=img_path, save=False, save_txt=False)
    elif num_model == 2:
        return model.predict(source=img_path, save=False, save_txt=False) # ??? YOLOv7 ???
    elif num_model == 3:
        return model.predict(source=img_path, save=False, save_txt=False)
    elif num_model == 4:
        convert_to_tensor = ObjectDetection()
        img = Image.open(img_path)
        tensor_img = convert_to_tensor(img)
        result = model([tensor_img])[0]
        return result


#draw image (don't optimized for yolo 1-3 for flexibility)
def draw_image(result, img_path, num_model):
    if num_model == 1:
        color_img1 = cv2.cvtColor(result[0].plot(), cv2.COLOR_BGR2RGB)
        plt.imshow(color_img1)
        plt.title('Model 1 - YOLOv8m')
        plt.show()
    
    elif num_model == 2: # ??? YOLOv7 ???
        color_img1 = cv2.cvtColor(result[0].plot(), cv2.COLOR_BGR2RGB)
        plt.imshow(color_img1)
        plt.title('Model 2 - YOLOv7')
        plt.show() 
    
    elif num_model == 3:
        color_img1 = cv2.cvtColor(result[0].plot(), cv2.COLOR_BGR2RGB)
        plt.imshow(color_img1)
        plt.title('Model 3 - YOLOv5s')
        plt.show()
    
    elif num_model == 4:
        labels_dict={'1': 'head', '2': 'helmet'} # labels for classes
        colors_dict={'1': 'red', '2': 'green'} # colors for classes
        img = Image.open(img_path)
        # construct tensor of labels for predicted classes
        labels = [labels_dict[str(label.item())] + ': ' + \
                  str(round(result['scores'][idx].item(), 2)) \
                  for idx, label in enumerate(result['labels'])]
        # construct tensor of colors for prediction boxes colors
        colors = [colors_dict[str(label.item())] for label in result['labels']]
        #constructing image with boxes
        box = draw_bounding_boxes(pil_to_tensor(img),
                                  boxes=result['boxes'],
                                  labels=labels,
                                  colors=colors,
                                  width=3)
        plt.imshow(to_pil_image(box.detach()))
        plt.title('Model 4 - SSDLite')
        plt.show()
        print(result)
    

def main():
    flag = True
    img_flag = True # flag for image choose part
    # programm lifecicle
    while flag: 
        # get model number
        num_model = model_chooser()

        # exit if 0
        if num_model == 0:
            return

        while num_model == -1:
            num_model = model_chooser()
            
            # exit if 0
            if num_model == 0:
                return

        # load chosen model
        model = load_model(num_model)

        # Choose image and make prediction
        while img_flag:
            img_path = choose_image()
            print(img_path)
            # exit if 0 chosen
            if img_path == 0:
                return
            elif img_path == 1: # to choose another model
                break
            # make prediction
            result = make_prediction(img_path, model, num_model)
            # draw image
            draw_image(result, img_path, num_model)

    print('End of programm')


if __name__ == "__main__":
    main()