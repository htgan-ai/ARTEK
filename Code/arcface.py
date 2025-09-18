import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from nets.arcface import Arcface as arcface
from utils.utils import preprocess_input, resize_image, show_config


class Arcface(object):
    _defaults = {
        #--------------------------------------------------------------------------#
        #   To use your trained model for prediction, modify model_path to point to the weight file in the logs folder.
        #   After training, there will be multiple weight files in the logs folder. Choose the one with lower validation loss.
        #   Lower validation loss doesn't necessarily mean higher accuracy, it only indicates better generalization on the validation set.
        #--------------------------------------------------------------------------#

        "model_path"        : r"./weights/model_weights.pth",

        #-------------------------------------------#
        #   Input image size.
        #-------------------------------------------#
        "input_shape"       : [256, 256, 3],
        #-------------------------------------------#
        "backbone"          : "mobilefacenet",
        #-------------------------------------------#
        #   Whether to use distortionless resize
        #-------------------------------------------#
        "letterbox_image"   : True,
        #-------------------------------------------#
        #   Whether to use CUDA
        #   Set to False if no GPU is available
        #-------------------------------------------#
        "cuda"              : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   Initialize Arcface
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        self.generate()
        
        show_config(**self._defaults)
        
    def generate(self):
        # ---------------------------------------------------#
        #   Load model and weights
        # ---------------------------------------------------#
        print('Loading weights into state dict...')
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net    = arcface(num_classes=1000, backbone=self.backbone, mode="predict").eval()
        self.net.load_state_dict(torch.load(self.model_path, map_location=device), strict=False)
        print('{} model loaded.'.format(self.model_path))

        if self.cuda:
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
            self.net = self.net.cuda()
    
    #---------------------------------------------------#
    #   Detect images
    #---------------------------------------------------#
    def detect_image(self, image_1, image_2):
        #---------------------------------------------------#
        #   Image preprocessing and normalization
        #---------------------------------------------------#
        with torch.no_grad():
            image_1 = resize_image(image_1, [self.input_shape[1], self.input_shape[0]], letterbox_image=self.letterbox_image)
            image_2 = resize_image(image_2, [self.input_shape[1], self.input_shape[0]], letterbox_image=self.letterbox_image)
            
            photo_1 = torch.from_numpy(np.expand_dims(np.transpose(preprocess_input(np.array(image_1, np.float32)), (2, 0, 1)), 0))
            photo_2 = torch.from_numpy(np.expand_dims(np.transpose(preprocess_input(np.array(image_2, np.float32)), (2, 0, 1)), 0))
            
            if self.cuda:
                photo_1 = photo_1.cuda()
                photo_2 = photo_2.cuda()
                
            #---------------------------------------------------#
            #   Pass images through the network for prediction
            #---------------------------------------------------#
            output1 = self.net(photo_1).cpu().numpy()
            output2 = self.net(photo_2).cpu().numpy()
            
            #---------------------------------------------------#
            #   Calculate the distance between the two outputs
            #---------------------------------------------------#
            l1 = np.linalg.norm(output1 - output2, axis=1)
        
        plt.subplot(1, 2, 1)
        plt.imshow(np.array(image_1))

        plt.subplot(1, 2, 2)
        plt.imshow(np.array(image_2))
        plt.text(-12, -12, 'Distance:%.3f' % l1, ha='center', va= 'bottom',fontsize=11)
        plt.show()
        return l1

    def get_FPS(self, image, test_interval):

        image_data  = resize_image(image, [self.input_shape[1], self.input_shape[0]], self.letterbox_image)

        image_data  = torch.from_numpy(np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0))
        with torch.no_grad():

            preds = self.net(image_data).cpu().numpy()
            
        import time
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():

                preds = self.net(image_data).cpu().numpy()
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time
