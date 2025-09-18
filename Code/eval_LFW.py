import torch
import torch.backends.cudnn as cudnn

from nets.arcface import Arcface
from utils.dataloader import LFWDataset
from utils.utils_metrics import test


if __name__ == "__main__":
    #--------------------------------------#
    #   Whether to use Cuda
    #   Set to False if no GPU is available
    #--------------------------------------#
    cuda            = True
    #--------------------------------------#
    #   Selection of backbone feature extraction network
    #   mobilefacenet
    #   mobilenetv1
    #   iresnet18
    #   iresnet34
    #   iresnet50
    #   iresnet100
    #   iresnet200
    #--------------------------------------#
    backbone        = "mobilefacenet"
    #--------------------------------------#
    #   Input image size
    #--------------------------------------#
    input_shape     = [256, 256, 3]
    #--------------------------------------#
    #   Trained weights file
    #--------------------------------------#
    model_path      = "./weights/arcface_mobilefacenet.pth"
    #--------------------------------------#
    #   LFW evaluation dataset file path
    #   and corresponding txt file
    #--------------------------------------#

    lfw_dir_path    = "./data/lfw"
    lfw_pairs_path  = "./data/lfw_pair.txt"
    
    #--------------------------------------#
    #   Evaluation batch size and log interval
    #--------------------------------------#
    batch_size      = 256
    log_interval    = 1
    #--------------------------------------#
    #   ROC curve save path
    #--------------------------------------#
    png_save_path   = "./results/roc_test.png"

    test_loader = torch.utils.data.DataLoader(
        LFWDataset(dir=lfw_dir_path, pairs_path=lfw_pairs_path, image_size=input_shape), batch_size=batch_size, shuffle=False)

    model = Arcface(backbone=backbone, mode="predict")

    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model  = model.eval()

    if cuda:
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model = model.cuda()

    test(test_loader, model, png_save_path, log_interval, batch_size, cuda)
