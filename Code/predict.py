from PIL import Image

from arcface import Arcface

if __name__ == "__main__":
    model = Arcface()
        
    #----------------------------------------------------------------------------------------------------------#
    #   mode is used to specify the testing mode:
    #   'predict' means single image prediction. If you want to modify the prediction process, such as saving images
    #   or cropping objects, you can check the detailed comments below.
    #   'fps' means testing fps, using the image in the img folder. See comments below for details.
    #----------------------------------------------------------------------------------------------------------#
    mode            = "predict"
    #-------------------------------------------------------------------------#
    #   test_interval   used to specify the number of image detections when measuring fps
    #                   theoretically, the larger the test_interval, the more accurate the fps.
    #   fps_test_image  image for fps testing
    #-------------------------------------------------------------------------#
    test_interval   = 100
    # fps_test_image  = r"./data/test/0001.BMP"
    
    if mode == "predict":
        while True:
            image_1 = r"./data/test/test1.BMP"
            try:
                image_1 = Image.open(image_1)
            except:
                print('Image_1 Open Error! Try again!')
                continue

            image_2 = r"./data/test/test3.BMP"
            try:
                image_2 = Image.open(image_2)
            except:
                print('Image_2 Open Error! Try again!')
                continue
            
            probability = model.detect_image(image_1,image_2)
            print(probability)
            break

    elif mode == "fps":
        img = Image.open(fps_test_image)
        tact_time = model.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')