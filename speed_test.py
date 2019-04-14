import os
import cv2
from mtcnn_pytorch.src.detector import Predictor


def get_images():

    images_dir = "mtcnn_pytorch/images"
    images = []

    for file in os.listdir(images_dir):
        source_image = os.path.join(images_dir, file)

        if file.endswith('.jpg'):
            try:
                image = cv2.imread(source_image)
                images.append(image)
            except FileExistsError:
                print('Problem with file {}'.format(source_image))

    return images


def speed_test():

    images = get_images()

    detect_time = 0
    detect_num = 0
    timer = cv2.TickMeter()
    predictor = Predictor()

    # skip first predict because it may go longer
    predictor.predict_bounding_boxes(images[0])

    for image in images:
        image = cv2.resize(image, (640, 480))
        timer.start()
        predictor.predict_bounding_boxes(image)
        timer.stop()

        detect_time += timer.getTimeMilli()
        detect_num += 1

        timer.reset()

    average_time = detect_time / detect_num
    print("average mtcnn prediction_time {} msec".format(average_time))

    return


if __name__ == "__main__":
    speed_test()
