import os
from time import time
from PIL import Image
from mtcnn_pytorch.src import detect_faces


def get_images():

    images_dir = "mtcnn_pytorch/images"
    images = []

    for file in os.listdir(images_dir):

        source_image = os.path.join(images_dir, file)

        if file.endswith('.jpg'):
            try:
                image = Image.open(source_image)
                images.append(image)
            except FileExistsError:
                print('Problem with file {}'.format(source_image))

    return images


def speed_test():

    images = get_images()

    detect_time = 0
    detect_num = 0

    for image in images:
        start_time = time()
        detect_faces(image)
        end_time = time()

        detect_time += (end_time - start_time)
        detect_num += 1

    average_time = detect_time / detect_num
    print("average mtcnn prediction_time {}".format(average_time))

    return


if __name__ == "__main__":
    speed_test()