import numpy as np
import torch
from .get_nets import PNet, RNet, ONet
from .box_utils import nms, calibrate_box, get_image_boxes, convert_to_square, _preprocess
import cv2
import os
import math


class Predictor:
    def __init__(self, device='cuda:0') -> None:
        super().__init__()

        # LOAD MODELS
        self.pnet = PNet()
        self.rnet = RNet()
        self.onet = ONet()
        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()

        self.device = device

        self.pnet.to(self.device)
        self.rnet.to(self.device)
        self.onet.to(self.device)

    def __call__(self, image):
        return self.predict_bounding_boxes_and_landmarks(image)

    def predict_bounding_boxes(self,
                               image,
                               min_face_size=20.0,
                               thresholds=None,
                               nms_thresholds=None):

        """
        Arguments:
            image: an instance of cv2 image or path to the image
            min_face_size: a float number.
            thresholds: a list of length 3.
            nms_thresholds: a list of length 3.

        Returns:
            two float numpy arrays of shapes [n_boxes, 4] and [n_boxes, 10],
            bounding boxes and facial landmarks.
        """

        bboxes_and_landmarks = self.predict_bounding_boxes_and_landmarks(image,
                                                                         min_face_size,
                                                                         thresholds=thresholds,
                                                                         nms_thresholds=nms_thresholds)

        return bboxes_and_landmarks[0]

    def predict_bounding_boxes_and_landmarks(self,
                                             image,
                                             min_face_size=20.0,
                                             thresholds=None,
                                             nms_thresholds=None):
        """
        Arguments:
            image: an instance of cv2 image or path to the image
            min_face_size: a float number.
            thresholds: a list of length 3.
            nms_thresholds: a list of length 3.

        Returns:
            two float numpy arrays of shapes [n_boxes, 4] and [n_boxes, 10],
            bounding boxes and facial landmarks.
        """

        if thresholds is None:
            thresholds = [0.6, 0.7, 0.8]
        if nms_thresholds is None:
            nms_thresholds = [0.7, 0.7, 0.7]

        if isinstance(image, str):
            if os.path.exists(image):
                image = cv2.imread(image)
            else:
                raise FileNotFoundError(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # BUILD AN IMAGE PYRAMID
        original_height, original_width = image.shape[:2]
        min_length = min(original_height, original_width)

        min_detection_size = 12
        factor = 0.707  # sqrt(0.5)

        # scales for scaling the image
        scales = []

        # scales the image so that
        # minimum size that we can detect equals to
        # minimum face size that we want to detect
        m = min_detection_size/min_face_size
        min_length *= m

        factor_count = 0
        while min_length > min_detection_size:
            scales.append(m*factor**factor_count)
            min_length *= factor
            factor_count += 1

        # STAGE 1

        # it will be returned
        bounding_boxes = []

        # run P-Net on different scales
        for s in scales:
            boxes = self.run_first_stage(image, scale=s, threshold=thresholds[0])
            bounding_boxes.append(boxes)

        # collect boxes (and offsets, and scores) from different scales
        bounding_boxes = [i for i in bounding_boxes if i is not None]
        if len(bounding_boxes) == 0:
            return [], []

        bounding_boxes = np.vstack(bounding_boxes)

        keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])
        bounding_boxes = bounding_boxes[keep]

        # use offsets predicted by pnet to transform bounding boxes
        bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
        # shape [n_boxes, 5]

        bounding_boxes = convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

        # STAGE 2

        img_boxes = get_image_boxes(bounding_boxes, image, size=24)
        img_boxes = torch.FloatTensor(img_boxes)
        img_boxes = img_boxes.to(self.device)
        output = self.rnet(img_boxes)
        offsets = output[0].data.cpu().numpy()  # shape [n_boxes, 4]
        probs = output[1].data.cpu().numpy()  # shape [n_boxes, 2]

        keep = np.where(probs[:, 1] > thresholds[1])[0]
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]

        keep = nms(bounding_boxes, nms_thresholds[1])
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
        bounding_boxes = convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

        # STAGE 3

        img_boxes = get_image_boxes(bounding_boxes, image, size=48)
        if len(img_boxes) == 0:
            return [], []
        else:
            img_boxes = torch.FloatTensor(img_boxes)
            img_boxes = img_boxes.to(self.device)
            output = self.onet(img_boxes)
            landmarks = output[0].data.cpu().numpy()  # shape [n_boxes, 10]
            offsets = output[1].data.cpu().numpy()  # shape [n_boxes, 4]
            probs = output[2].data.cpu().numpy()  # shape [n_boxes, 2]

            keep = np.where(probs[:, 1] > thresholds[2])[0]
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
            offsets = offsets[keep]
            landmarks = landmarks[keep]

            # compute landmark points
            width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
            height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
            xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
            landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1)*landmarks[:, 0:5]
            landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1)*landmarks[:, 5:10]

            bounding_boxes = calibrate_box(bounding_boxes, offsets)
            keep = nms(bounding_boxes, nms_thresholds[2], mode='min')
            bounding_boxes = bounding_boxes[keep]
            landmarks = landmarks[keep]

            bounding_boxes = self._prepare_bounding_boxes_result(bounding_boxes, original_width, original_height)

            return bounding_boxes, landmarks

    def run_first_stage(self, image, scale, threshold):
        """Run P-Net, generate bounding boxes, and do NMS.

        Arguments:
            image: an instance of PIL.Image.
            net: an instance of pytorch's nn.Module, P-Net.
            scale: a float number,
                scale width and height of the image by this number.
            threshold: a float number,
                threshold on the probability of a face when generating
                bounding boxes from predictions of the net.

        Returns:
            a float numpy array of shape [n_boxes, 9],
                bounding boxes with scores and offsets (4 + 1 + 4).
        """

        # scale the image and convert it to a float array
        height, width = image.shape[:2]
        sw, sh = math.ceil(width * scale), math.ceil(height * scale)
        img = cv2.resize(image, (sw, sh))
        img = np.asarray(img, 'float32')

        img = torch.FloatTensor(_preprocess(img))
        img = img.to(self.device)
        output = self.pnet(img)
        probs = output[1].data.cpu().numpy()[0, 1, :, :]
        offsets = output[0].data.cpu().numpy()
        # probs: probability of a face at each sliding window
        # offsets: transformations to true bounding boxes

        boxes = self._generate_bboxes(probs, offsets, scale, threshold)
        if len(boxes) == 0:
            return None

        keep = nms(boxes[:, 0:5], overlap_threshold=0.5)
        return boxes[keep]

    @staticmethod
    def _generate_bboxes(probs, offsets, scale, threshold):
        """Generate bounding boxes at places
        where there is probably a face.

        Arguments:
            probs: a float numpy array of shape [n, m].
            offsets: a float numpy array of shape [1, 4, n, m].
            scale: a float number,
                width and height of the image were scaled by this number.
            threshold: a float number.

        Returns:
            a float numpy array of shape [n_boxes, 9]
        """

        # applying P-Net is equivalent, in some sense, to
        # moving 12x12 window with stride 2
        stride = 2
        cell_size = 12

        # indices of boxes where there is probably a face
        inds = np.where(probs > threshold)

        if inds[0].size == 0:
            return np.array([])

        # transformations of bounding boxes
        tx1, ty1, tx2, ty2 = [offsets[0, i, inds[0], inds[1]] for i in range(4)]
        # they are defined as:
        # w = x2 - x1 + 1
        # h = y2 - y1 + 1
        # x1_true = x1 + tx1*w
        # x2_true = x2 + tx2*w
        # y1_true = y1 + ty1*h
        # y2_true = y2 + ty2*h

        offsets = np.array([tx1, ty1, tx2, ty2])
        score = probs[inds[0], inds[1]]

        # P-Net is applied to scaled images
        # so we need to rescale bounding boxes back
        bounding_boxes = np.vstack([
            np.round((stride * inds[1] + 1.0) / scale),
            np.round((stride * inds[0] + 1.0) / scale),
            np.round((stride * inds[1] + 1.0 + cell_size) / scale),
            np.round((stride * inds[0] + 1.0 + cell_size) / scale),
            score, offsets
        ])
        # why one is added?

        return bounding_boxes.T

    @staticmethod
    def _prepare_bounding_boxes_result(bounding_boxes, image_width, image_height):
        result = []
        for i in range(0, bounding_boxes.shape[0]):
            confidence = bounding_boxes[i, 4]

            (x1, y1, x2, y2) = bounding_boxes[i, :4].astype('int')

            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0

            if x2 > image_width - 1:
                x2 = image_width - 1
            if y2 > image_height - 1:
                y2 = image_height - 1

            if x1 < x2 and y1 < y2:
                result += [[x1, y1, x2, y2, confidence]]

        return result
