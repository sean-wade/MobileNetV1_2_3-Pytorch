import numpy
from .transforms import *


class TrainAugmentation:
    def __init__(self, size, mean=0, std=1.0):
        """
        Args:
            size: the size the of final image.
            mean: mean pixel value per channel.
        """
        self.mean = mean
        self.size = size
        self.augment = Compose([                                
            Resize(self.size),
            SubtractMeans(self.mean),
            lambda img: (img / std),
            PhotometricDistortV2(),
            ToTensor()
        ])

    def __call__(self, img):
        """

        Args:
            img: the output of cv.imread in RGB layout.
            boxes: boundding boxes in the form of (x1, y1, x2, y2).
            labels: labels of boxes.
        """
        img = numpy.array(img)
        return self.augment(img)


class TestTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Compose([
            #ToPercentCoords(),
            Resize(size),
            SubtractMeans(mean),
            lambda img: (img / std),
            ToTensor(),
        ])

    def __call__(self, image):
        image = numpy.array(image)
        return self.transform(image)


class PredictionTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Compose([
            Resize(size),
            SubtractMeans(mean),
            lambda img: (img / std),
            ToTensor()
        ])

    def __call__(self, image):
        image = numpy.array(image)
        image = self.transform(image)
        return image
