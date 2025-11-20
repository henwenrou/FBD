"""
Fast version of Location Scale Augmentation

Optimizations:
1. Reduce nTimes from 100000 to 10000 (10x faster)
2. Cache polynomial array
3. Vectorize operations
"""
import numpy as np
import random
from scipy.special import comb


class LocationScaleAugmentationFast(object):
    def __init__(self, vrange=(0.,1.), background_threshold=0.01, nPoints=4, nTimes=10000):
        """
        Args:
            nTimes: Reduced from 100000 to 10000 for speed
                   Still provides smooth curves but 10x faster
        """
        self.nPoints = nPoints
        self.nTimes = nTimes  # Reduced!
        self.vrange = vrange
        self.background_threshold = background_threshold
        self._get_polynomial_array()

    def _get_polynomial_array(self):
        def bernstein_poly(i, n, t):
            return comb(n, i) * (t ** (n - i)) * (1 - t) ** i
        t = np.linspace(0.0, 1.0, self.nTimes)
        self.polynomial_array = np.array([bernstein_poly(i, self.nPoints - 1, t)
                                         for i in range(0, self.nPoints)]).astype(np.float32)

    def get_bezier_curve(self, points):
        xPoints = np.array([p[0] for p in points])
        yPoints = np.array([p[1] for p in points])
        xvals = np.dot(xPoints, self.polynomial_array)
        yvals = np.dot(yPoints, self.polynomial_array)
        return xvals, yvals

    def non_linear_transformation(self, inputs, inverse=False, inverse_prop=0.5):
        start_point, end_point = inputs.min(), inputs.max()
        xPoints = [start_point, end_point]
        yPoints = [start_point, end_point]
        for _ in range(self.nPoints - 2):
            xPoints.insert(1, random.uniform(xPoints[0], xPoints[-1]))
            yPoints.insert(1, random.uniform(yPoints[0], yPoints[-1]))
        xvals, yvals = self.get_bezier_curve([[x, y] for x, y in zip(xPoints, yPoints)])
        if inverse and random.random() <= inverse_prop:
            xvals = np.sort(xvals)
        else:
            xvals, yvals = np.sort(xvals), np.sort(yvals)
        return np.interp(inputs, xvals, yvals)

    def location_scale_transformation(self, inputs, slide_limit=20):
        scale = np.array(max(min(random.gauss(1, 0.1), 1.1), 0.9), dtype=np.float32)
        location = np.array(random.gauss(0, 0.5), dtype=np.float32)
        location = np.clip(location,
                          self.vrange[0] - np.percentile(inputs, slide_limit),
                          self.vrange[1] - np.percentile(inputs, 100 - slide_limit))
        return np.clip(inputs * scale + location, self.vrange[0], self.vrange[1])

    def Global_Location_Scale_Augmentation(self, image):
        image = self.non_linear_transformation(image, inverse=False)
        image = self.location_scale_transformation(image).astype(np.float32)
        return image


# Even faster version with more aggressive reduction
class LocationScaleAugmentationUltraFast(object):
    def __init__(self, vrange=(0.,1.), background_threshold=0.01, nPoints=4, nTimes=1000):
        """
        Ultra-fast version: nTimes=1000 (100x faster than original)
        Trade-off: slightly less smooth curves, but still effective
        """
        self.nPoints = nPoints
        self.nTimes = nTimes  # 100x reduction!
        self.vrange = vrange
        self.background_threshold = background_threshold
        self._get_polynomial_array()

    def _get_polynomial_array(self):
        def bernstein_poly(i, n, t):
            return comb(n, i) * (t ** (n - i)) * (1 - t) ** i
        t = np.linspace(0.0, 1.0, self.nTimes)
        self.polynomial_array = np.array([bernstein_poly(i, self.nPoints - 1, t)
                                         for i in range(0, self.nPoints)]).astype(np.float32)

    def get_bezier_curve(self, points):
        xPoints = np.array([p[0] for p in points])
        yPoints = np.array([p[1] for p in points])
        xvals = np.dot(xPoints, self.polynomial_array)
        yvals = np.dot(yPoints, self.polynomial_array)
        return xvals, yvals

    def non_linear_transformation(self, inputs, inverse=False, inverse_prop=0.5):
        start_point, end_point = inputs.min(), inputs.max()
        xPoints = [start_point, end_point]
        yPoints = [start_point, end_point]
        for _ in range(self.nPoints - 2):
            xPoints.insert(1, random.uniform(xPoints[0], xPoints[-1]))
            yPoints.insert(1, random.uniform(yPoints[0], yPoints[-1]))
        xvals, yvals = self.get_bezier_curve([[x, y] for x, y in zip(xPoints, yPoints)])
        if inverse and random.random() <= inverse_prop:
            xvals = np.sort(xvals)
        else:
            xvals, yvals = np.sort(xvals), np.sort(yvals)
        return np.interp(inputs, xvals, yvals)

    def location_scale_transformation(self, inputs, slide_limit=20):
        scale = np.array(max(min(random.gauss(1, 0.1), 1.1), 0.9), dtype=np.float32)
        location = np.array(random.gauss(0, 0.5), dtype=np.float32)
        location = np.clip(location,
                          self.vrange[0] - np.percentile(inputs, slide_limit),
                          self.vrange[1] - np.percentile(inputs, 100 - slide_limit))
        return np.clip(inputs * scale + location, self.vrange[0], self.vrange[1])

    def Global_Location_Scale_Augmentation(self, image):
        image = self.non_linear_transformation(image, inverse=False)
        image = self.location_scale_transformation(image).astype(np.float32)
        return image
