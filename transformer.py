import numpy as np
import cv2


class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        """
        Initializes the ViewTransformer with source and target points.

        :param source: A 2D numpy array (4x2) representing four points in the source plane.
        :param target: A 2D numpy array (4x2) representing four points in the target plane.
        """
        source = source.astype(np.float32)
        target = target.astype(np.float32)  # Fixed the typo here
        self.m = cv2.getPerspectiveTransform(
            source, target
        )  # Compute perspective matrix

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Transforms a set of points using the computed perspective transformation matrix.

        :param points: A 2D numpy array (Nx2) representing points in the source plane.
        :return: A 2D numpy array (Nx2) representing the transformed points in the target plane.
        """
        reshaped_points = points.reshape(-1, 1, 2).astype(
            np.float32
        )  # Reshape points for OpenCV
        transformed_points = cv2.perspectiveTransform(
            reshaped_points, self.m
        )  # Apply transform
        return transformed_points.reshape(-1, 2)  # Reshape back to original format
