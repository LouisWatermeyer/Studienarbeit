"""Dice Recognition Data Classes Module

This module defines the core data structures used in the dice recognition system.
It contains dataclasses that represent detected dice and their dots (blobs).

Dependencies:
    - dataclasses: For the @dataclass decorator
    - typing: For type annotations
    - cv2: For OpenCV KeyPoint conversion
    
Author: Louis Watermeyer
"""

from dataclasses import dataclass
from typing import Tuple, List
import cv2


@dataclass
class Blob:
    """Represents a detected blob (dot) on a die.

    A blob corresponds to a single dot on a die face.

    Attributes:
        position: (x, y) coordinates of the blob's center in the image
        is_dark: Whether the blob is dark (True) or light (False)
        size: Diameter of the blob in pixels
    """

    position: Tuple[int, int]  # (x, y) coordinates
    is_dark: bool  # True for dark blobs, False for light blobs
    size: float  # Size/diameter of the blob in pixels

    @classmethod
    def from_keypoint(cls, keypoint: cv2.KeyPoint, is_dark: bool) -> "Blob":
        """Create a Blob from an OpenCV KeyPoint.

        This factory method converts OpenCV's KeyPoint objects into our custom Blob dataclass.

        Args:
            keypoint: OpenCV KeyPoint object containing position and size
            is_dark: Whether this blob is dark (True) or light (False)

        Returns:
            A new Blob instance with data from the KeyPoint
        """
        return cls(
            position=keypoint.pt,  # KeyPoint.pt contains (x,y) coordinates
            size=keypoint.size,  # KeyPoint.size contains diameter
            is_dark=is_dark
        )


@dataclass
class Dice:
    """Represents a detected die with its dots and position.

    A die is a collection of blobs (dots) that have been grouped together.
    The number of blobs determines the die's value.

    Attributes:
        position: (x, y) coordinates of the die's center in the image
        blobs: List of blobs (dots) that make up this die's face
    """

    position: Tuple[int, int]  # (x, y) coordinates of die center
    blobs: List[Blob]  # Collection of dots that form this die

    @property
    def value(self) -> int:
        """Get the value of the die (number of dots).

        Returns:
            The number of dots on the die face
        """
        return len(self.blobs)
