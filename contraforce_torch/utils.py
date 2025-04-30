import cv2
import numpy as np
from collections import deque
from .config import STACK_SIZE

def preprocess(frame: np.ndarray) -> np.ndarray:
    """To grayscale, resize to 84×84, and normalize to [0,1]."""
    gray    = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84))
    return resized.astype(np.float32) / 255.0

def init_stack(frame: np.ndarray) -> deque:
    """Initialize a deque of size STACK_SIZE, all copies of first frame."""
    return deque([frame] * STACK_SIZE, maxlen=STACK_SIZE)

def stack_frames(stacked_frames: deque, new_frame: np.ndarray):
    """
    Append a new frame and return both the deque and a (STACK_SIZE,84,84) array.
    """
    stacked_frames.append(new_frame)
    return stacked_frames, np.stack(stacked_frames, axis=0)
