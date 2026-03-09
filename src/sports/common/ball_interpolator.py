"""
Real-time Ball Interpolation with Fixed-Size Buffer
====================================================

This module implements a single-pass video processing approach with delayed output
to enable real-time interpolation of missing ball detections.

Architecture:
-------------
1. Fixed-size buffer (collections.deque) stores frames + detection results
2. Main loop: read frame N → detect → store in buffer
3. Interpolation logic: analyze buffer content, fill gaps between valid detections
4. Output logic: when buffer is full, pop oldest frame (N-30) and write with interpolated data

Benefits:
---------
- Single video pass (no second read)
- Real-time capable (constant memory usage)
- Automatic gap filling between valid detections
- Handles edge cases (video start/end, no detections)
"""

from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np
import supervision as sv


@dataclass
class BufferedFrame:
    """Stores a frame with its detection result for delayed processing."""
    frame: np.ndarray  # Original frame
    detection: Optional[np.ndarray]  # Ball center coordinates [x, y] or None
    confidence: float  # Detection confidence (0.0 if None)
    frame_index: int  # Frame number in video


class RealTimeBallInterpolator:
    """
    Real-time ball interpolation using a fixed-size buffer with single-pass processing.
    
    This class maintains a deque buffer of frames with detection results. When the buffer
    is full, it analyzes the historical detections, performs linear interpolation for gaps,
    and outputs the oldest frame with corrected annotations.
    
    Attributes:
        buffer_size (int): Size of the internal buffer (delay in frames)
        buffer (deque): Circular buffer storing BufferedFrame objects
        frame_counter (int): Total frames processed
    """
    
    def __init__(self, buffer_size: int = 30):
        """
        Initialize the interpolator with a fixed buffer size.
        
        Args:
            buffer_size (int): Number of frames to buffer (default: 30 frames = ~1.2s @ 25fps)
        """
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.frame_counter = 0
        
    def add_frame(self, frame: np.ndarray, detections: sv.Detections) -> Optional[BufferedFrame]:
        """
        Add a new frame with its detection result to the buffer.
        
        This method:
        1. Extracts ball position from detections (or None if no ball)
        2. Stores frame + detection in buffer
        3. If buffer is full: performs interpolation and returns oldest frame
        4. If buffer not full yet: returns None (warming up)
        
        Args:
            frame (np.ndarray): Current video frame
            detections (sv.Detections): YOLO detections (should contain 0 or 1 ball)
        
        Returns:
            Optional[BufferedFrame]: The oldest frame with interpolated detection, or None if warming up
        """
        # Extract ball position (center coordinates)
        ball_center = None
        confidence = 0.0
        
        if len(detections) > 0:
            # Get center coordinates of the best detection
            centers = detections.get_anchors_coordinates(sv.Position.CENTER)
            ball_center = centers[0]  # Shape: [x, y]
            confidence = float(detections.confidence[0])
        
        # Create buffered frame
        buffered = BufferedFrame(
            frame=frame.copy(),
            detection=ball_center,
            confidence=confidence,
            frame_index=self.frame_counter
        )
        
        self.buffer.append(buffered)
        self.frame_counter += 1
        
        # If buffer is full, perform interpolation and return oldest frame
        if len(self.buffer) == self.buffer_size:
            self._interpolate_buffer()
            return self.buffer[0]  # Return oldest (will be popped by caller)
        else:
            return None  # Still warming up
    
    def _interpolate_buffer(self):
        """
        Perform linear interpolation on the current buffer content.
        
        This method scans the buffer for sequences of missing detections between
        two valid detections and fills the gaps using linear interpolation.
        
        Example:
            Buffer: [Ball(x=100), None, None, Ball(x=200)]
            Result: [Ball(x=100), Ball(x=133), Ball(x=166), Ball(x=200)]
        """
        if len(self.buffer) < 3:
            return  # Need at least 3 frames to interpolate
        
        # Find all valid detection indices
        valid_indices = []
        for i, buffered in enumerate(self.buffer):
            if buffered.detection is not None:
                valid_indices.append(i)
        
        if len(valid_indices) < 2:
            return  # Need at least 2 valid detections to interpolate between
        
        # Interpolate gaps between consecutive valid detections
        for i in range(len(valid_indices) - 1):
            start_idx = valid_indices[i]
            end_idx = valid_indices[i + 1]
            gap_size = end_idx - start_idx - 1
            
            if gap_size > 0:
                # Linear interpolation between start and end
                start_pos = self.buffer[start_idx].detection
                end_pos = self.buffer[end_idx].detection
                
                for j in range(1, gap_size + 1):
                    # Calculate interpolated position
                    alpha = j / (gap_size + 1)
                    interpolated_pos = start_pos * (1 - alpha) + end_pos * alpha
                    
                    # Update the buffered frame with interpolated detection
                    current_idx = start_idx + j
                    self.buffer[current_idx].detection = interpolated_pos
                    self.buffer[current_idx].confidence = 0.5  # Mark as interpolated (medium confidence)
    
    def flush_buffer(self) -> List[BufferedFrame]:
        """
        Flush remaining frames from buffer at end of video.
        
        This method should be called after all frames have been processed to
        retrieve and interpolate the remaining buffered frames.
        
        Returns:
            List[BufferedFrame]: All remaining frames with interpolated detections
        """
        if len(self.buffer) == 0:
            return []
        
        # Perform final interpolation
        self._interpolate_buffer()
        
        # Return all remaining frames
        remaining = list(self.buffer)
        self.buffer.clear()
        return remaining
    
    def get_detection_as_sv_detections(self, buffered: BufferedFrame) -> sv.Detections:
        """
        Convert a BufferedFrame detection back to supervision Detections format.
        
        Args:
            buffered (BufferedFrame): Frame with detection data
        
        Returns:
            sv.Detections: Supervision detections (empty if no detection)
        """
        if buffered.detection is None:
            return sv.Detections.empty()
        
        # Create bounding box around center point (estimate box size)
        center = buffered.detection
        box_half_size = 15  # Approximate ball radius in pixels
        
        xyxy = np.array([[
            center[0] - box_half_size,
            center[1] - box_half_size,
            center[0] + box_half_size,
            center[1] + box_half_size
        ]])
        
        return sv.Detections(
            xyxy=xyxy,
            confidence=np.array([buffered.confidence]),
            class_id=np.array([0])  # Ball class
        )


class InterpolatedBallAnnotator:
    """
    Annotator for visualizing interpolated ball detections with visual distinction.
    
    This annotator draws:
    - Solid circles for real detections (high confidence)
    - Dashed circles for interpolated detections (medium confidence)
    - Trail of recent positions with fading effect
    """
    
    def __init__(self, radius: int = 8, trail_length: int = 15):
        """
        Initialize the annotator.
        
        Args:
            radius (int): Circle radius for ball visualization
            trail_length (int): Number of recent positions to show as trail
        """
        self.radius = radius
        self.trail_length = trail_length
        self.trail_buffer = deque(maxlen=trail_length)
        self.color_palette = sv.ColorPalette.from_matplotlib('Wistia', trail_length)
    
    def annotate(self, frame: np.ndarray, buffered: BufferedFrame) -> np.ndarray:
        """
        Annotate frame with ball detection and trail.
        
        Args:
            frame (np.ndarray): Frame to annotate
            buffered (BufferedFrame): Buffered frame with detection data
        
        Returns:
            np.ndarray: Annotated frame
        """
        import cv2
        
        annotated = frame.copy()
        
        # Add current detection to trail
        if buffered.detection is not None:
            self.trail_buffer.append({
                'position': buffered.detection,
                'confidence': buffered.confidence
            })
        
        # Draw trail (older positions with lower opacity)
        for i, trail_point in enumerate(self.trail_buffer):
            pos = trail_point['position']
            conf = trail_point['confidence']
            
            # Calculate visual properties
            color = self.color_palette.by_idx(i)
            interpolated_radius = int(1 + i * (self.radius - 1) / max(1, len(self.trail_buffer) - 1))
            
            # Different style for interpolated vs real detections
            thickness = 2 if conf > 0.6 else 1
            line_type = cv2.LINE_AA if conf > 0.6 else cv2.LINE_8
            
            center = (int(pos[0]), int(pos[1]))
            cv2.circle(
                img=annotated,
                center=center,
                radius=interpolated_radius,
                color=color.as_bgr(),
                thickness=thickness,
                lineType=line_type
            )
            
            # Add "I" marker for interpolated detections
            if conf <= 0.6 and i == len(self.trail_buffer) - 1:
                cv2.putText(
                    img=annotated,
                    text='I',
                    org=(center[0] + self.radius + 5, center[1] - self.radius),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.4,
                    color=(255, 255, 0),  # Yellow for interpolated
                    thickness=1
                )
        
        return annotated
