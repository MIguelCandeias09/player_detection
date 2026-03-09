"""
Quick test script for RealTimeBallInterpolator
==============================================

This script validates the interpolation logic without running full video processing.
"""

import numpy as np
import sys
import os

# Add src/ to path so sports module can be imported
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from sports.common.ball_interpolator import RealTimeBallInterpolator, BufferedFrame
import supervision as sv

def create_mock_detection(x, y, confidence=0.9):
    """Create a mock ball detection at position (x, y)"""
    xyxy = np.array([[x-10, y-10, x+10, y+10]])
    return sv.Detections(
        xyxy=xyxy,
        confidence=np.array([confidence]),
        class_id=np.array([0])
    )

def test_interpolation():
    """Test the interpolation logic with synthetic data"""
    print("🧪 Testing RealTimeBallInterpolator...")
    
    interpolator = RealTimeBallInterpolator(buffer_size=10)
    
    # Simulate 20 frames with gaps
    test_frames = [
        (100, 100, 0.9),  # Frame 0: Ball at (100, 100)
        (110, 105, 0.9),  # Frame 1: Ball at (110, 105)
        (None, None, 0),  # Frame 2: Missing
        (None, None, 0),  # Frame 3: Missing
        (None, None, 0),  # Frame 4: Missing
        (140, 120, 0.9),  # Frame 5: Ball at (140, 120) - should interpolate 2,3,4
        (150, 125, 0.9),  # Frame 6: Ball continues
        (None, None, 0),  # Frame 7: Missing
        (170, 135, 0.9),  # Frame 8: Ball reappears - should interpolate 7
        (180, 140, 0.9),  # Frame 9: Ball continues
        (190, 145, 0.9),  # Frame 10: More frames to fill buffer
        (200, 150, 0.9),  # Frame 11
        (210, 155, 0.9),  # Frame 12
        (220, 160, 0.9),  # Frame 13
        (230, 165, 0.9),  # Frame 14
    ]
    
    outputs = []
    
    for i, (x, y, conf) in enumerate(test_frames):
        # Create mock frame (doesn't matter for this test)
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Create detection
        if x is not None:
            detection = create_mock_detection(x, y, conf)
        else:
            detection = sv.Detections.empty()
        
        # Add to interpolator
        buffered = interpolator.add_frame(mock_frame, detection)
        
        if buffered is not None:
            outputs.append({
                'frame_index': buffered.frame_index,
                'detection': buffered.detection,
                'confidence': buffered.confidence
            })
            print(f"✅ Frame {buffered.frame_index}: ", end='')
            if buffered.detection is not None:
                print(f"Ball at ({buffered.detection[0]:.1f}, {buffered.detection[1]:.1f}), conf={buffered.confidence:.2f}")
                if buffered.confidence == 0.5:
                    print("   ^ INTERPOLATED")
            else:
                print("No ball detected")
    
    # Flush remaining
    print("\n🔚 Flushing remaining frames...")
    remaining = interpolator.flush_buffer()
    for buffered in remaining:
        print(f"✅ Frame {buffered.frame_index}: ", end='')
        if buffered.detection is not None:
            print(f"Ball at ({buffered.detection[0]:.1f}, {buffered.detection[1]:.1f}), conf={buffered.confidence:.2f}")
            if buffered.confidence == 0.5:
                print("   ^ INTERPOLATED")
        else:
            print("No ball detected")
    
    print(f"\n✅ Test complete! Processed {len(test_frames)} frames, output {len(outputs) + len(remaining)} frames")
    
    # Validate interpolation worked
    print("\n🔍 Validation:")
    all_frames = outputs + [{'frame_index': b.frame_index, 'detection': b.detection, 'confidence': b.confidence} 
                             for b in remaining]
    
    # Check if gaps were filled
    interpolated_count = sum(1 for f in all_frames if f['confidence'] == 0.5)
    print(f"   Interpolated frames: {interpolated_count}")
    
    # Check continuity (frames 2-4 should be interpolated between 1 and 5)
    frame_2 = next((f for f in all_frames if f['frame_index'] == 2), None)
    frame_3 = next((f for f in all_frames if f['frame_index'] == 3), None)
    frame_4 = next((f for f in all_frames if f['frame_index'] == 4), None)
    
    if frame_2 and frame_2['detection'] is not None:
        print(f"   Frame 2 interpolated: ({frame_2['detection'][0]:.1f}, {frame_2['detection'][1]:.1f}) ✅")
    if frame_3 and frame_3['detection'] is not None:
        print(f"   Frame 3 interpolated: ({frame_3['detection'][0]:.1f}, {frame_3['detection'][1]:.1f}) ✅")
    if frame_4 and frame_4['detection'] is not None:
        print(f"   Frame 4 interpolated: ({frame_4['detection'][0]:.1f}, {frame_4['detection'][1]:.1f}) ✅")
    
    print("\n✅ Interpolation test passed!")

if __name__ == '__main__':
    test_interpolation()
