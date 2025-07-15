# Video-Blurring-Program
This Python program uses the OpenCV library to perform various video processing tasks. Its main goal is to take a set of input videos and apply several visual enhancements and overlays, then save the modified videos to an output directory.

Key Features
Brightness Adjustment: It intelligently detects if a video is shot at nighttime (based on overall brightness) and automatically increases the brightness of such videos for better visibility.

Face Blurring: The program identifies faces within each video frame using a pre-trained Haar Cascade classifier and applies a blur effect to them, which is useful for privacy or anonymity.

"Talking Head" Overlay: It overlays a smaller "talking head" video onto the main video, typically positioned in the top-left corner. If the talking head video ends, it loops from the beginning.

Alternating Watermarks: The program adds two different watermarks to the video, switching between them every 5 seconds. The watermarks are resized to the full width of the video and centered on the screen. It correctly handles watermarks with and without an alpha (transparency) channel.

End Screen Appending: After processing the main video, it appends a separate "endscreen" video, resizing it to fit the main video's dimensions.

How It Works
The process_videos function orchestrates the entire workflow. It configures paths to input videos and various assets (watermarks, face detector, talking head video, end screen video). It then iterates through each specified video, applying the effects frame by frame.

The overlay_watermark function is a utility that handles placing watermarks onto a background image, properly managing transparent watermarks to ensure they blend seamlessly.

In essence, this program is a comprehensive video editor for automating specific visual effects and branding elements across multiple video files.
