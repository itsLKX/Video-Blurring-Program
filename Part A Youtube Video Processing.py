import cv2
import numpy as np
import os

def overlay_watermark(background, watermark, x_offset, y_offset):
    """
    Overlays a watermark onto a background image. Handles both 3-channel (BGR) and
    4-channel (BGRA) watermarks.
    """
    bg_h, bg_w, _ = background.shape
    
    # Ensure the watermark is resized before this function is called.
    wm_h, wm_w, _ = watermark.shape

    # Boundary checks to prevent errors if the watermark is larger than the frame
    if y_offset < 0: y_offset = 0
    if x_offset < 0: x_offset = 0
    if y_offset + wm_h > bg_h:
        wm_h = bg_h - y_offset
        watermark = watermark[0:wm_h, :]
    if x_offset + wm_w > bg_w:
        wm_w = bg_w - x_offset
        watermark = watermark[:, 0:wm_w]


    roi = background[y_offset:y_offset + wm_h, x_offset:x_offset + wm_w]

    # --- FIX: Check for alpha channel and apply watermark accordingly ---
    if watermark.shape[2] == 4:  # If watermark has an alpha channel (BGRA)
        # Separate alpha channel and BGR channels
        alpha = watermark[:, :, 3] / 255.0
        bgr_watermark = watermark[:, :, :3]

        # Blend using the alpha channel
        for c in range(0, 3):
            roi[:, :, c] = (alpha * bgr_watermark[:, :, c] +
                          (1.0 - alpha) * roi[:, :, c])
    else:  # If watermark is standard BGR (no alpha)
        # Fallback to creating a mask
        gray_wm = cv2.cvtColor(watermark, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(gray_wm, 1, 255, cv2.THRESH_BINARY)[1]
        mask_inv = cv2.bitwise_not(mask)

        frame_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        wm_fg = cv2.bitwise_and(watermark, watermark, mask=mask)
        
        # Combine the foreground and background
        roi[:] = cv2.add(frame_bg, wm_fg)

    return background


def process_videos():
    """
    Main function to process a list of videos according to the assignment's requirements.
    It iterates through each video and applies a series of effects:
    1. Detects if the video is nighttime and adjusts brightness.
    2. Blurs all detected faces.
    3. Overlays a "talking head" video.
    4. Adds two separate watermarks to every video, alternating every 5 seconds.
    5. Appends an end screen video.
    """
    # --- Configuration ---
    # NOTE: The 'r' before the string is important for Windows paths.
    
    VIDEO_FILES_TO_PROCESS = [
        r"C:\Users\User\Desktop\Digital Image\Group Assignment\CSC2014- Group Assignment_Aug-2025\Recorded Videos (4)\alley.mp4",
        r"C:\Users\User\Desktop\Digital Image\Group Assignment\CSC2014- Group Assignment_Aug-2025\Recorded Videos (4)\office.mp4",
        r"C:\Users\User\Desktop\Digital Image\Group Assignment\CSC2014- Group Assignment_Aug-2025\Recorded Videos (4)\singapore.mp4",
        r"C:\Users\User\Desktop\Digital Image\Group Assignment\CSC2014- Group Assignment_Aug-2025\Recorded Videos (4)\traffic.mp4"
    ]

    # Full paths to all asset files
    TALKING_VIDEO_PATH = r'C:\Users\User\Desktop\Digital Image\Group Assignment\CSC2014- Group Assignment_Aug-2025\talking.mp4'
    ENDSCREEN_VIDEO_PATH = r'C:\Users\User\Desktop\Digital Image\Group Assignment\CSC2014- Group Assignment_Aug-2025\endscreen.mp4'
    WATERMARK1_PATH = r'C:\Users\User\Desktop\Digital Image\Group Assignment\CSC2014- Group Assignment_Aug-2025\watermark1.png'
    WATERMARK2_PATH = r'C:\Users\User\Desktop\Digital Image\Group Assignment\CSC2014- Group Assignment_Aug-2025\watermark2.png'
    FACE_CASCADE_PATH = r'C:\Users\User\Desktop\Digital Image\Group Assignment\CSC2014- Group Assignment_Aug-2025\face_detector.xml'
    
    # Create an output directory on your Desktop to make it easy to find.
    output_dir = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop', 'processed_videos')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory at: {output_dir}")

    # --- Load Resources ---
    try:
        face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        watermark1 = cv2.imread(WATERMARK1_PATH, cv2.IMREAD_UNCHANGED)
        watermark2 = cv2.imread(WATERMARK2_PATH, cv2.IMREAD_UNCHANGED)
        if face_cascade.empty() or watermark1 is None or watermark2 is None:
            raise IOError("A required file could not be loaded. Please check all file paths.")
    except Exception as e:
        print(f"Error during resource loading: {e}")
        return

    # --- Main Video Processing Loop ---
    for i, video_path in enumerate(VIDEO_FILES_TO_PROCESS):
        print(f"Processing video: {video_path}...")

        cap = cv2.VideoCapture(video_path)
        talking_cap = cv2.VideoCapture(TALKING_VIDEO_PATH)
        endscreen_cap = cv2.VideoCapture(ENDSCREEN_VIDEO_PATH)

        if not all([cap.isOpened(), talking_cap.isOpened(), endscreen_cap.isOpened()]):
            print(f"Error opening one or more video files for {video_path}.")
            continue

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if fps == 0:
            print(f"Warning: FPS for {video_path} is 0. Defaulting to 30.")
            fps = 30

        output_filename = os.path.join(output_dir, f"processed_{os.path.basename(video_path).split('.')[0]}.avi")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

        is_nighttime = False
        ret, first_frame = cap.read()
        if ret:
            gray_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray_frame)
            if brightness < 75:  
                is_nighttime = True
                print(f"  - Nighttime detected for {video_path}. Applying brightness correction.")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        frame_count = 0
        frames_in_5_seconds = int(fps * 5)
        
        while True:
            ret, frame = cap.read()
            if not ret: break

            if is_nighttime:
                frame = cv2.add(frame, np.ones(frame.shape, dtype="uint8") * 50)
            
            gray_frame_for_faces = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame_for_faces, scaleFactor=1.1, minNeighbors=10, minSize=(40, 40))
            for (x, y, w, h) in faces:
                frame[y:y+h, x:x+w] = cv2.GaussianBlur(frame[y:y+h, x:x+w], (99, 99), 30)

            ret_talk, talking_frame = talking_cap.read()
            if not ret_talk:
                talking_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret_talk, talking_frame = talking_cap.read()

            if ret_talk:
                target_w = frame_width // 4
                target_h = int(talking_frame.shape[0] * (target_w / talking_frame.shape[1]))
                talking_frame_resized = cv2.resize(talking_frame, (target_w, target_h))
                frame[10:10+target_h, 10:10+target_w] = talking_frame_resized

            current_interval = frame_count // frames_in_5_seconds

            if current_interval % 2 == 0:
                # --- Watermark 1 (Full width, center of screen) ---
                wm1_target_w = int(frame_width * 1.0)
                wm1_target_h = int(watermark1.shape[0] * (wm1_target_w / watermark1.shape[1]))
                resized_wm1 = cv2.resize(watermark1, (wm1_target_w, wm1_target_h), interpolation=cv2.INTER_LANCZOS4)
                
                h1, w1, _ = resized_wm1.shape
                x1_offset = (frame_width - w1) // 2
                y1_offset = (frame_height - h1) // 2
                
                frame = overlay_watermark(frame, resized_wm1, x1_offset, y1_offset)
            else:
                # --- CORRECTED: Watermark 2 (Full width, center of screen) ---
                wm2_target_w = int(frame_width * 1.0)
                wm2_target_h = int(watermark2.shape[0] * (wm2_target_w / watermark2.shape[1]))
                resized_wm2 = cv2.resize(watermark2, (wm2_target_w, wm2_target_h), interpolation=cv2.INTER_LANCZOS4)
                
                h2, w2, _ = resized_wm2.shape
                x2_offset = (frame_width - w2) // 2
                y2_offset = (frame_height - h2) // 2
                
                frame = overlay_watermark(frame, resized_wm2, x2_offset, y2_offset)
            
            out.write(frame)
            frame_count += 1

        print("  - Appending end screen...")
        while True:
            ret_end, end_frame = endscreen_cap.read()
            if not ret_end: break
            end_frame_resized = cv2.resize(end_frame, (frame_width, frame_height))
            out.write(end_frame_resized)

        print(f"Finished processing. Output saved to {output_filename}")
        cap.release()
        talking_cap.release()
        endscreen_cap.release()
        out.release()

    cv2.destroyAllWindows()
    print(f"All videos processed successfully. Check the '{output_dir}' folder on your Desktop.")


if __name__ == '__main__':
    process_videos()
