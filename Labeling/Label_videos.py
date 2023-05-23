import cv2
import keyboard
import csv
from moviepy.editor import VideoFileClip

def process_frame(frame):
    # Here you can implement your logic to process each frame
    # For this example, let's assume you want to label frames as 'event' or 'no event'
    label = None
    # Display the frame
    cv2.imshow("Frame", frame)
    # Wait for a keystroke
    key = cv2.waitKey(0)
    if key == ord('e'):
        label = 'event'
    elif key == ord('n'):
        label = 'no event'
    return label

def main(video_path, output_csv):
    # Open the video file
    video = VideoFileClip(video_path)
    
    frame_labels = []
    
    for i, frame in enumerate(video.iter_frames()):
        # Process the current frame
        label = process_frame(frame)
        frame_labels.append(label)
        
        # Break the loop if the user presses 'q'
        if keyboard.is_pressed('q'):
            break
    
    # Write the frame labels to a CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Frame', 'Label'])
        for i, label in enumerate(frame_labels):
            writer.writerow([i+1, label])

    # Close the OpenCV windows
    cv2.destroyAllWindows()


# Provide the video file path and the output CSV file path
video_path = 'TORM_2m_24h/TR1 - Caja 5 - A_L.mp4'
output_csv = 'TORM_2m_24h/TR1 - Caja 5 - A_L.csv'

# Call the main function
main(video_path, output_csv)
