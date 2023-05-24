import cv2
import keyboard
import csv
from moviepy.editor import VideoFileClip
from tkinter import Tk, filedialog

def process_frame(frame):
    # Implement your logic to process each frame
    label = None
    # Display the frame
    cv2.imshow("Frame", frame)
    # Wait for a keystroke
    key = cv2.waitKey(0)
    if key == ord('a'):
        label = 'Exploring_left'
    elif key == ord('d'):
        label = 'Exploring_right'
    elif key == ord('s'):
        label = 'No exploration'
    elif key == ord('f'):
        label = 'Freezing'
    elif key == ord('g'):
        label = 'Grooming'
    elif key == ord('w'):
        label = 'back'
    return label

def main():
    # Create a Tkinter window
    root = Tk()
    root.withdraw()

    # Open a file dialog to select the video file
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4")])

    if not video_path:
        print("No video file selected.")
        return

    # Open the video file
    video = VideoFileClip(video_path)
    
    frame_labels = []
    frame_generator = video.iter_frames()
    frame_list = list(frame_generator) # This takes a while
    current_frame = 0
    
    while current_frame < len(frame_list):
        frame = frame_list[current_frame]
        
        # Process the current frame
        label = process_frame(frame)
        
        if label == 'back':
            # Go back one frame
            current_frame = max(0, current_frame - 1)
            continue
        
        frame_labels.append(label)
        
        # Break the loop if the user presses 'q'
        if keyboard.is_pressed('q'):
            break
        
        current_frame += 1

    # Write the frame labels to a CSV file
    output_csv = video_path.rsplit('.', 1)[0] + '.csv'
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Frame', 'Label'])
        for i, label in enumerate(frame_labels):
            writer.writerow([i+1, label])

    # Close the OpenCV windows
    cv2.destroyAllWindows()


# Call the main function
main()
