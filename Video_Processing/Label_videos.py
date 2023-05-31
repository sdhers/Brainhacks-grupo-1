import cv2
import keyboard
import csv
from moviepy.editor import VideoFileClip
from tkinter import Tk, filedialog

def process_frame(frame, frame_number):
    # Implement your logic to process each frame
    left = 0
    right = 0
    back = False
    
    # Display the frame
    frame_number_text = "Frame: {}".format(frame_number)
    
    # Define the position and font properties for the text
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (10, 30)  # Adjust the position as needed
    font_scale = 1
    font_color = (0, 255, 0)  # BGR color format
    thickness = 2
    
    # Add the text to the frame
    cv2.putText(frame, frame_number_text, position, font, font_scale, font_color, thickness)
    
    # Display the frame with the frame number
    cv2.imshow("Frame", frame)
    # Wait for a keystroke
    key = cv2.waitKey(0)
    if key == ord('1'):
        left = 1
    elif key == ord('3'):
        right = 1
    elif key == ord('2'):
        pass
        # label = 'No exploration'
    # elif key == ord('f'):
     #    label = 'Freezing'
    # elif key == ord('g'):
     #    label = 'Grooming'
    elif key == ord('5'):
        back = True
    return left, right, back

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
    
    frame_generator = video.iter_frames()
    frame_list = list(frame_generator) # This takes a while
    frame_labels_left = ["-"]*len(frame_list)
    frame_labels_right = ["-"]*len(frame_list)
    current_frame = 0
    print(len(frame_list))
    
    while current_frame < len(frame_list):
        frame = frame_list[current_frame]
        
        # Process the current frames
        left, right, back = process_frame(frame, current_frame)
        
        if back:
            # Go back one frame
            current_frame = max(0, current_frame - 1)
            continue
        
        # Break the loop if the user presses 'q'
        if keyboard.is_pressed('q'):
            break
        
        frame_labels_left[current_frame] = left
        frame_labels_right[current_frame] = right
        
        current_frame += 1

    # Write the frame labels to a CSV file
    output_csv = video_path.rsplit('.', 1)[0] + '.csv'
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Frame', 'Left', 'Right'])
        for i, (left, right) in enumerate(zip(frame_labels_left, frame_labels_right)):
            writer.writerow([i+1, left, right])

    # Close the OpenCV windows
    cv2.destroyAllWindows()


# Call the main function
main()
