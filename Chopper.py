import os
import cv2
from PyQt5 import QtWidgets, QtCore

class Chopper(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        # Set up the GUI
        self.setWindowTitle('Video Chopper')
        self.resize(400, 100)

        self.folder_label = QtWidgets.QLabel('Select a folder to chop:')
        self.folder_button = QtWidgets.QPushButton('Browse')
        self.folder_button.clicked.connect(self.browse_folder)

        self.run_button = QtWidgets.QPushButton('Run')
        self.run_button.clicked.connect(self.run)
        
        self.time_label = QtWidgets.QLabel('Enter length of chopped videos:')
        self.time_edit = QtWidgets.QTimeEdit()
        self.time_edit.setDisplayFormat("mm:ss")
        self.time_edit.timeChanged.connect(lambda time: setattr(self, "total_minutes", time))
        
        
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.folder_label)
        layout.addWidget(self.folder_button)
        layout.addWidget(self.run_button)
        layout.addWidget(self.time_label)
        layout.addWidget(self.time_edit)

        self.setLayout(layout)
        self.show()

        # Set up the video processing parameters
        self.fps = 30
        self.frame_width = 0
        self.frame_height = 0

    def browse_folder(self):
        """Open a file dialog to select a folder."""
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select a folder')
        self.folder_button.setText(folder_path)
        self.folder_path = folder_path

    def run(self):
        """Process all videos in the selected folder."""
        if not hasattr(self, 'folder_path'):
            return

        # Create the "videos cortados" folder if it does not exist
        cortados_path = os.path.join(self.folder_path, 'videos cortados')
        if not os.path.exists(cortados_path):
            os.makedirs(cortados_path)

        # Get the list of video files in the folder
        filenames = os.listdir(self.folder_path)
        video_filenames = [filename for filename in filenames if os.path.splitext(filename)[1].lower() in ('.mp4', '.avi')]

        total_seconds = - (self.total_minutes.secsTo(QtCore.QTime(0, 0)))
        
        # Process each video file
        for video_filename in video_filenames:
            video_path = os.path.join(self.folder_path, video_filename)
            video = cv2.VideoCapture(video_path)

            # Get the video information
            self.fps = int(video.get(cv2.CAP_PROP_FPS))
            self.frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calculate the frames to process (X minutos)
            frames_to_process = min(self.fps * total_seconds, self.frame_count)

            # Create the output video writers
            video_name = os.path.splitext(video_filename)[0]
            video_izq_path = os.path.join(cortados_path, video_name + '_L.mp4')
            video_der_path = os.path.join(cortados_path, video_name + '_R.mp4')
            video_izq = cv2.VideoWriter(video_izq_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.frame_height, self.frame_width//2))
            video_der = cv2.VideoWriter(video_der_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.frame_height, self.frame_width//2))

            # Process each frame of the video
            frame_index = 0
            while True:
                ret, frame = video.read()
                if not ret or frame_index >= frames_to_process:
                    break

                # Split the frame in half and rotate each half
                frame_izq = frame[:, :self.frame_width//2, :]
                frame_izq = cv2.rotate(frame_izq, cv2.ROTATE_90_COUNTERCLOCKWISE)

                frame_der = frame[:, self.frame_width//2:, :]
                frame_der = cv2.rotate(frame_der, cv2.ROTATE_90_CLOCKWISE)
                
                # Write the halves to the output videos
                video_izq.write(frame_izq)
                video_der.write(frame_der)
                
                frame_index += 1
                
                # Release the input and output video resources
            video.release()
            video_izq.release()
            video_der.release()

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    splitter = Chopper()
    app.exec_()
