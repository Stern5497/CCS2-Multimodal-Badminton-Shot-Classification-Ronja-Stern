import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import pandas as pd
from moviepy.video.io.VideoFileClip import VideoFileClip
import librosa


# define filenames
file = open("filenames.txt", "r")
lines = file.readlines()
file.close()

files_test = [line for line in lines if line.split()[3] == "test"]
files_train = [line for line in lines if line.split()[3] == "train"]

files_test = []
files_train = files_train[496:]
data = []
index = 31  # choose one index (0-61) where we draw pose detection on image

"""# for debugging only:
files_test = ["smash_ronja_9_f.mp4 42.40210884353741 44.40210884353741 test",
              "slice_ronja_18_f.mp4 72.95537414965986 74.95537414965986 test"]
files_train = ["smash_sarah_13.mp4 46.47179138321995 48.47179138321995 train",
               "smash_ronja_21_e.mp4 87.10580498866213 89.10580498866213 train",
               "slice_ronja_20_f.mp4 80.96845804988662 82.96845804988662 train"]"""
progress = 0
for files, split in zip([files_test, files_train], ["test", "train"]):
    junk_len = len(files)
    junk = 0
    for file in files:
        filename = file.split()[0]
        # get shot label
        shot = filename.split("_")[0]
        filename = filename.strip()
        # detect pose as numpy array
        clip = VideoFileClip(f'segmented_videos/{split}/{filename}')
        frames = clip.iter_frames()

        # Below VideoWriter object will create
        # a frame of above defined
        mp_holistic = mp.solutions.holistic
        mp_drawing = mp.solutions.drawing_utils

        # Initialize holistic pose estimation object
        holistic = mp_holistic.Holistic()
        detected_poses = []

        for frame, i in zip(frames, range(62)):
            # Make a detection
            results = holistic.process(frame)
            # Example how to get the coordinates of the left shoulder
            # left_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER]
            # this includes a list of landmarks, each having a x, y, z and visibility property
            pose = []
            try:
                if results:
                    if results.pose_landmarks:
                        if results.pose_landmarks.landmark:
                            for data_point in results.pose_landmarks.landmark:
                                pose.append([data_point.x, data_point.y, data_point.z, data_point.visibility])
            except ValueError as e:
                # Code to handle value error (invalid conversion)
                print(f"Error: could not process video: {file}", e)
            pose_array = np.array(pose)

            # save detected pose
            detected_poses.append(pose_array)

            if i == index:
                # Draw the detection points on the image
                annotated_image = frame.copy()
                mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

                # Display the resulting image
                cv.imshow('Annotated Frame', annotated_image)
                plt.imshow(annotated_image)  # Show with plt.imshow for comparison
                plt.savefig(f'plots/{split}/{filename.split(".")[0]}')

        # extract audio from video

        video_clip = VideoFileClip(f'segmented_videos/{split}/{filename}')
        audio_clip = video_clip.audio
        audio_array = np.stack(list(audio_clip.iter_frames()))[:, 0]  # only use one channel -> this proofed to be enough

        """
        # Compute the mel spectrogram
        sr = audio_clip.fps  # Sample rate
        mel_spectrogram = librosa.feature.melspectrogram(y=audio_array, sr=sr)
        """
        # add start and end points to be able to find shot in the original videos
        start = file.split()[1]
        end = file.split()[2]

        #data.append([filename, shot, detected_poses, audio_array, mel_spectrogram, start, end])
        data.append([filename, shot, detected_poses, audio_array, [], start, end])
        progress += 1
        print(progress)


        if progress >= (junk+1)*junk_len:
            # save df as file
            df = pd.DataFrame(data, columns=["filename", "shot", "pose", "audio", "mel_spectrogram", "start_point", "end_point"])
            # df.to_csv(f'data/{split}.csv', index=False)
            df.to_json(f'data/{split}_{junk+9}.json')
            junk += 1
            data = []
