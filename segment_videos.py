import moviepy.editor as mp
import librosa
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import sys

random.seed(42)  # Set the random seed

# define name of videos and constants
path = f"{os.path.dirname(os.path.abspath(sys.argv[0]))}/original_videos"
filenames = os.listdir(path)
filenames.remove('special')
filenames.remove('original long')
# filenames = ["ronja_smash_a"] for debugging
sample_rate = 44100
files = []

# slice_nora_7.mp4 34.04587301587301 36.04587301587301 train had to be removed
# smash_nora_12.mp4 45.54070294784581 47.54070294784581 train had to be removed

for filename in filenames:
    filename_short = filename[:-4]
    splits = filename_short.split("_")
    if len(splits) > 1:
        shot = splits[1]
        name = splits[0]
    else:
        print(f"Error: there was a file name with the wrong naming: {filename}")
        break
    # get starting and end times of cut videos
    video_clip = mp.VideoFileClip(f"original_videos/{filename}")
    audio_clip = video_clip.audio

    audio_array = np.stack(list(audio_clip.iter_frames()))[:, 0]  # only use one channel -> this proofed to be enough
    # Process audio to detect peaks
    peaks = librosa.util.peak_pick(audio_array, pre_max=50, post_max=50, pre_avg=50, post_avg=50, delta=0.3, wait=1000)
    start_times = [max(0, (peak_index - int(1 * sample_rate)) / sample_rate) for peak_index in peaks]
    end_times = [min(video_clip.duration, (peak_index + int(1 * sample_rate)) / sample_rate) for peak_index in peaks]
    # visualise cuts depending on sound -> one channel
    time = np.arange(len(audio_array)) / sample_rate
    start_points = [max(0, peak_index - int(1 * sample_rate)) for peak_index in peaks]
    end_points = [min(len(audio_array), peak_index + int(1 * sample_rate)) for peak_index in peaks]

    # Plot the audio signal with peaks and cut points
    plt.figure(figsize=(12, 6))
    plt.plot(time, audio_array, label='Normalized Audio')
    plt.scatter(np.array(peaks) / sample_rate, audio_array[peaks], color='red', label='Peaks')
    plt.vlines(np.array(start_points) / sample_rate, 0, 1, color='green', linestyle='--', label='Start Points')
    plt.vlines(np.array(end_points) / sample_rate, 0, 1, color='blue', linestyle='--', label='End Points')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title(f'Audio with Peaks and Cut Points for {filename}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'segmentation_plots/{filename_short}.png')
    plt.show()

    # decide randomly what videos are in the test or in the train set
    test_indices = random.sample(range(len(peaks)), int(len(peaks) * 0.2))

    # cut video and save them
    for i in range(len(peaks)):
        # Calculate starting and ending points around peaks
        sub_clip = video_clip.subclip(start_times[i], end_times[i])
        sub_clip = sub_clip.resize(newsize=(1080, 1920))
        name_tmp = '_'.join([shot, name, str(i+1)])
        if len(splits)==3:
            identifier = splits[2]
            name_tmp += f"_{identifier}"
        if i in test_indices:
            split = "test"
        else:
            split = "train"
        sub_clip.write_videofile(f"segmented_videos/{split}/{name_tmp}.mp4")
        files.append(f"{name_tmp}.mp4 {start_times[i]} {end_times[i]} {split}")

    # Close the video clip
    video_clip.close()

with open('filenames.txt', mode='a+', encoding='utf-8') as myfile:
    myfile.write('\n'.join(files))
    myfile.write('\n')
