CCS2 2024 - Multimodal Badminton Shot Classification - Ronja

GitHub: https://github.com/Stern5497/CCS2-Multimodal-Badminton-Shot-Classification-Ronja-Stern

- data: Contains test and train splits, both in multiple files, without the original video material (but pose)
- original videos: Contains the original not cropped videos
plots: containing for both train and test split for each video an example how the poses were extracted
- segmentation_plots: for each orgiginal video a plot of how it was cropped
- segmented_videos: the cropped original videos split into train and test splits
- writing: contains images, presentation, project proposal, plots and the final version of the project report

- audio-pose-combi.ipynb: The notebook where we train and evaluate our audio, pose and combi model. Google Colab was used to run experiments.
- dataset-overview.ipynb: The notebook to illustrate dataset properties as the class balance or the different splits.
- ensemble.ipynb: the notebook used to train and evaluate the ensemble model. Google Colab was used to run experiments
- prepare_dataset: A file containing python code to extract poses from the videos and save all needed properties as a pandas Dataframe in a .csv file. The results of this code were saved in folder data and an example of the extracted poses in plots folder
- segment_videos: A file containing python code to crop the original videos into single shot videos (can be found in segmented videos). The plots of how a video was segmented can be found in segmentation_plots.