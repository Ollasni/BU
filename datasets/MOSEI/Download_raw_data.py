import os
from mmsdk import mmdatasdk

# Set your data path and API key
data_path = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/CMU_MOSEI/Raw_Data"
api_key = "your_api_key"

# Load the dataset
dataset = mmdatasdk.mmdataset(mmdatasdk.cmu_mosei.raw, data_path)

# # Define the dataset you want to download
# cmu_mosei_highlevel = mmdatasdk.mmdataset(mmdatasdk.cmu_mosei.highlevel, data_path)
# cmu_mosei_labels = mmdatasdk.mmdataset(mmdatasdk.cmu_mosei.labels, data_path)
# cmu_mosei_raw = mmdatasdk.mmdataset(mmdatasdk.cmu_mosei.raw, data_path)
#
# # Download the data
# cmu_mosei_highlevel.download(api_key)
# cmu_mosei_labels.download(api_key)
# cmu_mosei_raw.download(api_key)

# Load the downloaded data
# cmu_mosei_raw = mmdatasdk.mmdataset(mmdatasdk.cmu_mosei.raw, data_path)

# from mmsdk import mmdatasdk
#
# # Load the dataset assuming it has already been downloaded
# data_path = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/CMU_MOSEI/Raw_Data/"
# cmu_mosei = mmdatasdk.mmdataset(data_path)
#
# # Accessing the computational sequences
# for key in cmu_mosei.computational_sequences.keys():
#     print(f"Modality: {key}")
#     print(cmu_mosei.computational_sequences[key].keys())
#     print(f"Number of Entries: {len(cmu_mosei.computational_sequences[key].keys())}")
#     print(f"Feature Dimensions: {cmu_mosei.computational_sequences[key]['metadata']['dimensions']}")
#
#
# # Access data normally
# print(cmu_mosei.summary())
#
#
# # Example: Print out a summary of the dataset
# print(cmu_mosei_raw.summary())
#
# # Load the aligned features
# aligned_data = mmdatasdk.mmdataset(mmdatasdk.cmu_mosei.aligned, "path_to_data_directory")
#
# # Accessing specific data
# text_features = aligned_data.computational_sequences["text"]
# audio_features = aligned_data.computational_sequences["audio"]
# video_features = aligned_data.computational_sequences["video"]
#
# print(text_features.shape)
# print(audio_features.shape)
# print(video_features.shape)