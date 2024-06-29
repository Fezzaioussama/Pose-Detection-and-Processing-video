# Pose Detection and Processing Script

## Overview
This script processes a video clip to detect human poses using keypoints provided in a JSON file. It calculates various bounding boxes and vectors for body parts and saves the results in a CSV file. Additionally, it draws the detected poses on the video frames and saves the annotated video.

## Requirements
- Python 3.x
- OpenCV
- NumPy
- pandas
- argparse

## Installation
Before running the script, ensure you have the required libraries installed. You can install them using pip:

```sh
pip install opencv-python numpy pandas
```

## Usage
To run the script, use the following command:

python script.py --path_to_clip_input <input_video_path> --path_to_clip_output <output_video_path> --path_to_json <pose_json_path> --path_to_csv <output_csv_path>

## Arguments

--path_to_clip_input: The path to the input video file. Default is D:/to_send/clip_05.mp4.

--path_to_clip_output: The path to the output video file with annotated poses. Default is D:/to_send/output_clip_05.mp4.

--path_to_json: The path to the JSON file containing pose detection data. Default is D:/to_send/clip_05_pose.json.

--path_to_csv: The path to the CSV file where the processed data will be saved. Default is D:/to_send/clip_05_output.csv.


## Script Description
### Functions
####  unit_vector_and_length(p1, p2): 
Calculates the unit vector and length between two points.
####    bounding_box(points): 
Calculates the bounding box center coordinates, width, and height for a given set of points.


## Main Processing
Load Pose Data: Loads pose keypoints from the specified JSON file.Video Processing: Opens the input video and reads frame by frame.
Pose Detection and Annotation: For each frame, it:
        Extracts keypoints for the detected persons.
        Calculates bounding boxes for the head and upper body.
        Computes vectors and lengths for various limbs.
        Draws bounding boxes and limb lines on the frame.
Save Annotated Video: Writes the annotated frames to the output video file.Save Data to CSV: Stores the calculated data (bounding boxes, vectors, lengths) in a CSV file.

## Output Data
![Output Data Example](https://github.com/Fezzaioussama/Pose-estimation-from-video/blob/main/Capture.PNG)
The output CSV file contains the following columns:

frame_ID, person_ID
head_center_x, head_center_y, head_width, head_height
body_center_x, body_center_y, body_width, body_height
left_shoulder_x, left_shoulder_y, left_shoulder_vec_x, left_shoulder_vec_y, left_upper_arm_length
left_elbow_vec_x, left_elbow_vec_y, left_lower_arm_length
right_shoulder_x, right_shoulder_y, right_shoulder_vec_x, right_shoulder_vec_y, right_upper_arm_length
right_elbow_vec_x, right_elbow_vec_y, right_lower_arm_length
left_hip_x, left_hip_y, left_hip_vec_x, left_hip_vec_y, left_upper_leg_length
left_knee_vec_x, left_knee_vec_y, left_lower_leg_length
right_hip_x, right_hip_y, right_hip_vec_x, right_hip_vec_y, right_upper_leg_length
right_knee_vec_x, right_knee_vec_y, right_lower_leg_length

## Example Command
```sh
python script.py --path_to_clip_input "D:/to_send/clip_05.mp4" --path_to_clip_output "D:/to_send/output_clip_05.mp4" --path_to_json "D:/to_send/clip_05_pose.json" --path_to_csv "D:/to_send/clip_05_output.csv"
```
