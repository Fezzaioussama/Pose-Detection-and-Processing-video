import json
import cv2
import numpy as np
import pandas as pd
import os
import argparse


# Create the parser
parser = argparse.ArgumentParser()


parser.add_argument('--path_to_clip_input', type=str, default='D:/to_send/clip_05.mp4', help="The path to input clip video")  
parser.add_argument('--path_to_clip_output', type=str, default='D:/to_send/output_clip_05.mp4', help="The path to output clip video")
parser.add_argument('--path_to_json', type=str, default='D:/to_send/clip_05_pose.json', help="The path to the JSON file")
parser.add_argument('--path_to_csv', type=str, default='D:/to_send/clip_05_output.csv', help="The path to the CSV file")


args = parser.parse_args()

path_clip_input = args.path_to_clip_input
path_clip_output = args.path_to_clip_output
path_to_json = args.path_to_json
path_to_csv = args.path_to_csv




# Function to calculate the unit vector and length between two points
def unit_vector_and_length(p1, p2):
    vector = np.array(p2) - np.array(p1)
    length = np.linalg.norm(vector)
    if length > 0:
        unit_vector = vector / length
    else:
        unit_vector = [0, 0]
    return unit_vector[0], unit_vector[1], length

# Function to calculate bounding box center coordinates, width, and height
def bounding_box(points):
    points = np.array(points)
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    return center_x, center_y, width, height

# Load the pose detection JSON data
with open(path_to_json) as f:
    pose_data = json.load(f)

# Prepare the CSV output data
output_data = []

# Open the video file
cap = cv2.VideoCapture(path_clip_input)

# Prepare the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(path_clip_output, fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

frame_id = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    if str(frame_id) in pose_data["frames"]:
        for person_id, person_data in pose_data["frames"][str(frame_id)].items():
            keypoints = person_data['pose_kpts']

            # Extract relevant keypoints for bounding boxes and limb articulation
            def get_keypoint(idx):
                if idx in keypoints and keypoints[idx][2] > 0:
                    return keypoints[idx][:2]
                return [-1, -1]

            nose = get_keypoint("0")
            neck = get_keypoint("1")
            right_shoulder = get_keypoint("2")
            right_elbow = get_keypoint("3")
            right_wrist = get_keypoint("4")
            left_shoulder = get_keypoint("5")
            left_elbow = get_keypoint("6")
            left_wrist = get_keypoint("7")
            mid_hip = get_keypoint("8")
            right_hip = get_keypoint("9")
            right_knee = get_keypoint("10")
            right_ankle = get_keypoint("11")
            left_hip = get_keypoint("12")
            left_knee = get_keypoint("13")
            left_ankle = get_keypoint("14")
            right_eye = get_keypoint("15")
            left_eye = get_keypoint("16")
            right_ear = get_keypoint("17")
            left_ear = get_keypoint("18")

            # Calculate head bounding box (including eyes and ears, excluding neck)
            head_bbox = bounding_box([nose, right_eye, left_eye, right_ear, left_ear])
            
            # Calculate upper body bounding box (neck to hips)
            upper_body_bbox = bounding_box([neck, right_shoulder, left_shoulder, right_hip, left_hip])
            
            # Calculate limb vectors and lengths
            l_shoulder_vec_x, l_shoulder_vec_y, l_upper_arm_len = unit_vector_and_length(left_shoulder, left_elbow)
            l_elbow_vec_x, l_elbow_vec_y, l_lower_arm_len = unit_vector_and_length(left_elbow, left_wrist)
            r_shoulder_vec_x, r_shoulder_vec_y, r_upper_arm_len = unit_vector_and_length(right_shoulder, right_elbow)
            r_elbow_vec_x, r_elbow_vec_y, r_lower_arm_len = unit_vector_and_length(right_elbow, right_wrist)
            
            l_hip_vec_x, l_hip_vec_y, l_upper_leg_len = unit_vector_and_length(left_hip, left_knee)
            l_knee_vec_x, l_knee_vec_y, l_lower_leg_len = unit_vector_and_length(left_knee, left_ankle)
            r_hip_vec_x, r_hip_vec_y, r_upper_leg_len = unit_vector_and_length(right_hip, right_knee)
            r_knee_vec_x, r_knee_vec_y, r_lower_leg_len = unit_vector_and_length(right_knee, right_ankle)

            # Add data to the output
            output_data.append([
                frame_id, person_id,
                head_bbox[0], head_bbox[1], head_bbox[2], head_bbox[3],
                upper_body_bbox[0], upper_body_bbox[1], upper_body_bbox[2], upper_body_bbox[3],
                left_shoulder[0], left_shoulder[1], l_shoulder_vec_x, l_shoulder_vec_y, l_upper_arm_len,
                l_elbow_vec_x, l_elbow_vec_y, l_lower_arm_len,
                right_shoulder[0], right_shoulder[1], r_shoulder_vec_x, r_shoulder_vec_y, r_upper_arm_len,
                r_elbow_vec_x, r_elbow_vec_y, r_lower_arm_len,
                left_hip[0], left_hip[1], l_hip_vec_x, l_hip_vec_y, l_upper_leg_len,
                l_knee_vec_x, l_knee_vec_y, l_lower_leg_len,
                right_hip[0], right_hip[1], r_hip_vec_x, r_hip_vec_y, r_upper_leg_len,
                r_knee_vec_x, r_knee_vec_y, r_lower_leg_len
            ])

            # Draw bounding boxes and lines on the frame
            if nose[0] != -1 and nose[1] != -1:
                cv2.rectangle(frame, (int(head_bbox[0] - head_bbox[2] / 2), int(head_bbox[1] - head_bbox[3] / 2)),
                              (int(head_bbox[0] + head_bbox[2] / 2), int(head_bbox[1] + head_bbox[3] / 2)), (0, 255, 0), 2)
            if neck[0] != -1 and right_hip[0] != -1 and left_hip[0] != -1:
                cv2.rectangle(frame, (int(upper_body_bbox[0] - upper_body_bbox[2] / 2), int(upper_body_bbox[1] - upper_body_bbox[3] / 2)),
                              (int(upper_body_bbox[0] + upper_body_bbox[2] / 2), int(upper_body_bbox[1] + upper_body_bbox[3] / 2)), (255, 0, 0), 2)
            
            # Draw lines for limbs
            def draw_limb(p1, p2, color=(0, 0, 255)):
                if p1[0] != -1 and p2[0] != -1:
                    cv2.line(frame, tuple(map(int, p1)), tuple(map(int, p2)), color, 2)
            
            draw_limb(left_shoulder, left_elbow, color=(0, 0, 255))
            draw_limb(left_elbow, left_wrist,color=(0, 0, 255))
            draw_limb(right_shoulder, right_elbow, color=(0, 0, 255))
            draw_limb(right_elbow, right_wrist, color=(0, 0, 255))
            draw_limb(left_hip, left_knee, color=(0, 0, 255))
            draw_limb(left_knee, left_ankle, color=(0, 0, 255))
            draw_limb(right_hip, right_knee, color=(0, 0, 255))
            draw_limb(right_knee, right_ankle, color=(0, 0, 255))

    out.write(frame)
    frame_id += 1

cap.release()
out.release()

# Create a DataFrame and save to CSV
columns = [
    'frame_ID', 'person_ID',
    'head_center_x', 'head_center_y', 'head_width', 'head_height',
    'body_center_x', 'body_center_y', 'body_width', 'body_height',
    'left_shoulder_x', 'left_shoulder_y', 'left_shoulder_vec_x', 'left_shoulder_vec_y', 'left_upper_arm_length',
    'left_elbow_vec_x', 'left_elbow_vec_y', 'left_lower_arm_length',
    'right_shoulder_x', 'right_shoulder_y', 'right_shoulder_vec_x', 'right_shoulder_vec_y', 'right_upper_arm_length',
    'right_elbow_vec_x', 'right_elbow_vec_y', 'right_lower_arm_length',
    'left_hip_x', 'left_hip_y', 'left_hip_vec_x', 'left_hip_vec_y', 'left_upper_leg_length',
    'left_knee_vec_x', 'left_knee_vec_y', 'left_lower_leg_length',
    'right_hip_x', 'right_hip_y', 'right_hip_vec_x', 'right_hip_vec_y', 'right_upper_leg_length',
    'right_knee_vec_x', 'right_knee_vec_y', 'right_lower_leg_length'
]

df = pd.DataFrame(output_data, columns=columns)

# Ensure no file handle is open to avoid permission errors
csv_output_path = path_to_csv 
try:
    os.remove(csv_output_path)
except OSError:
    pass

df.to_csv(csv_output_path, index=False)

print("Processing complete.")
