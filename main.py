import cv2
import mediapipe as mp
import json
import ffmpeg
import argparse
import os

os.environ["GLOG_minloglevel"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--video', type=str, required=True)
parser.add_argument('-o', '--output', type=str, default='output.json')

args = parser.parse_args()

# abort if the video doesn't exist
if not os.path.exists(args.video):
  print(f'🛑 Video {args.video} does not exist')
  exit(1)

# abort if the output file already exists
if os.path.exists(args.output):
  print(f'🛑 Output file {args.output} already exists')
  exit(1)

print(f'📽️ Processing video {args.video}...')
# first convert the video to 10 fps
(
  ffmpeg
  .input(args.video)
  .filter('fps', fps=10)
  .output('temp.mp4')
  .global_args("-loglevel", "error")
  .global_args("-hide_banner")
  .global_args("-nostdin")
  .run()
)

mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture("temp.mp4")
with mp_pose.Pose(
  min_detection_confidence=0.5,
  min_tracking_confidence=0.5) as pose:

  pose_data = []
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      break

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # turn results.pose_landmarks into a json object
    pose_json = []
    for landmark in results.pose_landmarks.landmark:
      pose_json.append({
        "x": landmark.x,
        "y": landmark.y,
        "z": landmark.z,
        "visibility": landmark.visibility
      })
    pose_data.append(pose_json)
    
    if cv2.waitKey(5) & 0xFF == 27:
      break

  with open(args.output, 'w') as f:
    f.write(json.dumps(pose_data))
cap.release()

# delete the temp file
os.remove('temp.mp4')

print(f'🎉 Pose data successfully saved to {args.output}!')