from email import header
from fileinput import filename
import cv2
import sys
import pandas as pd

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split(".")

if __name__ == "__main__":
    # Read video
    filename = "kick.mp4"
    video = cv2.VideoCapture(f"videos/{filename}")

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print("Cannot read video file")
        sys.exit()

    # List of bounding box coordinates
    bboxes = []
    frame_count = 0

    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break
        if frame_count % 15 == 0:
            bbox = cv2.selectROI(frame, False)
            bboxes.append(bbox)
        frame_count += 1

df = pd.DataFrame(bboxes)
pd.DataFrame.to_csv(df, f"data/{filename}-gt.csv", header=None, index=False)
