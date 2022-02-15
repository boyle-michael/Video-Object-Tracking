import cv2
import sys
import pandas as pd

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split(".")

if __name__ == "__main__":

    filenames = [
        "frigatebird.mp4",
        "Dusty_snow.mp4",
        "Merlin_run.mp4",
        "fall.mp4",
        "kick.mp4",
    ]

    tracker_types = [
        "BOOSTING",
        "MIL",
        "KCF",
        "TLD",
        "MEDIANFLOW",
        "MOSSE",
        "CSRT",
    ]
    for filename in filenames:
        for tracker_type in tracker_types:
            print(f"Using tracker {tracker_type} on {filename}")
            if int(minor_ver) < 3:
                tracker = cv2.Tracker_create(tracker_type)
            else:
                if tracker_type == "BOOSTING":
                    tracker = cv2.legacy.TrackerBoosting_create()
                if tracker_type == "MIL":
                    tracker = cv2.TrackerMIL_create()
                if tracker_type == "KCF":
                    tracker = cv2.TrackerKCF_create()
                if tracker_type == "TLD":
                    tracker = cv2.legacy.TrackerTLD_create()
                if tracker_type == "MEDIANFLOW":
                    tracker = cv2.legacy.TrackerMedianFlow_create()
                if tracker_type == "GOTURN":
                    tracker = cv2.legacy.TrackerGOTURN_create()
                if tracker_type == "MOSSE":
                    tracker = cv2.legacy.TrackerMOSSE_create()
                if tracker_type == "CSRT":
                    tracker = cv2.TrackerCSRT_create()

            # Read video
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

            # Define an initial bounding box
            gt_df = pd.read_csv(f"data/{filename}-gt.csv")
            gt = gt_df.values
            bbox = gt[0]

            # Uncomment the line below to select a different bounding box
            # bbox = cv2.selectROI(frame, False)

            # Initialize tracker with first frame and bounding box
            ok = tracker.init(frame, bbox)

            bboxes = []
            fps_tracker = []
            frame_count = 0

            while True:
                # Read a new frame
                ok, frame = video.read()
                if not ok:
                    break

                # Start timer
                timer = cv2.getTickCount()

                # Update tracker
                ok, bbox = tracker.update(frame)

                if frame_count % 15 == 0:
                    bboxes.append(bbox)
                frame_count += 1

                # Calculate Frames per second (FPS)
                fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
                fps_tracker.append(fps)

            print(f"Writing data for {filename} {tracker_type}")
            avg_fps = sum(fps_tracker) / len(fps_tracker)

            df = pd.DataFrame(bboxes)
            pd.DataFrame.to_csv(
                df,
                f"data/{filename}-{tracker_type}-result.csv",
                header=None,
                index=False,
            )
            with open(f"data/{filename}-{tracker_type}-result.csv", "a") as file:
                file.write(str(avg_fps))

            print("Complete")
