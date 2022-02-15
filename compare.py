import pandas as pd

filenames = [
    "frigatebird.mp4",
    "Dusty_snow.mp4",
    "Merlin_run.mp4",
    "fall.mp4",
    "kick.mp4",
]

for filename in filenames:
    print(f"Video: {filename}")
    tracker_types = [
        "BOOSTING",
        "MIL",
        "KCF",
        "TLD",
        "MEDIANFLOW",
        "MOSSE",
        "CSRT",
    ]

    gt_df = pd.read_csv(f"data/{filename}-gt.csv")
    gt = gt_df.values

    for tracker in tracker_types:
        results_df = pd.read_csv(f"data/{filename}-{tracker}-result.csv")
        results = results_df.values

        error = []
        for i in range(len(gt)):
            error.append(sum(abs(gt[i] - results[i])) / 4)

        print(
            f"Tracker: {tracker}; AVG error: {sum(error) / len(error):.2f}; AVG FPS: {results[-1][0]:.2f}"
        )
    print()
