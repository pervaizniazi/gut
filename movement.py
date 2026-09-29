```python
import cv2
import numpy as np
import matplotlib.pyplot as plt


VIDEO_PATH = (
    "E:\\guts\\data\\3D Perfusion\\Gut Motility\\"
    "Pefusion 3D Colon 1 STW5-II lyo 260820 "
    "treatment STW5-II lyo unten.mov"
)

FRAME_OUTPUT_DIR = "E:\\guts\\plots\\tmp-frames_new"
PLOT_OUTPUT_DIR = "E:\\guts\\plots\\tmp-plots_new"

BOUNDARY_INDEXES = np.array([
    28, 60, 92, 124, 156, 188, 220, 252, 284, 316, 348,
    380, 412, 444, 476, 508, 540, 572, 604, 632, 664,
    696, 728, 760, 792, 824, 858, 890, 922, 954, 986,
    1018, 1050
])

FRAME_SIZE = (1280, 720)
KERNEL = np.ones((20, 20), np.uint8)
DISTANCE_THRESHOLD = 0.03


def calculate_differences(indexes):
    distances = np.zeros(len(BOUNDARY_INDEXES))

    for i, x_position in enumerate(BOUNDARY_INDEXES):
        points = indexes[indexes[:, 1] == x_position]

        if len(points) >= 2:
            lower_point = points[0]
            upper_point = points[-1]
            distances[i] = np.linalg.norm(upper_point - lower_point)

    return distances


def process_frame(frame):
    frame = cv2.resize(
        frame,
        FRAME_SIZE,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(
        gray,
        40,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )

    blurred = cv2.GaussianBlur(binary, (3, 3), 0)
    closed = cv2.morphologyEx(
        blurred,
        cv2.MORPH_CLOSE,
        KERNEL
    )

    distance = cv2.distanceTransform(
        closed,
        cv2.DIST_L2,
        5
    )

    cv2.normalize(
        distance,
        distance,
        0,
        1.0,
        cv2.NORM_MINMAX
    )

    indexes = np.argwhere(
        (distance < DISTANCE_THRESHOLD) &
        (distance > 0)
    )

    indexes = indexes[
        np.isin(indexes[:, 1], BOUNDARY_INDEXES)
    ]

    distances = calculate_differences(indexes)

    for y, x in indexes:
        cv2.circle(
            frame,
            (x, y),
            4,
            (0, 0, 255),
            -1
        )

    return frame, distances


def save_outputs(frame, distances, frame_number):
    cv2.imwrite(
        f"{FRAME_OUTPUT_DIR}\\frame_{frame_number}.png",
        frame
    )

    plot_data = distances.reshape(1, -1)

    plt.imsave(
        f"{PLOT_OUTPUT_DIR}\\movement_{frame_number}.png",
        plot_data,
        vmin=plot_data.min(),
        vmax=200,
        cmap="jet"
    )


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")

    frame_number = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame, distances = process_frame(frame)
        save_outputs(frame, distances, frame_number)

        frame_number += 1

        if frame_number % 100 == 0:
            print(f"Processed {frame_number}/{total_frames} frames")

    cap.release()
    cv2.destroyAllWindows()

    print("done!!")


if __name__ == "__main__":
    main()
```
 
