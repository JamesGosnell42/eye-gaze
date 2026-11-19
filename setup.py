import sys
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
ft = 1
inch = 2
DEFAULT_VIDEO = 'setup_videos/' +str(ft)+'ft'+str(inch)+'in.mp4'
DEFAULT_KNOWN_DISTANCE = 12*ft   # real distance from the camera to the target (inches)
DEFAULT_REAL_WIDTH = inch        # real distance between the two rectangles (inches)

REDNESS_THRESHOLD = 1         # minimum "redness" for a pixel to count
MIN_CONTOUR_AREA = 10         # ignore contours smaller than this (pixels^2)
MAX_AREA_DIFF = 0.2           # max relative area difference between the two rectangles
MAX_RECT_ERROR = 0.3          # max (box_area - contour_area) / box_area, i.e. how "rectangular"
MIN_ASPECT_RATIO = 3          # long side / short side

debug = True        # show the intermediate processing windows
save_video = False  # write the annotated video to output.avi


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def calculate_distance(p1, p2):
    """Distance between two (x, y) points."""
    return float(np.linalg.norm(np.asarray(p1, dtype=np.float64) - np.asarray(p2, dtype=np.float64)))


def focal_length(measured_distance, real_width, width_in_rf_image):
    return (width_in_rf_image * measured_distance) / real_width


def long_axis(box):
    """Unit vector along the box's long side."""
    (_, _), (w, h), angle = box
    if w < h:
        angle += 90
    rad = np.deg2rad(angle)
    return np.array([np.cos(rad), np.sin(rad)])


def angle_between_axes(box1, box2):
    """Smallest angle (degrees) between two long axes, ignoring direction."""
    d = abs(np.dot(long_axis(box1), long_axis(box2)))
    return np.degrees(np.arccos(np.clip(d, 0, 1)))


def is_collinear(box1, box2, max_angle=15, max_offset_ratio=1.0):
    """True if the two boxes are roughly parallel and lie along the same line."""
    # 1. Parallel enough
    if angle_between_axes(box1, box2) > max_angle:
        return False

    # 2. Split the center-to-center vector into along/across the average axis
    axis1 = long_axis(box1)
    axis2 = long_axis(box2)
    axis = axis1 + np.sign(np.dot(axis1, axis2)) * axis2
    axis /= np.linalg.norm(axis)
    normal = np.array([-axis[1], axis[0]])

    d = np.array(box2[0]) - np.array(box1[0])
    across = abs(np.dot(d, normal))

    # 3. Sideways offset must be small compared to the box thickness
    thickness = (min(box1[1]) + min(box2[1])) / 2
    return across <= max_offset_ratio * thickness


def long_side_angle(box):
    (_, _), (w, h), angle = box
    if w < h:           # the long side is the "h" side, so it is rotated 90 degrees
        angle += 90
    return angle % 180  # normalize to [0, 180)


def is_roughly_horizontal(box, tol=20):
    a = long_side_angle(box)
    return a <= tol or a >= 180 - tol


def closest_corner_connections(box1, box2):
    """
    Find the two shortest connections between the corners of two boxes, where
    each connection uses a different corner of each box.
    Returns [(p_from_box1, p_from_box2, distance), ...] sorted shortest first,
    or None if no valid pair of connections exists.
    """
    points1 = cv2.boxPoints(box1)
    points2 = cv2.boxPoints(box2)

    candidates = []
    for ci, p1 in enumerate(points1):
        for cj, p2 in enumerate(points2):
            candidates.append((calculate_distance(p1, p2), ci, cj))

    best_total = float("inf")
    best = None
    for a in range(len(candidates)):
        for b in range(a + 1, len(candidates)):
            distance1, ci1, cj1 = candidates[a]
            distance2, ci2, cj2 = candidates[b]

            # each connection must use different corners from both boxes
            if ci1 == ci2 or cj1 == cj2:
                continue

            total = distance1 + distance2
            if total < best_total:
                best_total = total
                best = (candidates[a], candidates[b])

    if best is None:
        return None

    connections = [(points1[ci], points2[cj], dist) for dist, ci, cj in best]
    connections.sort(key=lambda c: c[2])
    return connections


def draw_pair(painting, box1, box2, connections):
    """Draw both boxes and the two corner-to-corner connections."""
    cv2.polylines(painting, [cv2.boxPoints(box1).astype(np.int32)], True, (0, 0, 255), 2)
    cv2.polylines(painting, [cv2.boxPoints(box2).astype(np.int32)], True, (0, 0, 255), 2)

    colors = [(255, 0, 0), (125, 0, 125)]
    for (pa, pb, _), color in zip(connections, colors):
        cv2.line(painting, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), color, 2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO
    known_distance = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_KNOWN_DISTANCE
    real_width = float(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_REAL_WIDTH

    #get hight from camera
    
    height_cam = None
    height = None
    length = None 

    while True:
        x = input("Enter height from camera to top of screen in centimeters:")
        try:
            x = float(x);
            height_cam = x
            break
        except:
            print("Wrong input, please use a number.")

    while True:
        x = input("Enter height of screen in centimeters:")
        try:
            x = float(x);
            height = x
            break
        except:
            print("Wrong input, please use a number.")

    while True:
        x = input("Enter length of screen in centimeters:")
        try:
            x = float(x);
            length = x
            break
        except:
            print("Wrong input, please use a number.")


    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    # Writer
    out = None
    if save_video:
        frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vid.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, fps, (frame_width, frame_height))

    # the measured distance of the best pair from every frame
    distances = []
    # distance measured in the previous frame, used to prefer consistent matches
    previous_distance = None

    while True:
        ret, frame = vid.read()
        if not ret:
            break

        # blank painting for drawing on, same size as the video frame
        painting = np.zeros_like(frame)

        # to find the areas that are the most red, remove areas with high blue and green
        frame_blue, frame_green, frame_red = cv2.split(frame.astype(np.float32))

        # take the average of blue and green
        blue_green = 0.5 * frame_blue + 0.5 * frame_green

        # redness: red minus the average of blue and green
        # we want only the pure red areas, so weigh the blue green average more heavily than the red
        redness = frame_red - 1.2 * blue_green
        # clip the values to be between 0 and 255 and convert to uint8
        redness = np.clip(redness, 0, 255).astype(np.uint8)

        # threshold and find contours
        _, frame_red_thresh = cv2.threshold(redness, REDNESS_THRESHOLD, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(frame_red_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Checks that only involve one contour are done once up front:
        # big enough, rectangular, long and thin, and roughly horizontal.
        valid = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area <= MIN_CONTOUR_AREA:
                continue

            box = cv2.minAreaRect(contour)
            w, h = sorted(box[1], reverse=True)  # w = long side, h = short side
            box_area = w * h
            if h == 0 or box_area == 0:
                continue

            rect_error = abs(box_area - area) / box_area
            if rect_error >= MAX_RECT_ERROR:
                continue
            if w / h <= MIN_ASPECT_RATIO:
                continue
            if not is_roughly_horizontal(box):
                continue

            valid.append((contour, area, box, rect_error))

        # we need to find two contours that are vaguely rectangular and are the same size,
        # these will be the two rectangles representing a known distance between them
        contour_pairs = []
        for i, (contour1, area1, box1, error1) in enumerate(valid):
            for contour2, area2, box2, error2 in valid[i + 1:]:
                # check if the two contours are similar in size
                if abs(area1 - area2) / max(area1, area2) >= MAX_AREA_DIFF:
                    continue

                # check that the two boxes are lined up with each other
                if not is_collinear(box1, box2):
                    continue

                connections = closest_corner_connections(box1, box2)
                if connections is None:
                    continue

                # connections are sorted shortest first
                pair_distance = connections[0][2]

                if previous_distance:
                    # prefer the pair whose distance is closest to the previous frame's
                    error = abs(pair_distance - previous_distance) / previous_distance
                else:
                    error = (error1 + error2) / 2

                contour_pairs.append((box1, box2, error, connections))

                if debug:
                    cv2.drawContours(painting, [contour1], -1, (0, 255, 0), -1)
                    cv2.drawContours(painting, [contour2], -1, (0, 255, 0), -1)

        min_distance = None
        if contour_pairs:
            # the best pair is the one with the lowest error
            box1, box2, error, connections = min(contour_pairs, key=lambda p: p[2])
            draw_pair(painting, box1, box2, connections)

            min_distance = connections[0][2]
            previous_distance = min_distance
            distances.append(min_distance)
        else:
            # lost the target, don't compare against a stale position
            previous_distance = None

        # display different stages of the processing if debug is true
        if debug:
            cv2.imshow("Red", frame_red.astype(np.uint8))
            cv2.imshow("Threshold", frame_red_thresh)
            cv2.imshow("Painting", painting)

        print(f"min_distance: {min_distance}")

        # add the painting to the original frame with some transparency
        frame = cv2.addWeighted(frame, 1, painting, 0.8, 0)
        cv2.imshow("video", frame)

        if out is not None:
            out.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    vid.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()

    if distances:
        focal = focal_length(known_distance, real_width, np.median(distances))
        file_path = "config.txt"
        with open(file_path, 'w') as file:
            file.write(f"focal lenth:{focal}\nCamera Height:{height_cam}\nScreen Height:{height}\nScreen Length:{length}")
        print(f"Focal Length: {focal}")
    else:
        print("No rectangle pair was detected, cannot compute the focal length.")



if __name__ == "__main__":
    main()
