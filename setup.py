import cv2
import numpy as np
import matplotlib.pyplot as plt

#calculate the distance between two points
def calculate_distance(x, y, x1, y1):
    return np.sqrt((x - x1)**2 + (y - y1)**2)

#calculate the focal length
def focal_length(measured_distance, real_width, width_in_rf_image):
    focal_length_value = (width_in_rf_image * measured_distance) / real_width
    return focal_length_value

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
    # 1. Parallel enough
    if angle_between_axes(box1, box2) > max_angle:
        return False

    # 2. Center-to-center vector, split along/across the average axis
    c1 = np.array(box1[0])
    c2 = np.array(box2[0])
    axis = long_axis(box1) + np.sign(np.dot(long_axis(box1), long_axis(box2))) * long_axis(box2)
    axis /= np.linalg.norm(axis)
    normal = np.array([-axis[1], axis[0]])

    d = c2 - c1
    along = abs(np.dot(d, axis))
    across = abs(np.dot(d, normal))

    # 3. Sideways offset must be small compared to the box thickness
    thickness = (min(box1[1]) + min(box2[1])) / 2
    return across <= max_offset_ratio * thickness

def long_side_angle(box):
    (_, _), (w, h), angle = box
    if w < h:           # the long side is the "h" side, so it is rotated 90°
        angle += 90
    return angle % 180  # normalize to [0, 180)

def is_roughly_horizontal(box, tol=20):
    a = long_side_angle(box)
    return a <= tol or a >= 180 - tol


#1ft2in.mp4
#1ft2in2.mp4
#1ft2in_spin.mp4
vid = cv2.VideoCapture('setup_videos/3ft2in.mp4')
_, frame_base = vid.read()

#debug variable
debug = True
#save video variable
save_video = False

# Properties
frame_width = int(vid.get(3))
frame_height = int(vid.get(4))
fps = vid.get(cv2.CAP_PROP_FPS)

# Writer
if save_video:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, fps, (frame_width, frame_height))


#location of the prevous frames rectanges
px1, py1, px2, py2 = 0, 0, 0, 0
while(True):
    min_distance = None
    ret, frame = vid.read()
    if not ret:
        break
    #blank painting for drawing on, same size as the video frames
    painting = np.zeros(frame_base.shape, np.uint8)

    #to find the areas that are the most red, remove areas with high blue and green
    frame_blue, frame_green, frame_red = cv2.split(frame.astype(np.float32))

    
    
    #take the average of blue and green
    blue_green = 0.5*frame_blue + 0.5*frame_green

    # redness: red minus the average of blue and green
    #we want only the pure red areas, so weigh the blue green average more heavily than the red
    redness = frame_red - 1.2*blue_green
    #clip the values to be between 0 and 255 and convert to uint8
    redness = np.clip(redness, 0, 255).astype(np.uint8)

    #threashold and find contours
    _, frame_red_thresh = cv2.threshold(redness, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(frame_red_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_pairs = []
    #if any are returned draw the largest given that it meets a minimum area requirement
    distances = []
    if contours:
        #we need to find two countours that are vaugly rectangular and are the same size, these will be the two green rectangles representing a 1 inch distance between them
        for i, contour1 in enumerate(contours):
            if cv2.contourArea(contour1) > 10:
                #if debug:
                    #painting = cv2.drawContours(painting, [contour1], -1, (0, 255, 0), -1)
                for contour2 in contours[i+1:]:
                    if cv2.contourArea(contour2) > 10:
                
                        #check if the two contours are similar in size
                        area1 = cv2.contourArea(contour1)
                        area2 = cv2.contourArea(contour2)

                        area_diff_percent = abs(area1 - area2) / max(area1, area2)

                        if area_diff_percent < 0.2:
                            #check if the two contours are roughly rectangular
                            box1 = cv2.minAreaRect(contour1)
                            
                            box2 = cv2.minAreaRect(contour2)

                            w1, h1 = box1[1]
                            w2, h2 = box2[1]
                            area_box1 = w1 * h1
                            area_box2 = w2 * h2
                            error1 = abs(area_box1 - area1) / area_box1
                            error2 = abs(area_box2 - area2) / area_box2

                            if w1<h1:
                                w1, h1 = h1, w1
                            if w2<h2:
                                w2, h2 = h2, w2
                                
                            aspect_ratio1 = w1 / h1
                            aspect_ratio2 = w2 / h2

                            angle1 = box1[2]
                            angle2 = box2[2]
                            angle_diff = abs(angle1 - angle2)

                            if error1 < 1 and error2 < 1 and aspect_ratio1 > 3 and aspect_ratio2 > 3 and is_collinear(box1, box2) and is_roughly_horizontal(box1) and is_roughly_horizontal(box2):
                                error = float('inf')
                                
                                minpoints1 = None
                                minpoints2 = None
                                points = cv2.boxPoints(box1)
                                points2 = cv2.boxPoints(box2)
                                #get the minimum two distances between the two sets of points
                                candidates = []

                                for j, p1 in enumerate(points):
                                    for k, p2 in enumerate(points2):
                                        distance = calculate_distance(p1[0], p1[1], p2[0], p2[1])
                                        candidates.append((distance, j, k))

                                best_total = float("inf")
                                best_pair = None

                                for a in range(len(candidates)):
                                    for b in range(a + 1, len(candidates)):
                                        distance1, i1, j1 = candidates[a]
                                        distance2, i2, j2 = candidates[b]

                                        # Each connection must use different corners from both boxes.
                                        if i1 == i2 or j1 == j2:
                                            continue

                                        total_distance = distance1 + distance2
                                        if total_distance < best_total:
                                            best_total = total_distance
                                            best_pair = (candidates[a], candidates[b])

                                if best_pair is not None:
                                    (min_distance1, i1, j1), (min_distance2, i2, j2) = best_pair
                                    minpoints1 = (points[i1], points2[j1])
                                    minpoints2 = (points[i2], points2[j2])
                                min_distance = min(min_distance1, min_distance2)

                                if [px1, py1, px2, py2] != [0, 0, 0, 0]:
                                    #calculate the error as the difference between the previous distance and the current distance
                                    previous_distance = calculate_distance(px1, py1, px2, py2)
                                    error = abs(min_distance - previous_distance) / previous_distance
                                else:
                                    error = (error1 + error2) / 2

                                contour_pairs.append((box1, box2, error, minpoints1, minpoints2))

                                if debug:
                                    painting = cv2.drawContours(painting, [contour1], -1, (0, 255, 0), -1)
                                    painting = cv2.drawContours(painting, [contour2], -1, (0, 255, 0), -1)
        if contour_pairs:
            #sort the pairs by error
            contour_pairs.sort(key=lambda x: x[2])
            best_pair = contour_pairs[0]
            box1, box2, error, minpoints1, minpoints2 = best_pair
            points1 = cv2.boxPoints(box1).astype(np.int32)
            points2 = cv2.boxPoints(box2).astype(np.int32)

            px1, py1 = minpoints1[0]
            px2, py2 = minpoints1[1]

            #draw the two rectangles on the painting
            cv2.polylines(painting, [points1], isClosed=True, color=(0, 0, 255), thickness=2)
            cv2.polylines(painting, [points2], isClosed=True, color=(0, 0, 255), thickness=2)

            #draw a line between the two closest points of the two rectangles
            if minpoints1 is not None:
                cv2.line(painting, (int(minpoints1[0][0]), int(minpoints1[0][1])), (int(minpoints1[1][0]), int(minpoints1[1][1])), (255, 0, 0), 2)
            
            if minpoints2 is not None:
                cv2.line(painting, (int(minpoints2[0][0]), int(minpoints2[0][1])), (int(minpoints2[1][0]), int(minpoints2[1][1])), (125, 0, 125), 2)

            dist1 = calculate_distance(int(minpoints2[0][0]), int(minpoints2[0][1]), int(minpoints2[1][0]), int(minpoints2[1][1]))
            dist2 = calculate_distance(int(minpoints1[0][0]), int(minpoints1[0][1]), int(minpoints1[1][0]), int(minpoints1[1][1]))
            min_distance = min(dist1, dist2)
            #print(f"min_distance: {min_distance}")
            distances.append(min_distance)

    #display diferent stages of the processing if debug is true
    if debug:
        cv2.imshow("Red", frame_red)
        cv2.imshow("Threshold", frame_red_thresh)
        cv2.imshow("Painting", painting)

    #add the painting to the original frame with some transparency
    

    if save_video:
        out.write(frame)
    #display and break
    

    print(f"min_distance: {min_distance}")
    """
    if min_distance is not None and (min_distance < 150 or min_distance > 200):
        print("pausing")
        for countors in contour_pairs:
            box1, box2, error, minpoints1, minpoints2 = countors
            points1 = cv2.boxPoints(box1).astype(np.int32)
            points2 = cv2.boxPoints(box2).astype(np.int32)

            #draw the two rectangles on the painting
            cv2.polylines(painting, [points1], isClosed=True, color=(0, 0, 255), thickness=2)
            cv2.polylines(painting, [points2], isClosed=True, color=(0, 0, 255), thickness=2)

            #draw a line between the two closest points of the two rectangles
            if minpoints1 is not None:
                cv2.line(painting, (int(minpoints1[0][0]), int(minpoints1[0][1])), (int(minpoints1[1][0]), int(minpoints1[1][1])), (255, 0, 0), 2)
            
            if minpoints2 is not None:
                cv2.line(painting, (int(minpoints2[0][0]), int(minpoints2[0][1])), (int(minpoints2[1][0]), int(minpoints2[1][1])), (125, 0, 125), 2)
        frame = cv2.addWeighted(frame, 1, painting, 0.8, 0)
        cv2.imshow("video",frame)
        key = cv2.waitKey(0) & 0xFF  # Wait for a key before continuing
    else:
    """
    frame = cv2.addWeighted(frame, 1, painting, 0.8, 0)
    cv2.imshow("video",frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

focal = focal_length(36, 2, np.mean(distances))
#2ft2in = 1003
print(f"Focal Length: {focal}")

vid.release()
if save_video:
    out.release()
cv2.destroyAllWindows()
