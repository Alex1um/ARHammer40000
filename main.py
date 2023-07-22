import cv2
from RobotAPI import Robot

import numpy as np
import matplotlib as mpl
import cv2
from const import *
from ai import *
import gymnasium as gym

# %%

# %%
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}


# %% md
# Detect ArUco markers on video and display thier frames and IDs
# %%
def aruco_detect(corners, ids, rejected, image):
    detected = []

    if len(corners) > 0:

        ids = ids.flatten()

        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4, 2))
            corners = np.float32([corners[0], corners[3], corners[2], corners[1]])  # change order to counter-clockwise
            detected.append((int(markerID), corners))

    return sorted(detected)


# %%
def aruco_display(marker, image):
    corners = marker[1]
    topLeft, bottomLeft, bottomRight, topRight = corners

    topRight = (int(topRight[0]), int(topRight[1]))
    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
    bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
    topLeft = (int(topLeft[0]), int(topLeft[1]))

    cv2.line(image, topLeft, bottomLeft, (0, 255, 0), 2)
    cv2.line(image, bottomLeft, bottomRight, (0, 255, 0), 2)
    cv2.line(image, bottomRight, topRight, (0, 255, 0), 2)
    cv2.line(image, topRight, topLeft, (0, 255, 0), 2)

    # cv2.putText(image, str(marker[0]),(topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image


def get_perspective_matrix(video: cv2.VideoCapture, arucoDetector: cv2.aruco.ArucoDetector, corners_ids):

    while video.isOpened():

        ret, img = video.read()

        corners, ids, rejected = arucoDetector.detectMarkers(img)
        markers = aruco_detect(corners, ids, rejected, img)

        for marker in markers:
            img = aruco_display(marker, img)

        cv2.imshow('video', img)

        # break cycle only if all 4 corner markers are detected
        key = cv2.waitKey(100) & 0xFF
        if key == 13 and all([marker_id in ids for marker_id in corners_ids]):
            break

    # video.release()
    cv2.destroyAllWindows()

    # only markers whose ids in corners_ids will be added
    ids = [marker[0] for marker in markers]
    corner_markers = [markers[ids.index(marker_id)] for marker_id in corners_ids]

    src = np.float32([corner_markers[i][1][i] for i in range(4)])
    dst = np.float32([[0, 0], [0, RESOLUTION[1]], [RESOLUTION[0], RESOLUTION[1]], [RESOLUTION[0], 0]])

    M = cv2.getPerspectiveTransform(src, dst)

    return M


# %% md
# Make interface respond to mouse clicks
# %%
def clickEvents(event, x, y, flags, param):
    corners, robot = param
    global active, point, is_way_found, robot_grid_point

    if event == cv2.EVENT_LBUTTONDOWN:
        if mpl.path.Path(corners).contains_point((x, y)):
            active = True
            robot.stop()
            robot.led_on()
        else:
            active = False
            robot.stop()
            robot.led_off()
    else:
        if active and event == cv2.EVENT_RBUTTONDOWN:
            robot.stop()
            new_point = approximate_point_to_grid(*RESOLUTION, GRID_WIDTH, GRID_HEIGHT, x, y)
            if point != new_point and robot_grid_point != new_point:
                point = new_point
            is_way_found = False


# %% md
# Find robot's marker among all
# %%
def find_robot(markers, robot_id):
    ids = [marker[0] for marker in markers]
    if robot_id in ids:
        return markers[ids.index(robot_id)]
    else:
        return None


# %% md
# Auxiliary function for robot
# %%
def robot_deviation(marker, point):
    center = (marker[1][0] + marker[1][1] + marker[1][2] + marker[1][3]) / 4.0
    top_center = (marker[1][0] + marker[1][3]) / 2.0

    d = np.float32(point) - center
    d /= np.linalg.norm(d)
    v = top_center - center
    v /= np.linalg.norm(v)
    angle = np.arccos(np.dot(d, v)) / np.pi * 180
    if d[0] * v[1] - d[1] * v[0] < 0:
        angle *= -1

    return (int(center[0]), int(center[1])), angle


# %% md
# Main block
# %%
aruco_type = "DICT_4X4_50"
arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])
arucoParams = cv2.aruco.DetectorParameters()
arucoDetector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

corners_ids = [1, 2, 3, 4]
robot_id = 5

robot = Robot(ROBOT_ADDRESS)

active = False
aimed = False
moving = False
is_way_found = False

ms = 125  # moving speed
ts = 150  # turning speed
sea = 30  # small enough angle, when stop aiming
bea = 45  # big enough angle, when start reaiming
point_changed = True

point: tuple[int, int] | None = None
robot_grid_point: tuple[int, int] | None = None
# %%
video = cv2.VideoCapture(VIDEO_CAPTURE_DEVICE)
M = get_perspective_matrix(video, arucoDetector, corners_ids)

# %%
corners, ids, shape = find_aruco_markers(video, arucoDetector, M)
# %%
frozen_lake = make_frozen_lake(corners, ids, robot_id, GRID_WIDTH, GRID_HEIGHT, *shape)

# %%
while video.isOpened():

    ret, img = video.read()
    img = cv2.warpPerspective(img, M, RESOLUTION)

    # width = 1920
    # height = 1080
    # img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)

    # corners, ids, rejected = cv2.aruco.detectMarkers(img, arucoDict, parameters=arucoParams)
    corners, ids, rejected = arucoDetector.detectMarkers(img)

    markers = aruco_detect(corners, ids, rejected, img)

    robot_marker = find_robot(markers, robot_id)

    ### START OF INTERFACE CONTROL SECTION
    if point:
        # show target in aim-like style
        cv2.circle(img, point, 3, (0, 255, 0), -1)
        cv2.circle(img, point, 12, (0, 255, 0), 2)

    if robot_marker is not None:
        robot_x, robot_y = (robot_marker[1][0] + robot_marker[1][1] + robot_marker[1][2] + robot_marker[1][3]) / 4.0
        robot_grid_point = approximate_point_to_grid(*shape, GRID_WIDTH, GRID_HEIGHT, robot_x, robot_y)
        corners = robot_marker[1]  # corners are needed for setMouseCallBack

        if active:

            img = aruco_display(robot_marker, img)  # show frames

            if point:
                if not is_way_found:

                    frozen_part = frozen_lake
                    frozen_part[robot_grid_point[0], robot_grid_point[1]] = 'S'
                    frozen_part[point[0], point[1]] = 'G'
                    env = gym.make('FrozenLake-v1', desc=frozen_part, is_slippery=False)

                    state_space = env.observation_space.n
                    print("There are ", state_space, " possible states")

                    action_space = env.action_space.n
                    print("There are ", action_space, " possible actions")
                    Qtable_frozenlake = get_policy(state_space, action_space, env)
                    state, info = env.reset()
                    next_grid_point = None
                    is_way_found = True

                if not next_grid_point:
                    action = greedy_policy(Qtable_frozenlake, state) # 0: LEFT, 1: DOWN, 2: RIGHT, 3: UP
                    # Take the action (a) and observe the outcome state(s') and reward (r)
                    next_grid_point, reward, terminated, truncated, info = env.step(action)
                    next_grid_point = (next_grid_point % GRID_WIDTH, next_grid_point // GRID_HEIGHT)
                    next_img_point = grid_point_to_image_point(next_grid_point)
                    robot.stop()
                    aimed = False
                    moving = False

                center, angle = robot_deviation(robot_marker, next_img_point)

                # cv2.line(img, center, point, (0, 255, 0), 2)
                # cv2.putText(img, str(round(angle, 2)),(center[0], center[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if not aimed and not moving and angle >= sea:
                    moving = True
                    robot.set_speed(ts)
                    robot.turn_right()

                if not aimed and not moving and angle <= sea:
                    moving = True
                    robot.set_speed(ts)
                    robot.turn_left()

                if not aimed and moving and (angle > -sea and angle < sea):
                    robot.stop()
                    aimed = True
                    moving = False

                if aimed and not moving and not mpl.path.Path(corners).contains_point(point):
                    moving = True
                    robot.set_speed(ms)
                    robot.go_forward()

                if mpl.path.Path(corners).contains_point(point):
                    robot.stop()
                    aimed = False
                    moving = False

                if aimed and moving and (angle <= -bea or angle >= bea):
                    robot.stop()
                    aimed = False
                    moving = False

                if not aimed and not moving and next_grid_point == robot_grid_point:
                    next_grid_point = None
        else:
            robot.stop()

    else:
        corners = np.zeros((4, 2))
    ### END OF INTERFACE CONTROL SECTION

    cv2.namedWindow('video')
    cv2.setMouseCallback('video', clickEvents, (corners, robot))
    cv2.imshow('video', img)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        robot.stop()
        robot.led_off()
        break

video.release()
cv2.destroyAllWindows()
