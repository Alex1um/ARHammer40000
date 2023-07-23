import cv2
from playable_robot import Robots, Robot

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
        if all(marker_id in ids for marker_id in corners_ids):
            key = cv2.waitKey(3000) & 0xFF
            if key == 13:  # enter
                break

    # video.release()
    cv2.destroyAllWindows()

    # only markers whose ids in corners_ids will be added
    ids = [marker[0] for marker in markers]
    corner_markers = [markers[ids.index(marker_id)] for marker_id in corners_ids]

    src = np.float32([corner_markers[i][1][i] for i in range(4)])
    dst = np.float32([[0, 0], [0, RESOLUTION[1]], [RESOLUTION[0], RESOLUTION[1]], [RESOLUTION[0], 0]])

    M = cv2.getPerspectiveTransform(src, dst)

    return M, corner_markers


# %% md
# Make interface respond to mouse clicks
# %%
def clickEvents(event, x, y, flags, param):
    robots_corners, robots = param
    for rid, robot in robots.items():
        if event == cv2.EVENT_LBUTTONDOWN:
            if mpl.path.Path(robots_corners[rid]).contains_point((x, y)):
                robot.active = True
                robot.robot.stop()
                robot.robot.led_on()
            else:
                robot.active = False
                robot.robot.stop()
                robot.robot.led_off()
        else:
            if robot.active and event == cv2.EVENT_RBUTTONDOWN:
                robot.robot.stop()
                new_point = approximate_point_to_grid(*RESOLUTION, GRID_WIDTH, GRID_HEIGHT, x, y)
                if robot.point != new_point and robot.robot_grid_point != new_point:
                    robot.point = new_point
                    robot.is_way_found = False


# %% md
# Find robot's marker among all
# %%
def find_robots(markers, robots: Robots):
    ids = [marker[0] for marker in markers]
    robot_markers = dict()
    for rid in robots.keys():
        if rid in ids:
            robot_markers[rid] = markers[ids.index(rid)]
    return robot_markers


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

robots: dict[int, Robots.PlayableRobot] = Robots({
    5: Robot(ROBOT_ADDRESS)
})

# %%
video = cv2.VideoCapture(VIDEO_CAPTURE_DEVICE)
M, corner_markers = get_perspective_matrix(video, arucoDetector, corners_ids)

# %%
corners, ids, shape = find_aruco_markers(video, arucoDetector, M)
# %%
frozen_lake = make_frozen_lake(corners, ids, robots, GRID_WIDTH, GRID_HEIGHT, *shape)

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

    robot_markers = find_robots(markers, robots)

    robot_corners = dict()

    ### START OF INTERFACE CONTROL SECTION
    # if next_img_point:
    #     # show target in aim-like style
    #
    #     # next_img_point = grid_point_to_image_point(next_grid_point)
    #     cv2.circle(img, next_img_point, 3, (0, 255, 0), -1)
    #     cv2.circle(img, next_img_point, 12, (0, 255, 0), 2)
    #     dest = grid_point_to_image_point(point)
    #     cv2.circle(img, dest, 3, (0, 0, 255), -1)
    #     cv2.circle(img, dest, 12, (0, 0, 255), 2)

    for rid, robot_marker in robot_markers.items():
        robot = robots[rid]
        robot_x, robot_y = (robot_marker[1][0] + robot_marker[1][1] + robot_marker[1][2] + robot_marker[1][3]) / 4.0
        robot.robot_grid_point = approximate_point_to_grid(*shape, GRID_WIDTH, GRID_HEIGHT, robot_x, robot_y)
        robot_corners[rid] = robot_marker[1]  # corners are needed for setMouseCallBack

        if robot.active:

            img = aruco_display(robot_marker, img)  # show frames

            if robot.point:
                if not robot.is_way_found:
                    if path_is_complex(RESOLUTION[0]/GRID_WIDTH, corner_markers, (robot_x, robot_y), robot.point):
                        frozen_part = frozen_lake.copy()
                        frozen_part[robot.robot_grid_point[0], robot.robot_grid_point[1]] = 'S'
                        frozen_part[robot.point[0], robot.point[1]] = 'G'
                        env = gym.make('FrozenLake-v1', desc=frozen_part, is_slippery=False)

                        state_space = env.observation_space.n
                        print("There are ", state_space, " possible states")

                        action_space = env.action_space.n
                        print("There are ", action_space, " possible actions")
                        robot.Qtable_frozenlake = get_policy(state_space, action_space, env)
                        robot.state, info = env.reset()
                        robot.next_grid_point = None
                        robot.is_way_found = True
                    else:
                        robot.next_grid_point = robot.point
                        robot.is_way_found = True

                if not robot.next_grid_point:
                    while robot.next_grid_point == robot.robot_grid_point or robot.next_grid_point is None:
                        action = greedy_policy(robot.Qtable_frozenlake, robot.state) # 0: LEFT, 1: DOWN, 2: RIGHT, 3: UP
                        # Take the action (a) and observe the outcome state(s') and reward (r)
                        next_grid_point, reward, terminated, truncated, info = env.step(action)
                        robot.state = next_grid_point
                        robot.next_grid_point = (next_grid_point // GRID_HEIGHT, next_grid_point % GRID_HEIGHT)
                    robot.robot.stop()
                    robot.aimed = False
                    robot.moving = False

                if not robot.next_img_point:
                    robot.next_img_point = grid_point_to_image_point(robot.next_grid_point)

                center, angle = robot_deviation(robot_marker, robot.next_img_point)

                # cv2.line(img, center, point, (0, 255, 0), 2)
                # cv2.putText(img, str(round(angle, 2)),(center[0], center[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if not robot.aimed and not robot.moving and angle >= SEA:
                    robot.moving = True
                    robot.robot.set_speed(TS)
                    robot.robot.turn_right()

                if not robot.aimed and not robot.moving and angle <= SEA:
                    robot.moving = True
                    robot.robot.set_speed(TS)
                    robot.robot.turn_left()

                if not robot.aimed and robot.moving and (angle > -SEA and angle < SEA):
                    robot.robot.stop()
                    robot.aimed = True
                    robot.moving = False

                if robot.aimed and not robot.moving and not mpl.path.Path(corners).contains_point(robot.point):
                    robot.moving = True
                    robot.robot.set_speed(MS)
                    robot.robot.go_forward()

                if mpl.path.Path(corners).contains_point(robot.point):
                    robot.robot.stop()
                    robot.aimed = False
                    robot.moving = False

                if robot.aimed and robot.moving and (angle <= -BEA or angle >= BEA):
                    robot.robot.stop()
                    robot.aimed = False
                    robot.moving = False

                if not robot.aimed and not robot.moving and robot.next_grid_point == robot.robot_grid_point:
                    robot.next_grid_point = None
                    robot.next_img_point = None
                    if robot.robot_grid_point == robot.point:
                        robot.point = None

        else:
            robot.robot.stop()

    else:
        corners = np.zeros((4, 2))
    ### END OF INTERFACE CONTROL SECTION

    cv2.namedWindow('video')
    cv2.setMouseCallback('video', clickEvents, (robot_corners, robots))
    draw_grid(img)
    cv2.imshow('video', img)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        for robot in robots.values():
            robot.robot.stop()
            robot.robot.led_off()
        break

video.release()
cv2.destroyAllWindows()
