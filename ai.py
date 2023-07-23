import cv2
import numpy as np
import math
import random
from const import *
import matplotlib as mpl


def aruco_display_all(corners, ids, image):
    marker_centers = []
    if len(corners) > 0:
        ids = ids.flatten()
        for (marker_corner, markerID) in zip(corners, ids):
            corners = marker_corner.reshape((4, 2))
            (top_left, top_right, bottom_right, bottom_left) = corners

            top_right = (int(top_right[0]), int(top_right[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
            bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
            top_left = (int(top_left[0]), int(top_left[1]))

            cv2.line(image, top_left, top_right, (0, 255, 0), 2)
            cv2.line(image, top_right, bottom_right, (0, 255, 0), 2)
            cv2.line(image, bottom_right, bottom_left, (0, 255, 0), 2)
            cv2.line(image, bottom_left, top_left, (0, 255, 0), 2)

            c_x = int((top_left[0] + bottom_right[0]) / 2.0)
            c_y = int((top_left[1] + bottom_right[1]) / 2.0)
            cv2.circle(image, (c_x, c_y), 4, (0, 0, 255), -1)
            marker_centers.append((c_x, c_y))
            cv2.putText(image, str(markerID), (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
    return image, marker_centers


def oriented_angle(v0, v1):
    direction = v0 / np.linalg.norm(v0)
    vector = v1 / np.linalg.norm(v1)

    angle = np.arccos(np.dot(direction, vector)) / np.pi * 180
    # sign depends on basis orientation
    if v0[0] * v1[1] - v0[1] * v1[0] < 0:
        angle *= -1

    return angle


def display_path(marker, point, image):
    global busy

    center = (marker[1][0] + marker[1][1] + marker[1][2] + marker[1][3]) / 4.0
    top_center = (marker[1][0] + marker[1][3]) / 2.0

    d = np.float32(point) - center
    d /= np.linalg.norm(d)
    v = top_center - center
    v /= np.linalg.norm(v)
    angle = oriented_angle(d, v)

    center = (int(center[0]), int(center[1]))
    top_center = (int(top_center[0]), int(top_center[1]))

    cv2.line(image, center, point, (0, 255, 0), 2)

    cv2.putText(image, str(round(angle, 2)), (center[0], center[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image, angle  # angle to be removed


# мб здесь проблема, но тут вроде просто. передаю верхние левые и нижние правые углы
def rectangle_intersection(top_left1, bottom_right1, top_left2, bottom_right2):
    x1 = max(top_left1[0], top_left2[0])
    x2 = min(bottom_right1[0], bottom_right2[0])
    y1 = max(top_left1[1], top_left2[1])
    y2 = min(bottom_right1[1], bottom_right2[1])
    if x1 < x2 and y1 < y2:
        return True
    return False


def get_len(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def initialize_q_table(state_space, action_space):
    Qtable = np.zeros((state_space, action_space))
    return Qtable


def greedy_policy(Qtable, state):
    # Exploitation: take the action with the highest state, action value
    action = np.argmax(Qtable[state][:])

    return action


def epsilon_greedy_policy(env, Qtable, state, epsilon):
    # Randomly generate a number between 0 and 1
    random_num = random.uniform(0, 1)
    # if random_num > greater than epsilon --> exploitation
    if random_num > epsilon:
        # Take the action with the highest value given a state
        # np.argmax can be useful here
        action = greedy_policy(Qtable, state)
    # else --> exploration
    else:
        action = env.action_space.sample()

    return action


def get_policy(state_space, action_space, env):
    Qtable = initialize_q_table(state_space, action_space)
    for episode in range(N_TRAINING_EPISODES):
        # Reduce epsilon (because we need less and less exploration)
        epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-DECAY_RATE * episode)
        # Reset the environment
        state, info = env.reset()
        step = 0
        terminated = False
        truncated = False

        # repeat
        for step in range(MAX_STEPS):
            # Choose the action At using epsilon greedy policy
            action = epsilon_greedy_policy(env, Qtable, state, epsilon)

            # Take action At and observe Rt+1 and St+1
            # Take the action (a) and observe the outcome state(s') and reward (r)
            new_state, reward, terminated, truncated, info = env.step(action)

            # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            Qtable[state][action] = Qtable[state][action] + LEARNING_RATE * (
                    reward + GAMMA * np.max(Qtable[new_state]) - Qtable[state][action])

            # If terminated or truncated finish the episode
            if terminated or truncated:
                break

            # Our next state is the new state
            state = new_state
    return Qtable


def cells_initialization(model_to_map, initial_marker, detected_markers):
    top_left, top_right, bottom_right, bottom_left = initial_marker
    cell_centers = {}
    for i in range(GRID_HEIGHT):
        for j in range(GRID_WIDTH):
            v = model_to_map[i * GRID_WIDTH + j]
            cell_center = v + (bottom_left - top_left) / 2 + (top_right - top_left) / 2
            cell_centers[i * GRID_WIDTH + j] = cell_center.astype('int32')
            cv2.circle(detected_markers, (int(cell_center[0]), int(cell_center[1])), 4, (0, 255, 0), -1)
    return cell_centers


def approximate_point_to_grid(img_w, img_h, board_w, board_h, x_click, y_click):
    return int(x_click / img_w * board_w), int(y_click / img_h * board_h)


def make_frozen_lake(corners: tuple, ids, robot_id, board_width, board_height, img_width, img_height):
    frozen_lake = np.full((board_height, board_width), 'F')
    obstacles_corners = []
    if corners and ids.size > 0:
        for (corner, num) in zip(corners, ids):
            for point in corner[0]:
                if num[0] == robot_id:
                    continue
                x0, y0 = point
                i, j = approximate_point_to_grid(img_width, img_height, board_width, board_height, x0, y0)
                frozen_lake[i][j] = 'H'
                obstacles_corners.append(corner)

    # frozen_lake[0, 0] = 'S'
    # frozen_lake[grid_height - 1, grid_width - 1] = 'G'
    return frozen_lake, obstacles_corners


def path_is_complex(cell_size, markers, A, B):
    v1 = np.array([B[0]-A[0], B[1]-A[1]])
    for marker in markers:
        r = 2e9
        for corner in marker:
            v2 = (corner[0]-A[0], corner[1]-A[1])
            v21 = v1 * (np.dot(v1, v2)/np.dot(v1,v1))
            r = min(r, np.linalg.norm(v2 - v21))
        if r <= (cell_size/2):
            return True
    return False


def find_aruco_markers(video: cv2.VideoCapture, detector: cv2.aruco.ArucoDetector, perspective_matrix):

    sure = False
    while not sure:

        ret, img = video.read()
        img = cv2.warpPerspective(img, perspective_matrix, RESOLUTION)

        corners, ids, rejected = detector.detectMarkers(img)

        img_markers, _ = aruco_display_all(corners, ids, img)
        draw_grid(img_markers)

        cv2.imshow("markers", img_markers)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):

            while True:
                key = cv2.waitKey(500) & 0xFF
                if key == ord('q'):
                    sure = True
                break

    return corners, ids, RESOLUTION


def draw_grid(img):
    img_height, img_width, _ = img.shape
    unit_y, unit_x = img_height / GRID_HEIGHT, img_width / GRID_WIDTH
    for i in range(GRID_HEIGHT + 1):
        for j in range(GRID_WIDTH + 1):
            cv2.circle(img, (round(unit_x * j), round(unit_y * i)), 4, (0, 0, 255), 10)
    return img


def grid_point_to_image_point(grid_point: tuple[int, int]):
    unit_x, unit_y = RESOLUTION[0] / GRID_WIDTH, RESOLUTION[1] / GRID_HEIGHT
    return round(grid_point[0] * unit_x + unit_x / 2), round(grid_point[1] * unit_y + unit_y / 2)
