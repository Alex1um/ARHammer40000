import cv2
import numpy as np
import math
import random
from const import *


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


def approximate_point_to_grid(initial_marker, x, y):
    top_left, top_right, bottom_right, bottom_left = initial_marker
    x0, y0 = (x, y)
    x_const, y_const = top_left
    x_a, y_a = top_right - top_left
    x_b, y_b = bottom_left - top_left
    j = round(((x0 * y_b) / x_b - y0 + y_const - (x_const * y_b) / x_b) / ((x_a * y_b) / x_b - y_a) - 0.5)
    i = round(((x0 * y_a) / x_a - y0 + y_const - (x_const * y_a) / x_a) / ((x_b * y_a) / x_a - y_b) - 0.5)
    return i, j


def make_frozen_lake(id, initial_marker, marker_centers, ids, robot_id):
    top_left, top_right, bottom_right, bottom_left = initial_marker
    frozen_lake = np.full((GRID_HEIGHT, GRID_WIDTH), 'F')
    for (dot, num) in zip(marker_centers, ids):
        if num == id or num == robot_id:
            continue
        x0, y0 = dot
        i, j = approximate_point_to_grid(initial_marker, x0, y0)
        frozen_lake[i][j] = 'H'
        # for d1 in range(0, 2):
        #     for d2 in range(0, 2):
        #         frozen_lake[i + d1][j + d2] = 'H'

    # frozen_lake[0, 0] = 'S'
    # frozen_lake[grid_height - 1, grid_width - 1] = 'G'
    return frozen_lake


def build_map(id, corners, ids, height, width):
    model_to_map = {}
    for (marker, num) in zip(corners, ids):
        if num == id:
            marker_corner = marker
            break

    marker_corner = marker_corner.reshape((4, 2))
    y_sorted = sorted(marker_corner, key=lambda x: x[1])
    top_left = min(y_sorted[:2], key=lambda x: x[0])
    top_right = max(y_sorted[:2], key=lambda x: x[0])
    bottom_left = min(y_sorted[-2:], key=lambda x: x[0])
    bottom_right = min(y_sorted[-2:], key=lambda x: x[0])
    # (top_left, top_right, bottom_right, bottom_left) = sorted(marker_corner, key=lambda x: (x[1],x[0]))
    top_left = np.array(top_left)
    top_right = np.array(top_right)
    bottom_left = np.array(bottom_left)
    bottom_right = np.array(bottom_right)
    for i in range(height):
        for j in range(width):
            dot = top_left + (top_right - top_left) * j + (bottom_left - top_left) * i
            model_to_map[i * width + j] = dot.astype('int32')
    return model_to_map, (top_left, top_right, bottom_right, bottom_left)


def path_is_complex(A, B, markers):
    # TODO
    pass


def mark_up_map(cap: cv2.VideoCapture, detector: cv2.aruco.ArucoDetector, perspective_matrix):
    """
    1-й этап: размечаем карту по одному кубику
    :return:
    """

    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, img = cap.read()
        img = cv2.warpPerspective(img, perspective_matrix, RESOLUTION)

        h, w, _ = img.shape

        # width = 1000
        # height = int(width * (h / w))
        # img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
        corners, ids, rejected = detector.detectMarkers(img)

        detected_markers, marker_centers = aruco_display_all(corners, ids, img)
        model_to_map, initial_marker = build_map(corners, ids, GRID_HEIGHT, GRID_WIDTH)
        for val in model_to_map.values():
            cv2.circle(detected_markers, tuple(val), 4, (0, 0, 255), -1)
        cv2.imshow("Image", detected_markers)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return model_to_map, initial_marker, detected_markers, marker_centers
            break


def place_cubes(cap: cv2.VideoCapture, detector: cv2.aruco.ArucoDetector, perspective_matrix, model_to_map, initial_marker,
                detected_markers, robot_id):
    """
    2-й этап: выставляем все кубики
    """

    cell_centers = cells_initialization(model_to_map, initial_marker, detected_markers)
    #
    while True:
        ret, img = cap.read()
        img = cv2.warpPerspective(img, perspective_matrix, RESOLUTION)

        h, w, _ = img.shape

        # width = 1000
        # height = int(width * (h / w))
        # img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)

        corners, ids, rejected = detector.detectMarkers(img)

        detected_markers, marker_centers = aruco_display_all(corners, ids, img)
        for v in model_to_map.values():
            cv2.circle(detected_markers, v, 4, (0, 0, 255), -1)
        for v in cell_centers.values():
            cv2.circle(detected_markers, v, 4, (0, 255, 0), -1)
        cv2.imshow("Image", detected_markers)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    frozen_lake = make_frozen_lake(INITIAL_ID, initial_marker, marker_centers, ids, robot_id)
    return frozen_lake
