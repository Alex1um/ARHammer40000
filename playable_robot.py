from RobotAPI import Robot


class Robots(dict):

    class PlayableRobot:

        def __init__(self, robot: Robot, marker_id: int):
            self.robot = robot
            self.marker_id = marker_id
            self.active = False
            self.aimed = False
            self.moving = False
            self.is_way_found = False
            self.point: tuple[int, int] | None = None
            self.robot_grid_point: tuple[int, int] | None = None
            self.next_grid_point = None
            self.next_img_point = None

    def __init__(self, robots: dict[int, Robot]):
        dict.__init__(self)
        for rid, robot in robots.items():
            self[rid] = self.PlayableRobot(robot, rid)

    def is_marker_robot(self, marker_id: int):
        return marker_id in self

