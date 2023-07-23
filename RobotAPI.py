import dataclasses

import requests
import urllib3
from enum import Enum


class Robot:

    class States(Enum):
        STOPPED = 0
        MOVING_FORWARD = 1
        MOVING_BACKWARD = 2
        TURNING_LEFT = 3
        TURNING_RIGHT = 4

    def __init__(self, address: str, answer_timeout_extra_time_ms=100):
        self.address = f"http://" + address
        self.answer_timeout_extra_time_ms = answer_timeout_extra_time_ms
        self._state = self.States.STOPPED

    def get_state(self) -> States:
        return self._state

    @staticmethod
    def __try_req(*args, timeout=None, **kwargs):
        try:
            return requests.get(*args, timeout=timeout, **kwargs).status_code
        except requests.exceptions.ConnectionError as e:
            if len(e.args) > 0 and isinstance(e.args[0], urllib3.exceptions.ProtocolError):
                pe: urllib3.exceptions.ProtocolError = e.args[0]
                if str(pe.args[0]) == 'Connection aborted.':
                    return None
            raise e
        except requests.exceptions.ReadTimeout as e:
            if not timeout:
                raise e
            return None

    def turn_left(self):
        self._state = self.States.TURNING_LEFT
        return self.__try_req(self.address + "/go/left", timeout=self.answer_timeout_extra_time_ms)

    def turn_left_for(self, time_ms: int):
        return self.__try_req(self.address + "/go/left", params=[("delay", time_ms)], timeout=(None, (time_ms + self.answer_timeout_extra_time_ms) / 1000))

    def turn_left_sensor(self, sensor: float):
        return self.__try_req(self.address + "/go/left", params=[("sensor", sensor)])

    def turn_right(self):
        self._state = self.States.TURNING_RIGHT
        return self.__try_req(self.address + "/go/right", timeout=self.answer_timeout_extra_time_ms)

    def turn_right_for(self, time_ms: int):
        return self.__try_req(self.address + "/go/right", params=[("delay", time_ms)], timeout=(None, (time_ms + self.answer_timeout_extra_time_ms) / 1000))

    def turn_right_sensor(self, sensor: float):
        return self.__try_req(self.address + "/go/right", params=[("sensor", sensor)])

    def go_forward(self):
        self._state = self.States.MOVING_FORWARD
        return self.__try_req(self.address + "/go/forward")

    def go_forward_for(self, time_ms: int) -> int:
        return self.__try_req(self.address + "/go/forward", params=[("delay", time_ms)], timeout=(None, (time_ms + self.answer_timeout_extra_time_ms) / 1000))

    def go_backward(self):
        self._state = self.States.MOVING_BACKWARD
        return self.__try_req(self.address + "/go/backward", timeout=self.answer_timeout_extra_time_ms)

    def go_backward_for(self, time_ms: int) -> int:
        return self.__try_req(self.address + "/go/backward", params=[("delay", time_ms)], timeout=(None, (time_ms + self.answer_timeout_extra_time_ms) / 1000))

    def stop(self):
        self.__try_req(self.address + "/go/stop", timeout=self.answer_timeout_extra_time_ms)
        self._state = self.States.STOPPED

    def led_on(self):
        self.__try_req(self.address + "/led/on")

    def led_off(self):
        self.__try_req(self.address + "/led/off")

    @dataclasses.dataclass
    class AccelData:
        x: float
        y: float
        z: float
        sqrt: float

        def __iter__(self):
            return iter((self.x, self.y, self.z, self.sqrt))

    def get_accel(self) -> AccelData:
        ret = requests.get(self.address + "/info/accel").json()
        return self.AccelData(ret['x'], ret['y'], ret['z'], ret['sqrt'])

    @dataclasses.dataclass
    class GyroData:
        x: float
        y: float
        z: float

        def __iter__(self):
            return iter((self.x, self.y, self.z))

    def get_gyro(self) -> GyroData:
        ret = requests.get(self.address + "/info/gyro").json()
        return self.GyroData(ret['x'], ret['y'], ret['z'])

    def set_speed(self, speed: int):
        return self.__try_req(self.address + "/speed", params=[("set", speed)])