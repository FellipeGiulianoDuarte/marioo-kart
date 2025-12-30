import math

MAX_ANGLE_VELOCITY = 0.05
MAX_ACCELERATION = 0.25
BOOST_VELOCITY = 15

class KartPhysics:
    def __init__(self):
        self.last_speed = (0, 0)
        self.current_acceleration = 0

    def update_angle(self):
        return math.atan2(self.last_speed[1], self.last_speed[0])

    def update_speed(self):
        return math.sqrt(self.last_speed[0] ** 2 + self.last_speed[1] ** 2)

    def calculate_current_acceleration(self, friction):
        last_speed = self.update_speed()
        prev_angle = self.update_angle()
        current_accel = self.current_acceleration - friction * last_speed * math.cos(self.angle - prev_angle)
        return current_accel

    def calculate_current_speed(self, friction):
        last_speed = self.update_speed()
        current_acceleration = self.calculate_current_acceleration(friction)
        return current_acceleration + last_speed

    def calculate_x(self, friction, velocity=None):
        if velocity is None:
            current_speed = self.calculate_current_speed(friction)
        else:
            current_speed = velocity

        speed_x = current_speed * math.cos(self.angle)
        self.last_speed = (speed_x, self.last_speed[1])
        return self.last_position[0] + speed_x

    def calculate_y(self, friction, velocity=None):
        if velocity is None:
            current_speed = self.calculate_current_speed(friction)
        else:
            current_speed = velocity

        speed_y = current_speed * math.sin(self.angle)
        self.last_speed = (self.last_speed[0], speed_y)
        return self.last_position[1] + speed_y