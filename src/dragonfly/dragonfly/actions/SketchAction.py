#!/usr/bin/env python3
import math
import rx
import rx.operators as ops
import numpy as np

from geometry_msgs.msg import TwistStamped
from std_msgs.msg import String
from dragonfly_messages.msg import LatLon, PositionVector, SemaphoreToken
from .ActionState import ActionState

EARTH_CIRCUMFERENCE = 40008000

def position_equals(p1, p2):
    return p1.latitude == p2.latitude and p1.longitude == p2.longitude and p1.value == p2.value

class Drone:

    def __init__(self, threshold, latest_gradient, distance_sqrt_lambda):
        self.threshold = threshold
        self.position_reading_queue = []
        self.gradient = latest_gradient
        self.crossing_point = None
        self.before_crossing = None
        self.position = None
        self.distance_sqrt_lambda = distance_sqrt_lambda


    def update(self, position, partner_drone):

        if position.value > 420 and (len(self.position_reading_queue) == 0 or not position_equals(position, self.position_reading_queue[-1])):
            self.position_reading_queue.append(position)

        while len(self.position_reading_queue) > 3 and magnitude(difference_in_meters(self.position_reading_queue[0], self.position_reading_queue[-1])) > self.distance_sqrt_lambda:
            self.position_reading_queue.pop(0)

        if self.position is not None and ((position.value > self.threshold) == (self.position.value <= self.threshold)) and len(self.position_reading_queue) >= 3:
            self.crossing_point = position if math.fabs(position.value - self.threshold) > math.fabs(self.position.value - self.threshold) else self.position
            self.gradient = calculate_lsq(self.position_reading_queue + partner_drone.position_reading_queue)

            self.position = position

            return [self.crossing_point, self.before_crossing, self.gradient]

        self.position = position
        return None

    def update_before_crossing(self):
        self.before_crossing = self.position

    def clear_crossing(self):
        self.before_crossing = None


class ReadingPosition:

    def __init__(self, latitude, longitude, value):
        self.latitude = latitude
        self.longitude = longitude
        self.value = value

    def __str__(self):
        return f"{self.latitude} {self.longitude}, v:{self.value}"

    def __eq__(self, other):
        return self.latitude == other.latitude and self.longitude == other.longitude and self.value == other.value

def calculate_lsq(reading_queue):
    if len(reading_queue) < 3:
        raise ValueError("At least 3 readings are required")

    n = len(reading_queue)
    matrix = []
    values = []

    for i in range(n):
        # if i == 1:
        #     continue

        diff = difference_in_meters(reading_queue[i], reading_queue[1])
        # diff[1] = diff[1] / (diff[0] + 1e-9)

        value = reading_queue[i].value - reading_queue[1].value
        # value = value / (diff[0] + 1e-9)
        values.append(value)
        # diff[0] = 1
        matrix.append(diff)

    m = np.linalg.lstsq(matrix, values, rcond=None)[0]

    return unitary(m)

def calculate_angle(one, two):
    intermediate = ((one[0] * two[0]) + (one[1] * two[1])) / (np.linalg.norm(one) * np.linalg.norm(two))
    if intermediate > 1 or intermediate < -1:
        intermediate = 1
    return math.acos(intermediate)

def difference_in_meters(one, two):
    return [
        ((one.longitude - two.longitude) * (EARTH_CIRCUMFERENCE / 360) * math.cos(math.radians(one.latitude))),
        ((one.latitude - two.latitude) * (EARTH_CIRCUMFERENCE / 360))
    ]

def latlon_plus_meters(position, offset):
    longitude = position.longitude + ((offset[0]) / (math.cos(math.radians(position.latitude)) * (EARTH_CIRCUMFERENCE / 360)))
    latitude = position.latitude + ((offset[1]) / (EARTH_CIRCUMFERENCE / 360))
    return createLatLon(latitude, longitude)

def unitary(vector):
    vector_magnitude = np.linalg.norm(vector)
    if vector_magnitude == 0 and vector[0] == 0:
        return vector
    return np.array([vector[0] / vector_magnitude, vector[1] / vector_magnitude])

def rotate_vector(vector, angle):
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    return np.dot(rotation_matrix, vector)

def createLatLon(latitude, longitude):
    return LatLon(latitude=latitude, longitude=longitude, relative_altitude=0.0)

def average(one, two):
    return createLatLon((one.latitude + two.latitude) / 2, (one.longitude + two.longitude)/2)

def magnitude(vector):
    return np.linalg.norm(vector)

def calculate_turn_center(token, distance_lambda):
    direction_vector = [token.x, token.y]
    rotated_direction_vector = rotate_vector(direction_vector, token.a)

    opposite = np.array(unitary(rotated_direction_vector)) * (distance_lambda / 2) # lambda / 2

    hypotenuse = np.array(unitary(rotate_vector(rotated_direction_vector, math.pi / 2))) * ((distance_lambda / 2) / math.sin(token.a / 2))

    center_offset = opposite + hypotenuse

    center = latlon_plus_meters(token.position, center_offset)

    return center

def interpolate_crossing(threshold, d1, d2):
    position1 = d1.position
    position2 = d2.position
    min_value = min(d1.value, d2.value)
    max_value = max(d1.value, d2.value)

    if min_value == max_value:
        # Avoid division by zero if start_value equals end_value
        threshold_fraction = 0.5
    else:
        threshold_fraction = (threshold - min_value) / (max_value - min_value)

    interpolated_lat = position1.latitude + (position2.latitude - position1.latitude) * threshold_fraction
    interpolated_lon = position1.longitude + (position2.longitude - position1.longitude) * threshold_fraction

    return createLatLon(interpolated_lat, interpolated_lon)

class Sketch:

    INITIAL_ORIENTATION = True

    def __init__(self, distance_sqrt_lambda, lambda_value, threshold, starting_direction=None, latest_gradient=None):
        self.distance_sqrt_lambda = distance_sqrt_lambda
        self.distance_lambda = self.distance_sqrt_lambda * lambda_value / (math.sqrt(lambda_value))
        self.lambda_value = lambda_value
        self.threshold = threshold
        self.starting_direction = starting_direction

        self.drone1 = Drone(threshold, latest_gradient, distance_sqrt_lambda)
        self.drone2 = Drone(threshold, latest_gradient, distance_sqrt_lambda)

        self.encountered = False
        self.armed = False
        self.starting_location = None
        self.position_reading_queue = []
        self.latest_crossing_point = None
        self.latest_point_before_crossing = None
        self.latest_gradient = latest_gradient
        self.latest_reading_queue = []

        self.flip = False

    def update(self, d1, d2):

        update_result = self.drone2.update(d2, self.drone1)
        if update_result is not None:
            [self.latest_crossing_point, self.latest_point_before_crossing, self.latest_gradient] = update_result
        update_result = self.drone1.update(d1, self.drone2)
        if update_result is not None:
            [self.latest_crossing_point, self.latest_point_before_crossing, self.latest_gradient] = update_result

        if not self.encountered and (self.inside(self.drone1) or self.inside(self.drone1)):
            self.encountered = True
            self.starting_location = average(self.drone1.position, self.drone2.position)

        if self.encountered and magnitude(difference_in_meters(average(self.drone1.position, self.drone2.position), self.starting_location)) > (2 * self.distance_sqrt_lambda):
            self.armed = True

        self.position_reading_queue.append(d1)
        self.position_reading_queue.append(d2)

        while len(self.position_reading_queue) > 200:
            self.position_reading_queue.pop(0)

    # Algorithm 3 Initially, robots are √λ apart; one inside and one outside
    def boundary_sketch(self):

        while self.drone1.position is None or self.drone2.position is None:
            yield None

        self.target_position_vector = self.initial_direction(self.drone1, self.drone2)
        yield from self.until(self.target_position_vector, lambda: self.encountered)

        while self.incomplete():
            # D1, D2 ← the two robots
            # ∇ ← boundary gradient at point of crossing with line segment between D1 and D2
            # α ← √λ
            lambda_value = self.lambda_value
            if self.inside(self.drone1) ^ self.inside(self.drone2):
                # D1 and D2 both move λ distance in the direction of ∇
                # print("Sandwich")
                self.target_position_vector = self.forward(self.drone1, self.drone2)

                # print(f"Sandwich: {self.target_position_vector} {self.d1} {self.d2}")
                yield from self.until(self.target_position_vector, lambda: self.moved_forward_lambda(self.target_position_vector, self.drone1, self.drone2))

            if not self.inside(self.drone1) and not self.inside(self.drone2):
                # print("Outside")
                a = math.sqrt(lambda_value)
                # print(f"Outside: {a}")
                yield from self.cross_boundary(self.drone1, self.drone2, a)
            elif self.inside(self.drone1) and self.inside(self.drone2):
                # print("Inside")
                a = -math.sqrt(lambda_value)
                # print(f"Inside: {a}")
                yield from self.cross_boundary(self.drone2, self.drone1, a)

    # Algorithm 2 Reestablishes “Sandwich” Invariant
    def cross_boundary(self, drone1:Drone, drone2:Drone, a):
        if self.flip:
            drone1, drone2 = drone2, drone1
        was_inside = self.inside(drone1)
        # print(f"turning: {self.inside(self.d1)} {self.inside(self.d2)} was inside: {was_inside}")

        #   p ← last position of D1 before crossing
        #   R ← the vertices of the regular polygon including D1’s position with exterior angle √λ and
        #       the edge beginning at D1’s position facing the direction of ∇ + α.
        #   P ← the vertices of the convex hull of R ∪ {p}. For all i : 0 ≤ i ≤ |P| − 1, let
        #       Pi be the i-th vertex in this convex hull, ordered such that P0 = p and P1 = D1’s current position.
        #   ∇ ← gradient at the last boundary crossing of D1
        #   i ← 1.

        def angle_to(pv, p, t):
            return calculate_angle([pv.x, pv.y], difference_in_meters(p, t))

        def polygon_facing_crossing():
            if self.latest_point_before_crossing is not None:
                angle = angle_to(self.target_position_vector, self.latest_point_before_crossing, drone1.position)
                #                 print(f"cone: {angle} < {a}")
                return angle < math.fabs(a)
            return False

        i = 1
        #   while neither robot has crossed the boundary AND i + 1 < |P| do
        while (self.inside(drone1) == was_inside and self.inside(drone2) == was_inside and
               i < math.fabs(2*math.pi / a) and
               not polygon_facing_crossing()):
            #   D1 moves to Pi+1.
            #   D2 moves to closest point from it that is √λ distance away from Pi and orthogonal to ∇ + iα
            self.target_position_vector = self.turn(drone1, drone2, a)
            yield from self.until(self.target_position_vector, lambda: self.turned_lambda(self.target_position_vector, self.distance_lambda, i, drone1, drone2))
            #   i ← i + 1
            i = i + 1

        # print("crossed boundary", i < math.fabs(2*math.pi / a), i, math.fabs(2*math.pi / a), a)

        # print(f"angle_to: {angle_to(self.target_position_vector, self.d1_before_crossing, self.d1)} {i * a} {polygon_facing_crossing()}")

        #   while neither robot has crossed the boundary do
        while self.inside(drone1) == was_inside and self.inside(drone2) == was_inside:
            print("back to point")
            #       D1 moves towards point p taking steps of length λ.
            #       D2 moves to closest point from it that is √λ distance away from D1 and orthogonal to D1’s direction.
            #   if self.latest_point_before_crossing is not None:
            direction_back = unitary(difference_in_meters(self.latest_point_before_crossing, drone1.position))
            self.target_position_vector = self.forward_by_vector(self.drone1, self.drone2, direction_back)
            yield from self.until(self.target_position_vector, lambda: self.moved_forward_lambda(self.target_position_vector, self.drone1, self.drone2))
        #   if D2 crossed the boundary then
        if self.inside(drone2) != was_inside:
            # SYNCHRONIZE (D1, D2)
            yield from self.synchronize(drone1, drone2, a > 0 if was_inside else a < 0)
        #   else
        #   ∇ ← the current direction of D1.

    def synchronized(self, position_vector, drone1:Drone, drone2:Drone):

        distance_to_center = magnitude(difference_in_meters(average(drone1.position, drone2.position), position_vector.position))
        angle_to_direction = math.fabs(calculate_angle([position_vector.x, position_vector.y], difference_in_meters(drone1.position, drone2.position))) + (math.pi / 2)

        while angle_to_direction > math.pi / 2:
            angle_to_direction -= math.pi

        return distance_to_center < 1 and angle_to_direction < 0.2

    # Algorithm 1 Ensures the robots are at distance √λ from each other and are oriented in the same direction.
    def synchronize(self, drone1:Drone, drone2:Drone, left):
        #         print("synchronize")
        #   Path ← the polyline path of D2 from last crossing of BOUNDARY-SKETCH with the shape till current position.
        #   ∇ ← the gradient at the last boundary crossing for D2.
        #   L1 ← the line in the direction of ∇ through D1’s position.
        #   L2 ← the line in the direction of ∇ through D2’s position.

        eta = rotate_vector(drone2.gradient, math.pi / 2 if left else -math.pi / 2)
        position_vector = PositionVector()
        position_vector.movement = PositionVector.SYNCHRONIZE
        [position_vector.x, position_vector.y] = eta

        #   if L1 crosses Path then
        if np.dot(self.drone2.gradient, difference_in_meters(drone2.position, drone1.position)) > 0:
            # Move D2 in its current direction until it is √λ distance away from L1.
            # Change direction to ∇ and take a single step of length λ.
            # Move D1 along L1 until it is √λ away from D2.
            position_vector.position = latlon_plus_meters(drone1.position, unitary(drone2.gradient) * self.distance_sqrt_lambda / 2)

        #   else
        else:
            # Move D1 in its current direction until it is √ λ distance away from L2.
            self.flip = not self.flip
            position_vector.position = latlon_plus_meters(drone2.position, unitary(drone2.gradient) * self.distance_sqrt_lambda / 2)

        position_vector.distance = self.distance_lambda
        self.target_position_vector = position_vector
        yield from self.until(self.target_position_vector, lambda: self.synchronized(self.target_position_vector, drone1, drone2))
        #   Change direction to ∇ and move until the distance from D2 is √λ.

    def forward(self, drone1:Drone, drone2:Drone):
        if self.target_position_vector.movement == PositionVector.TURN:
            prev_vector = [self.target_position_vector.x, self.target_position_vector.y]
            angle =  self.target_position_vector.a * self.target_position_vector.p
            return self.forward_by_vector(drone1, drone2, rotate_vector(prev_vector, angle))
        else:
            return self.forward_by_vector(drone1, drone2, [self.target_position_vector.x, self.target_position_vector.y])

    def forward_by_vector(self, drone1:Drone, drone2:Drone, direction_vector):
        self.clear_crossing()
        gradient = self.calculate_gradient()
        drone1.update_before_crossing()
        drone2.update_before_crossing()

        # D1 and D2 both move λ distance in the direction of ∇
        position_vector = PositionVector()
        position_vector.movement = PositionVector.FORWARD
        [position_vector.x, position_vector.y] = direction_vector

        position_vector.position = average(drone1.position, drone2.position)
        position_vector.gradient = gradient

        position_vector.distance = self.distance_lambda

        return position_vector

    def turn(self, drone1:Drone, drone2:Drone, a):
        if a == self.target_position_vector.a:
            self.target_position_vector.p += 1
            # print(f"p : {self.target_position_vector.p}")
            # if self.latest_point_before_crossing is not None:
            #     self.target_position_vector.vector_to_crossing = unitary(difference_in_meters(self.latest_point_before_crossing, drone1.position))
            #     self.target_position_vector.updated_position = drone1.position
            return self.target_position_vector

        gradient = self.calculate_gradient()

        position_vector = PositionVector()
        position_vector.movement = PositionVector.TURN
        [position_vector.x, position_vector.y] = self.calculate_closest_gradient_tangent([self.target_position_vector.x, self.target_position_vector.y], gradient)
        position_vector.position = average(drone1.position, drone2.position)
        # position_vector.position = self.calculate_position()
        # direction = [self.target_position_vector.x, self.target_position_vector.y]
        # print(f"turn: {direction, [position_vector.x, position_vector.y]}")
        position_vector.a = a
        position_vector.p = 1
        position_vector.gradient = gradient
        #position_vector.prev_pt = self.latest_point_before_crossing

        # if self.latest_point_before_crossing is not None:
        #     position_vector.vector_to_crossing = unitary(difference_in_meters(self.latest_point_before_crossing, drone1.position))
        #     position_vector.updated_position = drone1.position

        return position_vector

    def incomplete(self):
        if self.drone1.position is None or self.drone2.position is None or self.starting_location is None:
            return True
        distance_to_start = magnitude(difference_in_meters(average(self.drone1.position, self.drone2.position), self.starting_location))
        return not (self.encountered and self.armed and distance_to_start < self.distance_sqrt_lambda)

    def inside(self, d:Drone):
        return d.position.value > self.threshold

    def calculate_gradient(self):
        return self.latest_gradient

    def clear_crossing(self):
        self.drone1.clear_crossing()
        self.drone2.clear_crossing()
        self.latest_crossing_point = None

    @staticmethod
    def line_cross(d1):
        return False

    def initial_direction(self, drone1:Drone, drone2:Drone):
        average_position = average(drone1.position, drone2.position)

        position_vector = PositionVector()
        position_vector.movement = PositionVector.FORWARD
        if self.starting_direction is None:
            position_vector.x = 0.0
            position_vector.y = 1.0
        else:
            position_vector.x = self.starting_direction[0]
            position_vector.y = self.starting_direction[1]
        position_vector.position = createLatLon(average_position.latitude, average_position.longitude)
        position_vector.distance = 5.0

        return position_vector

    @staticmethod
    def until(position_vector, callback):
        while not callback():
            yield position_vector

    @staticmethod
    def moved_forward_lambda(target_position_vector, drone1:Drone, drone2:Drone):
        target_offset = difference_in_meters(average(drone1.position, drone2.position), target_position_vector.position)
        distance = np.dot(target_offset, [target_position_vector.x, target_position_vector.y])

        return distance > target_position_vector.distance

    @staticmethod
    def turned_lambda(target_position_vector, distance_lambda, p, drone1:Drone, drone2:Drone):
        center = calculate_turn_center(target_position_vector, distance_lambda)
        target_offset = rotate_vector(difference_in_meters(target_position_vector.position, center), (target_position_vector.a * p))
        hyp = difference_in_meters(average(drone1.position, drone2.position), center)

        target_angle = math.fabs(target_position_vector.a)

        intermediate = ((target_offset[0] * hyp[0]) + (target_offset[1] * hyp[1])) / (magnitude(target_offset) * magnitude(hyp))
        if intermediate > 1 or intermediate < -1:
            intermediate = 1
        angle = math.acos(intermediate)

        # print(f"Turn passed: {angle} > {target_angle} = {angle > target_angle}")
        return angle < target_angle

    def calculate_closest_gradient_tangent(self, direction, gradient):
        orient = self.inside(self.drone1) ^ self.inside(self.drone2)
        if orient == self.INITIAL_ORIENTATION:
            orient = self.inside(self.drone2)

        sign = np.cross(np.array(direction), np.array(gradient)) > 0

        if sign == orient:
            return rotate_vector(gradient, math.pi / 2)
        else:
            return rotate_vector(gradient, -math.pi / 2)


class SketchAction:
    SAMPLE_RATE = 0.1
    SKETCH_STARTING_THRESHOLD = 20 # m

    def __init__(self, id, log_publisher, logger, local_setvelocity_publisher, announce_stream, distance_sqrt_lambda, lambda_value, partner, leader,
                 drone_stream_factory, semaphore_observable, semaphore_publisher, dragonfly_sketch_subject, position_vector_publisher, threshold):
        self.log_publisher = log_publisher
        self.logger = logger
        self.local_setvelocity_publisher = local_setvelocity_publisher
        self.id = id
        self.partner = partner
        self.distance_sqrt_lambda = distance_sqrt_lambda
        self.lambda_value = lambda_value
        self.distance_lambda = self.distance_sqrt_lambda * lambda_value / math.sqrt(lambda_value)
        # print(f"distance_lambda:  {self.distance_sqrt_lambda} * {lambda_value} / {math.sqrt(lambda_value)} = { self.distance_sqrt_lambda * lambda_value / math.sqrt(lambda_value)}")
        # print(f"distance_lambda: {self.distance_lambda}")
        self.leader = leader
        self.started = False
        self.threshold = threshold
        self.ros_subscriptions = []
        self.announce_stream = announce_stream
        self.drone_stream_factory = drone_stream_factory
        self.dragonfly_sketch_subject = dragonfly_sketch_subject
        self.position_vector_publisher = position_vector_publisher
        self.semaphore_observable = semaphore_observable
        self.semaphore_publisher = semaphore_publisher

        self.navigate_subscription = rx.empty().subscribe()
        self.leader_broadcast_subscription = rx.empty().subscribe()
        self.receive_semaphore_subscription = rx.empty().subscribe()

        self.target_position_vector = None
        self.sketch = Sketch(distance_sqrt_lambda, lambda_value, threshold)
        self.sketch_generator = self.sketch.boundary_sketch()

        self.position_reading_queue = []
        self.started = False

        self.status = ActionState.WORKING

    def navigate(self, input):

        twist = TwistStamped()

        for vector in input:
            twist.twist.linear.x += vector[0] / 2
            twist.twist.linear.y += vector[1] / 2

        self.local_setvelocity_publisher.publish(twist)

    @staticmethod
    def format_velocities(twist):
        return [
            twist.twist.linear.x,
            twist.twist.linear.y
        ]

    def step(self):
        if not self.started:
            self.started = True

            self.logger.info("Subscribing...")

            partner_drone = self.drone_stream_factory.get_drone(self.partner)
            self_drone = self.drone_stream_factory.get_drone(self.id)

            self.navigate_subscription = rx.combine_latest(
                partner_drone.get_position(),
                self_drone.get_position(),
                self.dragonfly_sketch_subject
            ).pipe(
                ops.map(lambda positions: self.tandem(positions[0], positions[1], positions[2]))
            ).subscribe(lambda vectors: self.navigate(vectors))

            if self.leader:
                self.leader_broadcast_subscription = rx.combine_latest(
                    self.setup_subject(self.partner),
                    self.setup_subject(self.id)
                ).subscribe(lambda position_reading_vector: self.broadcast_command(position_reading_vector[0], position_reading_vector[1]))
            else:
                self.receive_semaphore_subscription = self.semaphore_observable.subscribe(on_next = lambda token: self.listen_for_stop(token))

        return self.status

    @staticmethod
    def build_sketch_end_id(id):
        return f"sketch.{id}"

    def listen_for_stop(self, token):
        self.logger.info(f"{self.id}: Received token")
        if token.id == self.build_sketch_end_id(self.partner):
            self.end_sketch()

    def broadcast_end(self):
        self.logger.info(f"{self.id}: Publishing sketch stop")
        token = SemaphoreToken()
        token.drone = self.id
        token.id =  self.build_sketch_end_id(self.id)
        self.semaphore_publisher.publish(token)

    def end_sketch(self):
        self.status = ActionState.SUCCESS

    def setup_subject(self, drone):

        drone_streams = self.drone_stream_factory.get_drone(drone)

        position_subject = drone_streams.get_position().pipe(
            ops.timestamp()
        )
        velocity_subject = drone_streams.get_velocity().pipe(
            ops.timestamp()
        )
        co2_subject = drone_streams.get_co2().pipe(
            ops.timestamp()
        )

        def create_reading_position(co2_timestamp, position_timestamp, velocity_timestamp):
            position = position_timestamp[0]
            twist = velocity_timestamp[0]
            velocity = [
                twist.twist.linear.x,
                twist.twist.linear.y
            ]
            co2 = co2_timestamp[0]
            time_delta = co2_timestamp[1] - position_timestamp[1]
            # print(time_delta.total_seconds())
            updated_position = latlon_plus_meters(position, (time_delta.total_seconds() * np.array(velocity)))
            return ReadingPosition(updated_position.latitude, updated_position.longitude, co2.ppm)

        return co2_subject.pipe(
            ops.with_latest_from(position_subject, velocity_subject),
            ops.map(lambda tuple: create_reading_position(tuple[0], tuple[1], tuple[2]))
        )


    def broadcast_command(self, partner_position, self_position):
        self.sketch.update(self_position, partner_position)

        if self.sketch.incomplete():
            self.target_position_vector = next(self.sketch_generator)
        else:
            self.logger.info("End sketch")
            self.broadcast_end()
            self.end_sketch()

        self.log_publisher.publish(String(data=f"PV: {self.target_position_vector}"))

        if self.target_position_vector is not None:
            self.position_vector_publisher.publish(self.target_position_vector)
            self.dragonfly_sketch_subject.on_next(self.target_position_vector)

    def tandem(self, partner_position, self_position, positionVector):

        tandem_offset = self.calculate_tandem_offset(partner_position, self_position)
        tandem_angle = self.calculate_tandem_angle(partner_position, self_position, positionVector)
        straight_line = self.calculate_straight_line(positionVector)
        turning = self.calculate_turn(partner_position, self_position, positionVector)
        offset_error_correction = self.calculate_error_correction(partner_position, self_position, positionVector)

        return [tandem_offset, tandem_angle, straight_line, turning, offset_error_correction]

    def calculate_tandem_offset(self, partner_position, self_position):
        distance_m = difference_in_meters(partner_position, self_position)
        offset_unitary = unitary(distance_m)

        position_offset = np.array(distance_m) - (np.array(offset_unitary) * self.distance_sqrt_lambda)
        offset_magnitude = magnitude(position_offset)

        tandem_distance = position_offset * (2 / max(2, offset_magnitude))
        return tandem_distance

    @staticmethod
    def calculate_tandem_angle(partner_position, self_position, position_vector):
        if position_vector.movement == PositionVector.FORWARD or position_vector.movement == PositionVector.SYNCHRONIZE:
            distance_m = difference_in_meters(self_position, partner_position)
            offset_unitary = unitary(distance_m)

            direction_vector = np.array([position_vector.x, position_vector.y])

            return -2 * (np.dot(offset_unitary, direction_vector)) * direction_vector
        elif position_vector.movement == PositionVector.TURN:
            prev_vector = [position_vector.x, position_vector.y]
            angle =  position_vector.a * position_vector.p
            direction = rotate_vector(prev_vector, angle)

            offset_unitary = unitary(difference_in_meters(self_position, partner_position))

            return -(np.dot(offset_unitary, direction)) * direction
        else:
            return [0, 0]

    @staticmethod
    def calculate_straight_line(position_vector):
        if position_vector.movement == PositionVector.FORWARD:
            return [3 * position_vector.x, 3 * position_vector.y]
        else:
            return [0, 0]

    def calculate_turn(self, partner_position, self_position, position_vector):
        if position_vector.movement == PositionVector.TURN:

            center = calculate_turn_center(position_vector, self.distance_lambda)

            self_center = difference_in_meters(center, self_position)
            self_center_distance = magnitude(self_center)
            partner_center_distance = magnitude(difference_in_meters(center, partner_position))

            base_velocity = 3

            if self_center_distance > partner_center_distance:
                velocity = base_velocity
            else:
                velocity = (self_center_distance / partner_center_distance) * base_velocity

            vector_to_center = unitary(self_center)

            starting_position = difference_in_meters(center, position_vector.position)
            angle = math.atan2(self_center[1], self_center[0]) - math.atan2(starting_position[1], starting_position[0])

            rotated_vector = rotate_vector([position_vector.x, position_vector.y], angle)

            forward_force = np.array(rotated_vector) * velocity
            centripetal_force = np.array(vector_to_center) * 5.5 * velocity / self_center_distance

            return forward_force + centripetal_force
        else:
            return [0, 0]

    def calculate_error_correction(self, partner_position, self_position, position_vector):
        if position_vector.movement == PositionVector.FORWARD:
            vector_to_target =  difference_in_meters(average(self_position, partner_position), position_vector.position)

            dot_product = np.dot([position_vector.x, position_vector.y], vector_to_target)

            vector_to_line = dot_product * np.array([position_vector.x, position_vector.y]) - vector_to_target

            magnitude_calc = magnitude(vector_to_line)

            max_magnitude = 0.1

            if magnitude_calc > max_magnitude:
                return vector_to_line / (magnitude_calc / max_magnitude)
            else:
                return vector_to_line
        elif position_vector.movement == PositionVector.TURN:
            # Maintain distance-to-center
            center = calculate_turn_center(position_vector, self.distance_lambda) # position_vector.position ?
            expected_distance_to_center = magnitude(difference_in_meters(position_vector.position, center))

            average_to_center = difference_in_meters(average(self_position, partner_position), center)

            center_correction = unitary(average_to_center) * np.array(expected_distance_to_center - magnitude(average_to_center))

            return center_correction
        elif position_vector.movement == PositionVector.SYNCHRONIZE:
            vector_to_target =  difference_in_meters(position_vector.position, average(self_position, partner_position))

            magnitude_calc = magnitude(vector_to_target)

            max_magnitude = 1

            if magnitude_calc > max_magnitude:
                return vector_to_target / (magnitude_calc / max_magnitude)
            else:
                return vector_to_target
        else:
            return [0, 0]

    def stop(self):
        self.navigate_subscription.dispose()
        self.leader_broadcast_subscription.dispose()
