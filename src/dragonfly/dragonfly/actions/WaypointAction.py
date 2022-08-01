#!/usr/bin/env python3
import math

import rx
import rx.operators as ops

from rx.subject import Subject

from .ActionState import ActionState


def distance(position1, position2):
    deltax = position1.x - position2.x
    deltay = position1.y - position2.y
    deltaz = position1.z - position2.z

    return math.sqrt((deltax * deltax) + (deltay * deltay) + (deltaz * deltaz))


class WaypointAction:
    STOP_VELOCITY_THRESHOLD = 0.1
    SAMPLE_RATE = 1.0
    WAIT_FOR_WAYPOINT = 10
    WAYPOINT_ACCEPTANCE_ADJUSTMENT = {'x': 0, 'y': 0, 'z': 0}

    def __init__(self, id, logPublisher, local_setposition_publisher, waypoint, distance_threshold, local_velocity_observable, local_pose_observable):
        self.id = id
        self.logPublisher = logPublisher
        self.waypoint = waypoint
        self.distance_threshold = distance_threshold
        self.local_setposition_publisher = local_setposition_publisher
        self.status = ActionState.WORKING
        self.commanded = False
        self.position_update = None
        self.velocity_update = None
        self.velocity_subject = Subject()
        self.pose_subject = Subject()
        self.waypointAcceptanceSubscription = rx.empty().subscribe()
        self.local_velocity_observable = local_velocity_observable
        self.local_pose_observable = local_pose_observable

    def step(self):
        if not self.commanded:
            self.commanded = True

            def updatePosition(pose, velocity, time):
                # print("Distance to point:{} {} {}".format(self.waypoint.pose.position.x, self.waypoint.pose.position.y, self.waypoint.pose.position.z), \
                #       distance(self.waypoint.pose.position, localposition.pose.position))
                alteredposition = pose

                alteredposition.x += WaypointAction.WAYPOINT_ACCEPTANCE_ADJUSTMENT['x']
                alteredposition.y += WaypointAction.WAYPOINT_ACCEPTANCE_ADJUSTMENT['y']
                alteredposition.z += WaypointAction.WAYPOINT_ACCEPTANCE_ADJUSTMENT['z']

                magnitude = math.sqrt((velocity.x * velocity.x) + (velocity.y * velocity.y) + (velocity.z * velocity.z))
                # print("{} - {} @ {}".format(magnitude, distance(self.waypoint.pose.position, alteredposition), time))

                if distance(self.waypoint.pose.position, alteredposition) < self.distance_threshold:
                    self.status = ActionState.SUCCESS

                    self.stop()
                elif time > self.WAIT_FOR_WAYPOINT and magnitude < self.STOP_VELOCITY_THRESHOLD:
                    WaypointAction.WAYPOINT_ACCEPTANCE_ADJUSTMENT['x'] += self.waypoint.pose.position.x - pose.x
                    WaypointAction.WAYPOINT_ACCEPTANCE_ADJUSTMENT['y'] += self.waypoint.pose.position.y - pose.y
                    WaypointAction.WAYPOINT_ACCEPTANCE_ADJUSTMENT['z'] += self.waypoint.pose.position.z - pose.z
                    self.logPublisher.publish(
                        String(data="Adjusted waypoint acceptance: {}".format(WaypointAction.WAYPOINT_ACCEPTANCE_ADJUSTMENT)))

            self.waypointAcceptanceSubscription = rx.combine_latest(
                self.local_pose_observable, self.local_velocity_observable, rx.interval(1)).pipe(
                ops.sample(self.SAMPLE_RATE)
            ).subscribe(on_next=lambda values: updatePosition(values[0].pose.position, values[1].twist.linear, values[2]))

            self.local_setposition_publisher.publish(self.waypoint)

        return self.status

    def stop(self):
        self.waypointAcceptanceSubscription.dispose()