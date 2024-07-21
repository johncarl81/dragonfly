#!/usr/bin/env python3

import argparse
import math
import sys
import numpy as np

import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import NavSatFix
from dragonfly_messages.msg import CO2, LatLon
from dragonfly_messages.srv import SetupPlumes
from std_msgs.msg import String


def createLatLon(latitude, longitude):
    return LatLon(latitude=latitude, longitude=longitude, relative_altitude=0.0)

def unitary(vector):
    vector_magnitude = np.linalg.norm(vector)
    if vector_magnitude == 0 and vector[0] == 0:
        return vector
    return np.array([vector[0] / vector_magnitude, vector[1] / vector_magnitude])

class VirtualCO2Publisher:

    def __init__(self, id, node):
        self.id = id
        self.pub = node.create_publisher(CO2, f"{id}/co2", 10)
        node.create_service(SetupPlumes, f"/{self.id}/virtualco2/setup", self.setup)
        self.node = node
        self.logPublisher = self.node.create_publisher(String, f"{self.id}/log", qos_profile=QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10))
        self.plumes = []

    def differenceInMeters(self, one, two):
        earthCircumference = 40008000
        return [
            ((one.longitude - two.longitude) * (earthCircumference / 360) * math.cos(one.latitude * 0.01745)),
            ((one.latitude - two.latitude) * (earthCircumference / 360))
        ]

    def calculateCO2(self, position):

        value = 0
        h = 2
        for plume in self.plumes:
            [y, x] = self.rotate_vector(self.differenceInMeters(position, plume.source), plume.wind_direction * math.pi / 180)

            if x < 0:
                # Simple gaussian plume model adapted from: https://epubs.siam.org/doi/pdf/10.1137/10080991X
                # See equation 3.10, page 358.
                value += (plume.q / (2 * math.pi * plume.k * -x)) * math.exp(- (plume.u * ((pow(y, 2) + pow(h, 2))) / (4 * plume.k * -x)))

        if value < 0:
            return 420.0
        else:
            return 420.0 + value

    def calculate_co2_xy(self, latitude, longitude):
        return self.calculateCO2(createLatLon(latitude, longitude))

    def calculateGradient(self, position, delta=1e-6):
        lat = position.latitude
        lon = position.longitude

        # Numerical differentiation
        dlat = (self.calculate_co2_xy(lat + delta, lon) - self.calculate_co2_xy(lat - delta, lon)) / (2 * delta)
        dlon = (self.calculate_co2_xy(lat, lon + delta) - self.calculate_co2_xy(lat, lon - delta)) / (2 * delta)

        return unitary([dlon, dlat])

    def rotate_vector(self, vector, angle):
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])
        return np.dot(rotation_matrix, vector)

    def position_callback(self, data):

        if len(self.plumes) > 0:
            ppm = self.calculateCO2(data)

            gradient = self.calculateGradient(data)

            self.pub.publish(CO2(ppm=ppm,
                                 average_temp=55.0,
                                 humidity=0.0,
                                 humidity_sensor_temp=0.0,
                                 atmospheric_pressure=800,
                                 detector_temp=55.0,
                                 source_temp=55.0,
                                 status=CO2.NO_ERROR,
                                 gradient=gradient))

    def publish(self):
        self.node.create_subscription(NavSatFix, f"{self.id}/mavros/global_position/global",
                                      self.position_callback,
                                      qos_profile=QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))

    def setup(self, request, response):
        self.plumes = request.plumes
        self.logPublisher.publish(String(data=f"Setup {len(self.plumes)} plumes."))
        return SetupPlumes.Response(success=True, message=f"Setup plumes on ${self.id}")


def main():
    rclpy.init(args=sys.argv)
    node = rclpy.create_node('virtual_co2')

    parser = argparse.ArgumentParser(description='Starts ROS publisher for CO2 sensor.')
    parser.add_argument('id', type=str, help='Name of the drone.')
    args = parser.parse_args()

    publisher = VirtualCO2Publisher(args.id, node)

    publisher.publish()

    rclpy.spin(node)


if __name__ == '__main__':
    main()
