#!/usr/bin/env python3
import math
import sys

import rclpy
from rclpy.qos import QoSProfile
from rx import Observable
from rx.subject import Subject
from sensor_msgs.msg import NavSatFix

SAMPLE_RATE = 10
node = None


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


VIRTUAL_SOURCE = dotdict({
    "latitude": 35.19465,
    "longitude": -106.59625
})


def differenceInMeters(one, two):
    earthCircumference = 40008000
    return [
        ((one.longitude - two.longitude) * (earthCircumference / 360) * math.cos(one.latitude * 0.01745)),
        ((one.latitude - two.latitude) * (earthCircumference / 360))
    ]


def calculateCO2(position):
    [y, x] = differenceInMeters(position, VIRTUAL_SOURCE)

    if x == 0:
        x = 0.00001

    Q = 5000
    K = 2
    H = 2
    u = 1

    value = (Q / (2 * math.pi * K * -x)) * math.exp(- (u * ((pow(y, 2) + pow(H, 2))) / (4 * K * -x)))

    if value < 0:
        return 420
    else:
        return 420 + value


def log_vectors(vectors):
    for vector in vectors:
        print("{},{},{},{}".format(vector.longitude, vector.latitude, vector.altitude, calculateCO2(vector)), end=''),
    print()


def main():
    global node

    df1_subject = Subject()
    df2_subject = Subject()
    df3_subject = Subject()

    node.create_subscription(NavSatFix, "/dragonfly1/mavros/global_position/global",
                             lambda position: df1_subject.on_next(position), qos_profile=QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10))
    node.create_subscription(NavSatFix, "/dragonfly2/mavros/global_position/global",
                             lambda position: df2_subject.on_next(position), qos_profile=QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10))
    node.create_subscription(NavSatFix, "/dragonfly3/mavros/global_position/global",
                             lambda position: df3_subject.on_next(position), qos_profile=QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10))
    rclpy.spin(node)

    print(
        "df1.lon, df1.lat, df1.alt, df1.value, df2.lon, df2.lat, df2.alt, df2.value, df3.lon, df3.lat, df3.alt, "
        "df3.value")
    rx.combine_latest([df1_subject, df2_subject, df3_subject], lambda *positions: positions) \
        .sample(SAMPLE_RATE) \
        .subscribe(lambda vectors: log_vectors(vectors))


if __name__ == '__main__':
    rclpy.init(args=sys.argv)
    node = rclpy.create_node("position_logger")
    main()