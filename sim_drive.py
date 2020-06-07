import numpy as np
import matplotlib.pyplot as plt

car_loc = np.array([-20., 20])
car_dir = np.random.normal(np.arctan2(-car_loc[1], -car_loc[0]), 0.1)
car_error = .05
sample_rate = 10  # per second
car_width = 0.2  # distance from motor_l to motor_r
car_motor_l = 1.  # m/s
car_motor_r = 1.  # m/s
times = 1000  # frame


def your_decision(target_angle):
    global car_motor_l, car_motor_r
    car_motor_l += target_angle * -.01
    car_motor_r += target_angle * .01


# run it
interval = 1. / sample_rate
pi2 = np.pi * 2
save_loc = [car_loc]

for t in range(times):
    # calculate car direction
    c = np.cos(car_dir)
    s = np.sin(car_dir)
    d_l = car_motor_l * interval * abs(np.random.normal(1, car_error))
    d_r = car_motor_r * interval * abs(np.random.normal(1, car_error))

    dth = (d_r - d_l) / car_width 
    # car_dir = (car_dir + dth) % (2 * np.pi) - np.pi
    car_dir = (car_dir + dth) % pi2
    # calculate car location, did not consider rotation radius
    print(car_motor_l, car_motor_r)
    car_loc = car_loc + np.array([[c, -s], [s, c]]).dot([(d_l + d_r) / 2, 0])
    # save it
    save_loc.append(car_loc)

    # arrive?
    if np.all(car_loc < 0.1):
        break

    # you decision
    target_angle = (np.arctan2(-car_loc[1], -car_loc[0]) - car_dir) % pi2
    if target_angle > np.pi:
        target_angle -= pi2
    print(t, car_loc, car_dir * 180 / np.pi, target_angle * 180 / np.pi)
    if target_angle > np.pi / 2 or target_angle < -np.pi / 2:
        break

    your_decision(target_angle)


save_loc = np.array(save_loc)
plt.plot(save_loc[:, 0], save_loc[:, 1], '.-')
plt.plot(np.linspace(save_loc[0, 0], 0), np.linspace(save_loc[0, 1], 0))
plt.show()
