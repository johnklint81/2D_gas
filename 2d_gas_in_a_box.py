import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

N = 100
sigma = 0.01
epsilon = 0.001
mass = 0.1
timestep = 0.000001
L = 100 * sigma
T = 10000
v0 = 10 * np.sqrt(2 * epsilon / mass)
t0 = sigma * np.sqrt(mass / (2 * epsilon))

position_list = np.zeros([N, 2])
velocity_list = np.zeros([N, 2])


def initialize_particles(_position_list, _velocity_list, _sigma, _v0, _L, _N):
    _position_list = np.expand_dims(np.random.rand(2) * _L, axis=0)
    _angle = np.random.rand(_N) * 2 * np.pi
    _velocity_list[:, :] = 2 * _v0 * np.reshape((np.cos(_angle), np.sin(_angle)), (_N, 2))

    for i in range(_N - 1):
        placing = True
        while placing:
            _position_candidate = np.expand_dims(np.random.rand(2) * _L, axis=0)
            if np.all(get_distances(_position_candidate, _position_list) > sigma):

                _position_list = np.append(_position_list, _position_candidate, axis=0)
                placing = False
            else:
                print("Too close! Placing a new particle!")
    return _position_list, _velocity_list


def get_distances(_position_candidate, _position_list):
    _distances = np.linalg.norm(_position_candidate - _position_list, axis=1)
    return _distances


def get_forces(_position_list, _sigma, _epsilon, _N):
    _forces = np.zeros([N, 2])
    for i in range(_N):
        _distances = get_distances(_position_list[i], _position_list)
        _directions = _position_list - _position_list[i]
        index = np.where(_distances == 0)[0][0]
        _directions = np.delete(_directions, index, axis=0)
        _distances = np.delete(_distances, _distances == 0)
        _directions[:, 0] /= _distances
        _directions[:, 1] /= _distances
        _direction = np.sum(_directions, axis=0) / (_N - 1)
        # print(_direction)
        _magnitude = np.sum(4 * epsilon * (12 * (sigma ** 12 / _distances ** 13)
                                           - 6 * (sigma ** 6 / _distances ** 7)), axis=0)
        _forces[i, :] = _direction * _magnitude
    return _forces


def outside_box(_position_list, _velocity_list, _L, _N):
    for i in range(_N):
        if _position_list[i, 0] > _L:
            _position_list[i, 0] = 2 * _L - _position_list[i, 0]
            _velocity_list[i, 0] *= -1
        elif _position_list[i, 0] < 0:
            _position_list[i, 0] *= -1
            _velocity_list[i, 0] *= -1

        if _position_list[i, 1] > _L:
            _position_list[i, 1] = 2 * _L - _position_list[i, 1]
            _velocity_list[i, 1] *= -1
        elif _position_list[i, 1] < 0:
            _position_list[i, 1] *= -1
            _velocity_list[i, 1] *= -1
    return _position_list, _velocity_list


def step(_position_list, _velocity_list, _sigma, _epsilon, _N, _L, _m, _timestep):
    _position_list += _velocity_list * _timestep / 2
    _position_list, _velocity_list = outside_box(_position_list, _velocity_list, _L, _N)

    _velocity_list += get_forces(_position_list, _sigma, _epsilon, _N) / _m * _timestep

    _position_list += _velocity_list * _timestep / 2
    _position_list, _velocity_list = outside_box(_position_list, _velocity_list, _L, _N)
    return _position_list, _velocity_list


def update(i):
    data = position_array[:, :, i]
    position_plot.set_data(data[:, 0], data[:, 1])
    return position_plot,


fig = plt.figure()
ax = plt.axes(xlim=(0, L), ylim=(0, L))
ax.set_xlabel("x")
ax.set_ylabel("y")
position_plot, = ax.plot([], [], 'k.')

position_array = np.zeros([N, 2, T])
velocity_array = np.zeros([N, 2, T])
position_array[:, :, 0], velocity_array[:, :, 0] = initialize_particles(position_list, velocity_list, sigma, v0, L, N)

for i in range(T - 1):
    position_array[:, :, i + 1], velocity_array[:, :, i + 1] = step(position_array[:, :, i], velocity_array[:, :, i],
                                                                    sigma, epsilon, N, L, mass, timestep)
animation = FuncAnimation(fig, update, interval=, blit=True)

writervideo = matplotlib.animation.FFMpegWriter(fps=1000)
animation.save('animated_gas.mp4', writer=writervideo)

plt.show()
