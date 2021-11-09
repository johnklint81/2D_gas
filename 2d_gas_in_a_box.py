import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

N = 100
sigma = 1
epsilon = 1
mass = 1
L = 100 * sigma
number_of_timesteps = 10000
v0 = np.sqrt(2 * epsilon / mass)
t0 = sigma / v0
timestep = t0 * 0.001
t_plot = np.linspace(0, number_of_timesteps * timestep, number_of_timesteps - 1)

position_list = np.zeros([N, 2])
velocity_list = np.zeros([N, 2])
potential_energy_list = np.zeros(N)
kinetic_energy_list = np.zeros(N)
total_energy_list = np.zeros(N)
potential_energy_list_t = np.zeros(number_of_timesteps - 1)
kinetic_energy_list_t = np.zeros(number_of_timesteps - 1)
total_energy_list_t = np.zeros(number_of_timesteps - 1)


# Initialize and check so no particle is too close
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


# Get distance between particles
def get_distances(_position_candidate, _position_list):
    _distances = np.linalg.norm(_position_candidate - _position_list, axis=1)
    return _distances


# Get forces for all particle and return the sum of all forces
def get_forces(_position_list, _sigma, _epsilon, _N):
    _forces = np.zeros([N, 2])
    for i in range(_N):
        _distances = get_distances(_position_list[i], _position_list)
        _directions = _position_list[i] - _position_list     # get direction vectors between all particles
        _index = np.where(_distances == 0)[0][0]    # find the entry for the direction to the particle itself
        _directions = np.delete(_directions, _index, axis=0)    # delete it
        _distances = np.delete(_distances, _distances == 0)     # delete as well
        _directions[:, 0] /= _distances     # normalize direction vectors to unity
        _directions[:, 1] /= _distances     # same
        _magnitude = 4 * epsilon * (12 * (sigma ** 12 / _distances ** 13)
                                    - 6 * (sigma ** 6 / _distances ** 7))   # calculate the magnitude of the forces
        _forces[i, 0] = np.sum(_directions[:, 0] * _magnitude)      # combine direction vectors with magnitude
        _forces[i, 1] = np.sum(_directions[:, 1] * _magnitude)      # same
    return _forces


# Check if any particle is outside the box
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


def get_energy(_distances, _velocity, _epsilon, _sigma, _mass):
    _potential_energy = 0.5 * np.sum(4 * _epsilon * ((_sigma / _distances) ** 12 - (_sigma / _distances) ** 6))
    _kinetic_energy = 0.5 * _mass * np.linalg.norm(_velocity) ** 2
    _total_energy = _potential_energy + _kinetic_energy
    return _potential_energy, _kinetic_energy, _total_energy


# Advance algorithm one step for all particles
def step(_position_list, _velocity_list, _potential_energy_list, _kinetic_energy_list, _total_energy_list, _sigma,
         _epsilon, _N, _L, _mass, _timestep):
    # Leapfrog algorithm
    _position_list += _velocity_list * _timestep / 2
    _position_list, _velocity_list = outside_box(_position_list, _velocity_list, _L, _N)

    _velocity_list += get_forces(_position_list, _sigma, _epsilon, _N) / _mass * _timestep

    _position_list += _velocity_list * _timestep / 2
    _position_list, _velocity_list = outside_box(_position_list, _velocity_list, _L, _N)

    for i in range(_N):
        # Get energy at each timestep
        _distances = get_distances(_position_list[i], _position_list)
        _distances = np.delete(_distances, _distances == 0)
        _potential_energy_list[i], _kinetic_energy_list[i], _total_energy_list[i] = get_energy(_distances,
                                                                                               _velocity_list[i],
                                                                                               _epsilon,
                                                                                               _sigma, _mass)
    _potential_energy = np.sum(_potential_energy_list)
    _kinetic_energy = np.sum(_kinetic_energy_list)
    _total_energy = _potential_energy + _kinetic_energy
    return _position_list, _velocity_list, _potential_energy, _kinetic_energy, _total_energy


# For creating animation
def update(i):
    data = position_array[:, :, i]
    position_plot.set_data(data[:, 0], data[:, 1])
    return position_plot,


fig = plt.figure()
ax = plt.axes(xlim=(0, L), ylim=(0, L))
ax.set_xlabel("x")
ax.set_ylabel("y")
position_plot, = ax.plot([], [], 'k.', markersize=10)

position_array = np.zeros([N, 2, number_of_timesteps])
velocity_array = np.zeros([N, 2, number_of_timesteps])
position_array[:, :, 0], velocity_array[:, :, 0] = initialize_particles(position_list, velocity_list, sigma, v0, L, N)

# Evolve in time
for i in range(number_of_timesteps - 1):
    position_array[:, :, i + 1], velocity_array[:, :, i + 1], potential_energy_list_t[i], kinetic_energy_list_t[i], \
    total_energy_list_t[i] = step(position_array[:, :, i], velocity_array[:, :, i], potential_energy_list,
                                  kinetic_energy_list, total_energy_list, sigma, epsilon, N, L, mass, timestep)

animation = FuncAnimation(fig, update, frames=number_of_timesteps, repeat=False, blit=True)

writervideo = matplotlib.animation.FFMpegWriter(fps=2000)
animation.save('animated_gas.mp4', writer=writervideo)

fig2, ax = plt.subplots(3, 1)

ax[2].set_xlabel("t [t$_0$]")
ax[0].set_ylabel("K")
ax[1].set_ylabel("U")
ax[0].set_xticks([])
ax[1].set_xticks([])
ax[2].set_ylabel("E")
ax[0].plot(t_plot, kinetic_energy_list_t, 'r', alpha=0.7)
ax[1].plot(t_plot, potential_energy_list_t, 'b', alpha=0.7)
ax[2].plot(t_plot, total_energy_list_t, 'k--', alpha=0.7)

plt.show()
