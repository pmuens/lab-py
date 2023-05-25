import numpy as np


def egg_crate(x, y):
    return x**2 + y**2 + 25 * (np.sin(x) ** 2 + np.sin(y) ** 2)


def bat(population_size=10, loudness=0.25, pulse_rate=0.5):
    frequency_min = 0  # Lowest frequency.
    frequency_max = 2  # Highest frequency.

    iterations = 0
    stop_tolerance = 10 ** (-5)

    dimensions = 2

    frequencies = np.zeros(population_size)  # Frequency of each bat.
    velocities = np.zeros(
        (population_size, dimensions)
    )  # Speed of each bat in each dimension.

    temp_positions = np.zeros((population_size, dimensions))

    positions = np.random.randn(population_size, dimensions)
    fitness = [egg_crate(x, y) for x, y in positions]

    best = positions[np.argmin(fitness)]
    fitness_min = fitness[np.argmin(fitness)]

    while fitness_min > stop_tolerance:
        for i in range(population_size):
            beta = np.random.rand()
            frequencies[i] = frequency_min + (frequency_max - frequency_min) * beta
            velocities[i, :] = (
                velocities[i, :] + (positions[i, :] - best) * frequencies[i]
            )
            temp_positions[i, :] = positions[i, :] + velocities[i, :]

            alpha = 0.01
            if np.random.rand() > pulse_rate:
                temp_positions[i, :] = best + alpha * np.random.randn(1, dimensions)

            x = temp_positions[i, 0]
            y = temp_positions[i, 1]
            fitness_new = egg_crate(x, y)

            if fitness_new <= fitness[i] and np.random.rand() < loudness:
                positions[i, :] = temp_positions[i, :]
                fitness[i] = fitness_new

                if fitness_new < fitness_min:
                    best = temp_positions[i, :]
                    fitness_min = fitness_new

        iterations += population_size

    return best, fitness_min, positions, iterations


if __name__ == "__main__":
    best, fitness_min, positions, iterations = bat()
    print("Number of iterations:", iterations)
    print("Best Position:", best)
    print("Lowest test function:", fitness_min)
