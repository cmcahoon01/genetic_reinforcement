import random
from time import perf_counter as time
import gym
import tensorflow as tf
from tensorflow import keras
import numpy as np
from random import choices

population_size = 100
generations = 100
survival_rate = 0.2
mutation_rate = 0.1


def create_organism():
    model = keras.Sequential()
    model.add(tf.keras.Input(shape=(4,)))
    # model.add(keras.layers.Dense(units=4, activation='relu'))
    model.add(keras.layers.Dense(units=3, activation='relu'))
    model.add(keras.layers.Dense(units=2))
    return model


def test_population(population, env):
    rewards = []
    for model in population:
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = np.argmax(model.predict(state.reshape(1, 4)))
            state, reward, done, info = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
    return rewards


def breed_survivors(survivors):
    next_gen = []
    for i in range(population_size - len(survivors)):
        parent1 = random.choice(survivors)
        parent2 = random.choice(survivors)
        next_gen.append(breed(parent1, parent2))
    next_gen.extend(survivors)
    return next_gen


def breed(parent1, parent2):
    child = create_organism()
    for i, layer in enumerate(child.layers):
        weights = parent1.layers[i].get_weights()
        for j in range(len(layer.get_weights()[0])):
            for k in range(len(layer.get_weights()[0][j])):
                if random.random() < 0.5:
                    weights[0][j][k] = parent2.layers[i].get_weights()[0][j][k]
        for j in range(len(layer.get_weights()[1])):
            if random.random() < 0.5:
                weights[1][j] = parent2.layers[i].get_weights()[1][j]
        weights = mutate(weights)
        layer.set_weights(weights)
    return child


def mutate(weights):
    for i in range(len(weights[0])):
        for j in range(len(weights[0][i])):
            if random.random() < mutation_rate:
                weights[0][i][j] = np.random.normal(0, 1)
    for i in range(len(weights[1])):
        if random.random() < mutation_rate:
            weights[1][i] = np.random.normal(0, 1)
    return weights


def main():
    env = gym.make('CartPole-v0')
    print("go")

    population = [create_organism() for _ in range(population_size)]
    breed_weights = np.reciprocal(range(1, population_size + 1))

    print('Starting training...')
    for n in range(generations):
        rewards = test_population(population, env)
        sorted_population = np.argsort(rewards)[::-1]
        best_reward = rewards[sorted_population[0]]
        worst_reward = rewards[sorted_population[-1]]
        average_reward = sum(rewards) / population_size
        print(f'Generation {n}: best reward {best_reward}, worst reward: {worst_reward}, average reward {average_reward}')
        survivors = random.choices(population, weights=breed_weights, k=int(population_size * survival_rate))
        population = breed_survivors(survivors)


if __name__ == '__main__':
    main()
