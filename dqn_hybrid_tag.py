import gym
import numpy as np
import tensorflow as tf
import argparse
import itertools
import time
import os
import pickle
import code
import random

from dqn import DQN
from memory import Memory
from make_env import make_env
import general_utilities
import new_alg_utilities


def play(episodes, is_render, is_testing, checkpoint_interval,
         weights_filename_prefix, csv_filename_prefix, batch_size):
    # init statistics. NOTE: simple tag specific!
    statistics_header = ["episode"]
    statistics_header.append("steps")
    statistics_header.extend(["reward_{}".format(i) for i in range(env.n)])
    statistics_header.extend(["loss_{}".format(i) for i in range(env.n)])
    statistics_header.extend(["eps_greedy_{}".format(i) for i in range(env.n)])
    statistics_header.extend(["collisions_{}".format(i) for i in range(env.n)])
    print("Collecting statistics {}:".format(" ".join(statistics_header)))
    statistics = general_utilities.Time_Series_Statistics_Store(
        statistics_header)

    for episode in range(args.episodes):
        states = env.reset()
        episode_losses = np.zeros(env.n)
        episode_rewards = np.zeros(env.n)
        collision_count = np.zeros(env.n)
        steps = 0

        while True:
            steps += 1

            # render
            if args.render:
                env.render()

            # act
            actions = np.zeros(env.n)
            actions_onehot = []
            #the action is selected by the JAL deep Q network
            action = dqns[0].choose_action(states[0])
            speed = 0.9 
            
            #distribute actions to two players
            if action == 0:
                actions[0]=0
                actions[1]=0
            elif action == 1:
                actions[0]=0
                actions[1]=1
            elif action == 2:
                actions[0]=0
                actions[1]=2
            elif action == 3:
                actions[0]=0
                actions[1]=3
            elif action == 4:
                actions[0]=0
                actions[1]=4
                
            elif action == 5:
                actions[0]=1
                actions[1]=0
            elif action == 6:
                actions[0]=1
                actions[1]=1
            elif action == 7:
                actions[0]=1
                actions[1]=2
            elif action == 8:
                actions[0]=1
                actions[1]=3
            elif action == 9:
                actions[0]=1
                actions[1]=4
                
            elif action == 10:
                actions[0]=2
                actions[1]=0
            elif action == 11:
                actions[0]=2
                actions[1]=1
            elif action == 12:
                actions[0]=2
                actions[1]=2
            elif action == 13:
                actions[0]=2
                actions[1]=3
            elif action == 14:
                actions[0]=2
                actions[1]=4
                
            elif action == 15:
                actions[0]=3
                actions[1]=0
            elif action == 16:
                actions[0]=3
                actions[1]=1
            elif action == 17:
                actions[0]=3
                actions[1]=2
            elif action == 18:
                actions[0]=3
                actions[1]=3
            elif action == 19:
                actions[0]=3
                actions[1]=4
                
            elif action == 20:
                actions[0]=4
                actions[1]=0
            elif action == 21:
                actions[0]=4
                actions[1]=1
            elif action == 22:
                actions[0]=4
                actions[1]=2
            elif action == 23:
                actions[0]=4
                actions[1]=3
            elif action == 24:
                actions[0]=4
                actions[1]=4
                
            actions=actions.astype(int)
            for i in range(env.n):
                onehot_action = np.zeros(n_actions)
                onehot_action[actions[i]] = speed
                actions_onehot.append(onehot_action)

            # step
            states_next, rewards, done, info = env.step(actions_onehot)
            
            reward_cal=rewards[0]+rewards[1];

            # learn
            if not args.testing:
                size = memories.pointer
                batch = random.sample(range(size), size) if size < batch_size else random.sample(
                    range(size), batch_size)
                
                done_cal=np.logical_and(done[0],done[1])
                memories.remember(states[0], action,
                                     reward_cal, states_next[0], done_cal)
                memories2.remember(states[0], actions[0],
                                     reward_cal, states_next[0], done_cal)

                if memories.pointer > batch_size * 10:
                    history = dqns[0].learn(*memories.sample(batch))
                    dqns[1].learn (*memories2.sample(batch))
                    episode_losses[0] += history.history["loss"][0]
                else:
                    episode_losses[0] = -1
                    

            states = states_next
            episode_rewards += rewards
            collision_count += np.array(
                new_alg_utilities.count_agent_collisions(env))

            # reset states if done
            if any(done):
                episode_rewards = episode_rewards / steps
                episode_losses = episode_losses / steps

                statistic = [episode]
                statistic.append(steps)
                statistic.extend([episode_rewards[i] for i in range(env.n)])
                statistic.extend([episode_losses[i] for i in range(env.n)])
                statistic.extend([dqns[0].eps_greedy for i in range(env.n)])
                statistic.extend(collision_count.tolist())
                statistics.add_statistics(statistic)
                if episode % 25 == 0:
                    print(statistics.summarize_last())
                break

        if episode % checkpoint_interval == 0:
            statistics.dump("{}_{}.csv".format(csv_filename_prefix,
                                               episode))
            #general_utilities.save_dqn_weights(dqn,
           #                                    "{}_{}_".format(weights_filename_prefix, episode))
            #if episode >= checkpoint_interval:
              #  os.remove("{}_{}.csv".format(csv_filename_prefix,
              #                               episode - checkpoint_interval))

    return statistics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='algo_design', type=str)
    #parser.add_argument('--env', default='simple', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--episodes', default=2000, type=int)
    parser.add_argument('--render', default=True, action="store_true")
    parser.add_argument('--benchmark', default=False, action="store_true")
    parser.add_argument('--experiment_prefix', default=".",
                        help="directory to store all experiment data")
    parser.add_argument('--weights_filename_prefix', default='/save/tag-dqn',
                        help="where to store/load network weights")
    parser.add_argument('--csv_filename_prefix', default='/save/statistics-dqn',
                        help="where to store statistics")
    parser.add_argument('--checkpoint_frequency', default=500,
                        help="how often to checkpoint")
    parser.add_argument('--testing', default=False, action="store_true",
                        help="reduces exploration substantially")
    parser.add_argument('--random_seed', default=2, type=int)
    parser.add_argument('--memory_size', default=10000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epsilon_greedy', nargs='+', type=float,
                        help="Epsilon greedy parameter for each agent")
    args = parser.parse_args()

    general_utilities.dump_dict_as_json(vars(args),
                                        args.experiment_prefix + "/save/run_parameters.json")
    # init env
    env = make_env(args.env, args.benchmark)
    

    if args.epsilon_greedy is not None:
        if len(args.epsilon_greedy) == env.n:
            epsilon_greedy = args.epsilon_greedy
        else:
            raise ValueError("Must have enough epsilon_greedy for all agents")
    else:
        epsilon_greedy = [0.5 for i in range(env.n)]

    # set random seed
    env.seed(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)

    # init DQN for JAL learner
    n_actions = env.action_space[0].n
    state_sizes = env.observation_space[0].shape[0]
    memories = Memory(args.memory_size)
    memories2 = Memory(args.memory_size)
    
    dqns=[]
    #to implement JAL, the action space becomes a1b1, a1b2, a1b3, ..., a5b4, a5b5
    dqns.append( DQN(n_actions*n_actions, state_sizes, eps_greedy=epsilon_greedy[0]))
    #DQN for independent learner
    dqns.append( DQN(n_actions, state_sizes, eps_greedy=epsilon_greedy[0]))
    
    #general_utilities.load_dqn_weights_if_exist(
    #    dqns, args.experiment_prefix + args.weights_filename_prefix)
    
    

    start_time = time.time()

    # play
    statistics = play(args.episodes, args.render, args.testing,
                      args.checkpoint_frequency,
                      args.experiment_prefix + args.weights_filename_prefix,
                      args.experiment_prefix + args.csv_filename_prefix,
                      args.batch_size)

    # bookkeeping
    print("Finished {} episodes in {} seconds".format(args.episodes,
                                                      time.time() - start_time))
    general_utilities.save_dqn_weights(
        dqns, args.experiment_prefix + args.weights_filename_prefix)
    statistics.dump(args.experiment_prefix + args.csv_filename_prefix + ".csv")
