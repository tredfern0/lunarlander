from ll_pytorch import *
import json

def train_agent1(episodes=1000, alpha=.0005, minibatch_size=64, epsilon_decay=.99, final_epsilon=.1, update_target_network_frequency=1, max_steps=900, keep_float_episodes_pct=.2, focused_train_loops=1, show=False):
    """
    update_target_network_frequency: how often we update the cache of our network
    """
    env = gym.make('LunarLander-v2')
    env.seed(0)

    replay_buffer = []
    episode_rewards = []
    trailing_avg = []
    num_actions = 4
    landings = 0

    net = Net(alpha)
    if update_target_network_frequency:
        net_static = copy.deepcopy(net)
    else:
        net_static = net

    epsilon = 1
    best_avg = -999
    for j in range(episodes):
        epsilon*=epsilon_decay
        epsilon = max(final_epsilon, epsilon)

        #We receive an array on reset
        x = env.reset()
        #x = mod_x(x)

        episode_reward = 0
        #episode_buffer = []
        i = 0

        while True:
            #Tracking step count - ending a bit early, at 900 steps
            i+=1

            #Take action A, want to observe R, S'
            A = choose_action(x, net, epsilon, num_actions, False)
            result = env.step(A)

            if show and j%50==0:
                env.render()

            episode_reward += result[1]

            terminal_mult = 0 if (result[2] and not result[3]) else 1
            #SARS,Terminal - new state is [0] element in result, reward [1]
            experience_tuple = (x, A, result[1], result[0], terminal_mult)
            #MOVING CODE DIRECTLY TO REPLAY BUFFER NOW
            replay_buffer.append(experience_tuple)

            if result[2]:
                x, reward, terminal, null = result
                #x = mod_x(x)
                if reward==100:
                    landings += 1
                    print("########################")
                    print("SUCCESSFUL LANDING!")
                    print("TERMINAL RESULT", result)
                    print("CURRENT LEN SAMPLE BUCKET", len(replay_buffer_buckets[0]), "AND Y VAL", x[1])
                    print("########################")

            if len(replay_buffer)>minibatch_size:
                #samples = choose_samples_buffer(replay_buffer, minibatch_size)
                #EXPERIMENT 1 - always train on most recent samples
                samples = replay_buffer[-minibatch_size:]
                run_update(samples, net, net_static, num_updates=minibatch_size)

            x, reward, terminal, null = result
            #x = mod_x(x)

            if terminal:
                break

            #if i>max_steps:
            #    accept = np.random.random()
            #    #Only save some% of these episodes in our buffer
            #    if accept>keep_float_episodes_pct:
            #        floated = True
            #    break

        #Update target weights every other episode
        if update_target_network_frequency and i%update_target_network_frequency==0:
            net_static = copy.deepcopy(net)

        #Extend buffer for all ticks in current episode
        #if not floated:
        #    replay_buffer.extend(episode_buffer)
        #replay_buffer = replay_buffer[-50000:]

        avg = sum(episode_rewards[-100:])/100
        best_avg = max(best_avg, avg)
        print("EP", j, "STEPS", i, "EPSILON", epsilon, "REWARD", episode_reward, "AVG",avg)
        trailing_avg.append(avg)
        #Just for experiment7
        #if avg>200:
            #break
        #if avg>200:
            #torch.save(net, "./mod{}.torch".format(j))
        episode_rewards.append(episode_reward)
    print("LANDINGS", landings, "BEST AVG", best_avg)
    env.close()
    return net, episode_rewards

def train_agent2(episodes=1000, alpha=.0005, minibatch_size=64, epsilon_decay=.99, final_epsilon=.1, update_target_network_frequency=1, max_steps=900, keep_float_episodes_pct=.2, focused_train_loops=1, show=False):
    """
    update_target_network_frequency: how often we update the cache of our network
    """
    env = gym.make('LunarLander-v2')
    env.seed(0)

    replay_buffer = []
    episode_rewards = []
    trailing_avg = []
    num_actions = 4
    landings = 0

    net = Net(alpha)
    if update_target_network_frequency:
        net_static = copy.deepcopy(net)
    else:
        net_static = net

    epsilon = 1
    best_avg = -999
    for j in range(episodes):
        epsilon*=epsilon_decay
        epsilon = max(final_epsilon, epsilon)

        #We receive an array on reset
        x = env.reset()
        #x = mod_x(x)

        episode_reward = 0
        episode_buffer = []
        i = 0

        while True:
            #Tracking step count - ending a bit early, at 900 steps
            i+=1

            #Take action A, want to observe R, S'
            A = choose_action(x, net, epsilon, num_actions, False)
            result = env.step(A)

            if show and j%50==0:
                env.render()

            episode_reward += result[1]

            terminal_mult = 0 if (result[2] and not result[3]) else 1
            #SARS,Terminal - new state is [0] element in result, reward [1]
            experience_tuple = (x, A, result[1], result[0], terminal_mult)
            #MOVING CODE DIRECTLY TO REPLAY BUFFER NOW
            episode_buffer.append(experience_tuple)

            if result[2]:
                x, reward, terminal, null = result
                #x = mod_x(x)
                if reward==100:
                    landings += 1
                    print("########################")
                    print("SUCCESSFUL LANDING!")
                    print("TERMINAL RESULT", result)
                    print("CURRENT LEN SAMPLE BUCKET", len(replay_buffer_buckets[0]), "AND Y VAL", x[1])
                    print("########################")

            if len(replay_buffer)>minibatch_size:
                #EXPERIMENT 2 - will only have the previous episode
                samples = choose_samples_buffer(replay_buffer, minibatch_size)
                run_update(samples, net, net_static, num_updates=minibatch_size)

            x, reward, terminal, null = result
            #x = mod_x(x)

            if terminal:
                break

            #if i>max_steps:
            #    accept = np.random.random()
            #    #Only save some% of these episodes in our buffer
            #    if accept>keep_float_episodes_pct:
            #        floated = True
            #    break

        #Update target weights every other episode
        if update_target_network_frequency and i%update_target_network_frequency==0:
            net_static = copy.deepcopy(net)

        #Extend buffer for all ticks in current episode
        #Experiment 2 specific!  Overwriting
        replay_buffer = episode_buffer
        #replay_buffer = replay_buffer[-50000:]

        avg = sum(episode_rewards[-100:])/100
        best_avg = max(best_avg, avg)
        print("EP", j, "STEPS", i, "EPSILON", epsilon, "REWARD", episode_reward, "AVG",avg)
        trailing_avg.append(avg)
        #Just for experiment7
        #if avg>200:
            #break
        #if avg>200:
            #torch.save(net, "./mod{}.torch".format(j))
        episode_rewards.append(episode_reward)
    print("LANDINGS", landings, "BEST AVG", best_avg)
    env.close()
    return net, episode_rewards


def train_agent3(episodes=1000, alpha=.0005, minibatch_size=64, epsilon_decay=.99, final_epsilon=.1, update_target_network_frequency=1, max_steps=900, keep_float_episodes_pct=.2, focused_train_loops=1, show=False, replay_limit=2000):
    """
    update_target_network_frequency: how often we update the cache of our network
    """
    env = gym.make('LunarLander-v2')
    env.seed(0)

    replay_buffer = []
    episode_rewards = []
    trailing_avg = []
    num_actions = 4
    landings = 0

    net = Net(alpha)
    if update_target_network_frequency:
        net_static = copy.deepcopy(net)
    else:
        net_static = net

    epsilon = 1
    best_avg = -999
    for j in range(episodes):
        epsilon*=epsilon_decay
        epsilon = max(final_epsilon, epsilon)

        #We receive an array on reset
        x = env.reset()
        #x = mod_x(x)

        episode_reward = 0
        #episode_buffer = []
        i = 0

        while True:
            #Tracking step count - ending a bit early, at 900 steps
            i+=1

            #Take action A, want to observe R, S'
            A = choose_action(x, net, epsilon, num_actions, False)
            result = env.step(A)

            if show and j%50==0:
                env.render()

            episode_reward += result[1]

            terminal_mult = 0 if (result[2] and not result[3]) else 1
            #SARS,Terminal - new state is [0] element in result, reward [1]
            experience_tuple = (x, A, result[1], result[0], terminal_mult)
            #MOVING CODE DIRECTLY TO REPLAY BUFFER NOW
            replay_buffer.append(experience_tuple)

            if result[2]:
                x, reward, terminal, null = result
                #x = mod_x(x)
                if reward==100:
                    landings += 1
                    print("########################")
                    print("SUCCESSFUL LANDING!")
                    print("TERMINAL RESULT", result)
                    print("CURRENT LEN SAMPLE BUCKET", len(replay_buffer_buckets[0]), "AND Y VAL", x[1])
                    print("########################")

            if len(replay_buffer)>minibatch_size:
                #EXPERIMENT 2 - will only have the previous episode
                samples = choose_samples_buffer(replay_buffer, minibatch_size)
                run_update(samples, net, net_static, num_updates=minibatch_size)

            x, reward, terminal, null = result
            #x = mod_x(x)

            if terminal:
                break

            #if i>max_steps:
            #    accept = np.random.random()
            #    #Only save some% of these episodes in our buffer
            #    if accept>keep_float_episodes_pct:
            #        floated = True
            #    break

        #Update target weights every other episode
        if update_target_network_frequency and i%update_target_network_frequency==0:
            net_static = copy.deepcopy(net)

        #Extend buffer for all ticks in current episode
        #Experiment 2 specific!  Overwriting
        replay_buffer = replay_buffer[-replay_limit:]

        avg = sum(episode_rewards[-100:])/100
        best_avg = max(best_avg, avg)
        print("EP", j, "STEPS", i, "EPSILON", epsilon, "REWARD", episode_reward, "AVG",avg)
        trailing_avg.append(avg)
        #Just for experiment7
        #if avg>200:
            #break
        #if avg>200:
            #torch.save(net, "./mod{}.torch".format(j))
        episode_rewards.append(episode_reward)
    print("LANDINGS", landings, "BEST AVG", best_avg)
    env.close()
    return net, episode_rewards



_, episode_rewards1 = train_agent1(episodes=1000, alpha=.0005, minibatch_size=64, epsilon_decay=.99, final_epsilon=.1, update_target_network_frequency=1, max_steps=900, keep_float_episodes_pct=.2, focused_train_loops=1, show=False)
with open("replay1.json","w") as f:
    f.write(json.dumps(episode_rewards1))
_, episode_rewards2 = train_agent2(episodes=1000, alpha=.0005, minibatch_size=64, epsilon_decay=.99, final_epsilon=.1, update_target_network_frequency=1, max_steps=900, keep_float_episodes_pct=.2, focused_train_loops=1, show=False)
with open("replay2.json","w") as f:
    f.write(json.dumps(episode_rewards2))

_, episode_rewards3 = train_agent3(episodes=1000, alpha=.0005, minibatch_size=64, epsilon_decay=.99, final_epsilon=.1, update_target_network_frequency=1, max_steps=900, keep_float_episodes_pct=.2, focused_train_loops=1, show=False, replay_limit=2000)
with open("replay3.json","w") as f:
    f.write(json.dumps(episode_rewards3))

_, episode_rewards4 = train_agent3(episodes=1000, alpha=.0005, minibatch_size=64, epsilon_decay=.99, final_epsilon=.1, update_target_network_frequency=1, max_steps=900, keep_float_episodes_pct=.2, focused_train_loops=1, show=False, replay_limit=20000)
with open("replay4.json","w") as f:
    f.write(json.dumps(episode_rewards4))

_, episode_rewards5 = train_agent3(episodes=1000, alpha=.0005, minibatch_size=64, epsilon_decay=.99, final_epsilon=.1, update_target_network_frequency=1, max_steps=900, keep_float_episodes_pct=.2, focused_train_loops=1, show=False, replay_limit=50000)
with open("replay5.json","w") as f:
    f.write(json.dumps(episode_rewards5))

_, episode_rewards6 = train_agent3(episodes=1000, alpha=.0005, minibatch_size=64, epsilon_decay=.99, final_epsilon=.1, update_target_network_frequency=1, max_steps=900, keep_float_episodes_pct=.2, focused_train_loops=1, show=False, replay_limit=100000)
with open("replay6.json","w") as f:
    f.write(json.dumps(episode_rewards6))


#Want following tests -
#1  ZERO replay - sample is 64 previous steps each time
#2. Buffer of ONLY previous episode, but randomized
#3. Static Buffer of 2k
#4. Static buffer of 20k
#5. Static buffer of 50k
#6. NO LIMIT

