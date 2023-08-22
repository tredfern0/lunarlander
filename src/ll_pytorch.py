#https://gym.openai.com/envs/LunarLander-v2/
import gym
import numpy as np
import copy
import json
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('font', size=16)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#discount - can hardcode as never changing this
GAMMA = .99
np.random.seed(0)

class Net(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        #fc = fully connected
        self.fc1 = nn.Linear(8, 256, bias=True)
        self.fc2 = nn.Linear(256, 128, bias=True)
        self.fc3 = nn.Linear(128, 4, bias=True)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.criterion = nn.MSELoss()
    def forward(self, x):
        """ x is the raw input """
        #Applying activation functions here
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def choose_action(x, net, epsilon, num_actions, printa=False):
    eps = np.random.random()  #decide whether the action is greedy
    if eps<epsilon:
        #Only draw a random action when epsilon indicates
        Aprime = np.random.randint(0, num_actions)
    else:
        output = net(torch.Tensor(x))
        if printa:
            print(output)
        Aprime = int(torch.argmax(output))
    return Aprime


def update_net(net, states_arr, target_arr, target_mult):
    output = net(states_arr)
    net.optimizer.zero_grad()
    final_target_arr = (target_mult * output) + target_arr
    #final_target_arr: the LABELS, what we want to train towards
    #output: PREDICTIONS
    loss = net.criterion(final_target_arr, output)
    loss.backward()
    net.optimizer.step()

def run_update(samples, net, net_static, num_updates=32):

    #states, actions, rewards, state_prime, terminal_mult
    states = [x[0] for x in samples]
    states_arr = torch.Tensor(np.vstack((states)))

    actions = [x[1] for x in samples]

    #0 if true terminal state - does not include time limit cutoffs.
    #We do NOT want to bootstrap on true terminal states
    terminal_mult = torch.Tensor([x[4] for x in samples])

    #Need to use value estimates from our cached net
    sprimes = [x[3] for x in samples]
    sprimes_arr = torch.Tensor(np.vstack((sprimes)))
    out = net_static(sprimes_arr)

    #Q-learning, so we want max action
    max_vals, _ = torch.max(out, 1)
    max_vals = max_vals * terminal_mult
    best = GAMMA*max_vals

    rewards = torch.Tensor([x[2] for x in samples])

    target_vals = best+rewards
    target_vals = target_vals.view(num_updates,1)

    #These are state,action updates, so we need to match our output
    #if neural net output is [[1,2,3,4],[5,6,7,8]]
    #If we have value estimates of 10 and 20 for the 0th and 1st action
    #We need [[10,2,3,4],[5,20,7,8]] to train the network
    a = torch.as_tensor(actions)
    target_arr = F.one_hot(a, num_classes=4)

    #Want the inverse of our one_hot array, so we can zero out the predictions and add our target array
    target_mult = (-1*target_arr)+1
    #With broadcasting and one hot format will have 0s except for where we have value estimates
    target_arr = target_arr * target_vals

    update_net(net, states_arr, target_arr, target_mult)


############# Failed experiment
replay_buffer_buckets = {i:[] for i in range(4)}
def append_to_replay_bucket(replay_buffer_buckets, experience_tuple):
    y_val = experience_tuple[0][1]
    if y_val < .01:
        replay_buffer_buckets[0].append(experience_tuple)
    elif y_val <.2:
        replay_buffer_buckets[1].append(experience_tuple)
    elif y_val <.4:
        replay_buffer_buckets[2].append(experience_tuple)
    elif y_val <.6:
        replay_buffer_buckets[3].append(experience_tuple)

def choose_samples_buckets(replay_buffer_buckets, num_samples=32):
    try:
        samples = []
        num_updates = int(num_samples/4)
        for j in range(4):
            sample_indices = np.random.randint(0, len(replay_buffer_buckets[j]), num_updates)
            samp = [replay_buffer_buckets[j][i] for i in sample_indices]
            samples.extend(samp)
        return samples
    except Exception as e:
        print("SAMPLING FAILED", e)
        return []

def choose_samples_buffer(replay_buffer, num_updates):
    sample_indices = np.random.randint(0, len(replay_buffer), num_updates)
    samples = [replay_buffer[i] for i in sample_indices]
    return samples


def train_agent(episodes=1000, alpha=.0005, minibatch_size=64, epsilon_decay=.99, final_epsilon=.1, update_target_network_frequency=1, max_steps=900, keep_float_episodes_pct=.2, focused_train_loops=1, show=True):
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

        floated = False
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
                    for _ in range(focused_train_loops):
                        samples = choose_samples_buffer(episode_buffer, minibatch_size)
                        run_update(samples, net, net_static, num_updates=minibatch_size)

            if len(replay_buffer)>minibatch_size:
                samples = choose_samples_buffer(replay_buffer, minibatch_size)
                run_update(samples, net, net_static, num_updates=minibatch_size)

            x, reward, terminal, null = result
            #x = mod_x(x)

            if terminal:
                break

            if i>max_steps:
                accept = np.random.random()
                #Only save some% of these episodes in our buffer
                if accept>keep_float_episodes_pct:
                    floated = True
                break

        #Update target weights every other episode
        if update_target_network_frequency and i%update_target_network_frequency==0:
            net_static = copy.deepcopy(net)

        #Extend buffer for all ticks in current episode
        if not floated:
            replay_buffer.extend(episode_buffer)
        replay_buffer = replay_buffer[-50000:]

        avg = sum(episode_rewards[-100:])/100
        best_avg = max(best_avg, avg)
        print("EP", j, "STEPS", i, "EPSILON", epsilon, "REWARD", episode_reward, "AVG",avg)
        trailing_avg.append(avg)
        #Just for experiment7
        #if avg>200:
            #break
        #if avg>200:
            #torch.save(net, "./mod{}.torch".format(j))
        #Just for real training
        #if j>0 and j%1000==0:
        #    v = int(j/1000)
        #    torch.save(net, "./mod_{}.torch".format(v))
        episode_rewards.append(episode_reward)
    print("LANDINGS", landings, "BEST AVG", best_avg)
    env.close()
    return net, episode_rewards


def calc_trailing_average(episode_rewards):
    trailing_avg = []
    for i in range(1,len(episode_rewards)+1):
        startI = max(0, i-100)
        seq = episode_rewards[startI:i]
        avg = sum(seq)/len(seq)
        trailing_avg.append(avg)
    return trailing_avg

def train_default():
    #Original
    #episode_rewards = train_agent(alpha=.0005, minibatch_size=64, epsilon_decay=.99, final_epsilon=.05, update_target_network_frequency=2, max_steps=900, keep_float_episodes_pct=.2, show=False)
    #exp1_data = json.dumps(episode_rewards)
    #with open("res1.json", "w") as f:
    #    f.write(exp1_data)

    #Improved?
    net, episode_rewards = train_agent(episodes=3001, alpha=.0005, minibatch_size=64, epsilon_decay=.99, final_epsilon=.1, update_target_network_frequency=1, max_steps=900, keep_float_episodes_pct=.2, focused_train_loops=1, show=False)

    v = "v04"
    #torch.save(net, "./mod_{}.torch".format(v))
    data = json.dumps(episode_rewards)
    with open("{}.json".format(v), "w") as f:
        f.write(data)

    #trailing_avg = calc_trailing_average(episode_rewards)
    #print("AVERAGE REWARD - LAST 100", sum(episode_rewards)/len(episode_rewards))

def build_plot(train_file="v04", test_file="episode_rewards"):
    #Open the .json file with the episode by epsiode results
    with open("{}.json".format(train_file),"r") as f:
        train_rewards = json.loads(f.read())[:1000]
    with open("{}.json".format(test_file),"r") as f:
        test_rewards = json.loads(f.read())
    train_trailing_avg = calc_trailing_average(train_rewards)
    test_avg = sum(test_rewards) / len(test_rewards)

    avg_line = [test_avg for _ in range(len(test_rewards))]


    fig = plt.figure()

    #https://stackoverflow.com/questions/2265319/how-to-make-an-axes-occupy-multiple-subplots-with-pyplot-python
    gs = fig.add_gridspec(5,1)
    ax0 = fig.add_subplot(gs[0:3, 0])
    ax1 = fig.add_subplot(gs[3:5,0])

    plt.xlabel('Episode')
    #plt.ylabel('Reward')

    #y_error_vals = [plot_data[x] for x in x_lambda_vals]
    #episodes = list(range(1, len(plot_data)+1))
    ax0.plot(train_rewards, linewidth=1)
    ax0.plot(train_trailing_avg, linewidth=2, label="100 Avg")

    ax0.set_ylim(-400,330)
    #ax0.yaxis.set_ticklabels([-200, 0, 200])
    ax0.grid(True)
    ax0.set_title("Train Rewards Per Episode", fontsize="small")
    ax0.set_ylabel("Reward")

    ax1.plot(test_rewards)
    ax1.plot(avg_line, label="Avg")
    #ax1.yaxis.set_ticklabels([-200, 0, 200, 300])
    ax1.set_ylim(-200,330)
    ax1.grid(True)
    ax1.set_title("Test Rewards Per Episode", fontsize="small")
    ax1.set_ylabel("Reward")

    ax0.legend(fontsize='x-small')
    ax1.legend(fontsize='x-small')


    fig.suptitle("Agent Performance")
    #fig.text(0.01, 0.5, 'Reward', va='center', rotation='vertical', fontsize="large")
    #ax.legend()
    #plt.title("Fig")
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    fig.savefig('results.png')
    #plt.show()


def run_model(path="./mod_1.torch", show=True):
    env = gym.make('LunarLander-v2')
    env.seed(0)

    episode_rewards = []
    net = torch.load(path)
    landings = 0
    for j in range(100):
        epsilon = 0
        x = env.reset()
        episode_reward = 0
        while True:
            A = choose_action(x, net, epsilon, 4, False)
            result = env.step(A)
            if show:
                env.render()
            x, reward, terminal, null = result
            episode_reward += reward
            if terminal:
                print("EPISODE", j, "REWARD", episode_reward)
                if reward==100:
                    landings+=1
                break
        episode_rewards.append(episode_reward)
    print("AVG REWARD", sum(episode_rewards)/len(episode_rewards))
    print("LANDINGS", landings)
    with open("episode_rewards.json", "w") as f:
        f.write(json.dumps(episode_rewards))
    env.close()


if __name__=="__main__":
    run_model(path="./mod_1.torch", show=False)
    #build_plot(train_file="v04", test_file="episode_rewards")
    #train_default()
