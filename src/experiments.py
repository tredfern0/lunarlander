import ll_pytorch
import json
import time
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('font', size=16)

def run1():
    #Experiment 1: update_target_network_frequency
    #Try: 0 (never save), 1 (every episode), 2, 5, 10, 20, 30, 40.  8 runs.
    exp1 = {}
    i=0
    for utnf in [0, 1, 2, 5, 10, 25, 50, 100]:
        episode_rewards = ll_pytorch.train_agent(episodes=1000, alpha=.0005, minibatch_size=64, epsilon_decay=.99, final_epsilon=.1, update_target_network_frequency=utnf, max_steps=900, keep_float_episodes_pct=.2, focused_train_loops=1, show=False)
        exp1[i] = episode_rewards
        i+=1
    exp1_data = json.dumps(exp1)
    with open("exp1.json", "w") as f:
        f.write(exp1_data)

def run7():
    #Experiment 7: max_steps vs min_alpha
    #see if we can cap episodes in such a way to not waste time floating around and discover landings faster
    exp7 = {}
    i=0
    episode_times = []
    for max_steps in [300, 400, 500, 900]:
        for fin_eps in [.2, .1, .05, .01]:
            start_time = time.time()
            print("RUNNING EXP7", max_steps, fin_eps)
            _, episode_rewards = ll_pytorch.train_agent(episodes=2, alpha=.0005, minibatch_size=64, epsilon_decay=.99, final_epsilon=fin_eps, update_target_network_frequency=1, max_steps=max_steps, keep_float_episodes_pct=.2, focused_train_loops=1, show=False)
            total_time = time.time() - start_time
            episode_times.append(total_time)
            exp7[i] = episode_rewards
            i+=1
    exp7_data = json.dumps(exp7)
    exp7_times = json.dumps(episode_times)
    with open("exp7.json", "w") as f:
        f.write(exp7_data)
    with open("exp7_times.json", "w") as f:
        f.write(exp7_times)


def plot_results1(plot_data, keys, trailing_period, fig_name, fig_title):
    fig = plt.figure()
    ax = plt.subplot(111)
    plt.xlabel('Episode')
    plt.ylabel('Reward (Avg of Last {})'.format(trailing_period))
    plt.ylim(-150,300)
    plt.grid(True)

    #http://tristen.ca/hcl-picker/#/hlc/8/1.1/3D1A18/EBFE67
    color_scheme = {"7":"#3D1A18",
                   "6":"#5D2F40",
                   "5":"#695070",
                   "4":"#58789C",
                   "3":"#24A1B1",
                   "2":"#29C7A7",
                   "1":"#1f77b4",#84E888 Using default matplotlib color
                   "0":"#EBFE67",
                   }
    for i in ["7","6","5","4","3","2","1","0",]:
        ax.plot(plot_data[i], linewidth=3, label=keys[i], color=color_scheme[i])

    #Plotting it again so the line is on top - messes up legend when we try to do it above
    i="1"
    ax.plot(plot_data[i], linewidth=3, color=color_scheme[i])

    #ax.legend()
    ax.legend(bbox_to_anchor=(1, 1), fontsize='small')
    plt.title(fig_title)
    plt.tight_layout()
    fig.savefig('{}.png'.format(fig_name))
    plt.show()


def plot1():
    #Experiment 1: update_target_network_frequency
    #Try: 0 (never save), 1 (every episode), 2, 5, 10, 20, 30, 40.  8 runs.
    with open("exp1.json", "r") as f:
        data = f.read()
        data = json.loads(data)

    #print("KEYS", [k for k in data])
    i=0
    plot_data = {}
    keys = {}
    trailing_period = 30
    for utnf in [0, 1, 2, 5, 10, 25, 50, 100]:
        i = str(i)
        keys[i] = utnf
        data_here = []
        for j in range(trailing_period, len(data[i])):
        #Want to plot
            using = data[str(i)][j-trailing_period:j]
            avg = sum(using)/len(using)
            data_here.append(avg)
        #Make it match the length
        data_here = [data_here[0] for _ in range(trailing_period)] + data_here
        plot_data[i] = data_here
        i = int(i)+1
    keys["0"]="Never"
    plot_results1(plot_data, keys, trailing_period, "experiment1", "Update Target Network Frequency")


def plot_results_replay(plot_data, keys, trailing_period, fig_name, fig_title):
    fig = plt.figure()
    ax = plt.subplot(111)
    plt.xlabel('Episode')
    plt.ylabel('Reward (Avg of Last {})'.format(trailing_period))
    plt.ylim(-300,300)
    plt.grid(True)

    #https://colorbrewer2.org/#type=qualitative&scheme=Set3&n=6
    color_scheme = {"5":"#8dd3c7",
                   "4":"#1f77b4", #"#ffffb3",Using default matplotlib color
                   "3":"#bebada",
                   "2":"#fb8072",
                   "1":"#80b1d3",
                   "0":"#fdb462",
                   }
    for i in ["5","4","3","2","1","0",]:
        ax.plot(plot_data[i], linewidth=3, label=keys[i], color=color_scheme[i])

    #Plotting it again so the line is on top - messes up legend when we try to do it above
    i="4"
    ax.plot(plot_data[i], linewidth=3, color=color_scheme[i])

    #ax.legend()
    ax.legend(bbox_to_anchor=(1, 1), fontsize='small')
    plt.title(fig_title)
    plt.tight_layout()
    fig.savefig('{}.png'.format(fig_name))
    plt.show()

def plot_replay():
    with open("replay1.json", "r") as f:
        r1 = json.loads(f.read())
    with open("replay2.json", "r") as f:
        r2 = json.loads(f.read())
    with open("replay3.json", "r") as f:
        r3 = json.loads(f.read())
    with open("replay4.json", "r") as f:
        r4 = json.loads(f.read())
    with open("replay5.json", "r") as f:
        r5 = json.loads(f.read())
    with open("replay6.json", "r") as f:
        r6 = json.loads(f.read())

    #1  ZERO replay - sample is 64 previous steps each time
    #2. Buffer of ONLY previous episode, but randomized
    #3. Static Buffer of 2k
    #4. Static buffer of 20k
    #5. Static buffer of 50k
    #6. NO LIMIT
    trailing_period = 30
    keys = {"0":"No Replay",
            "1":"Prev Episode",
            "2":"2k",
            "3":"20k",
            "4":"50k",
            "5":"No Limit",
            }

    rs = [r1,r2,r3,r4,r5,r6]
    plot_data = {}

    for i in range(len(rs)):
        data = rs[i]
        print("I", i)
        data_here = []
        for j in range(trailing_period, len(data)):
        #Want to plot
            using = data[j-trailing_period:j]
            avg = sum(using)/len(using)
            data_here.append(avg)
        #Make it match the length
        data_here = [data_here[0] for _ in range(trailing_period)] + data_here
        i = str(i)
        plot_data[i] = data_here

    plot_results_replay(plot_data, keys, trailing_period, "experiment3", "Experience Replay Buffer Comparison")

def plot_results7(plot_data, data_x, keys, trailing_period, fig_name, fig_title):
    fig = plt.figure()


    gs = fig.add_gridspec(4,4)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[2, 0])
    ax3 = fig.add_subplot(gs[3, 0])
    ax4 = fig.add_subplot(gs[0, 1])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[2, 1])
    ax7 = fig.add_subplot(gs[3, 1])
    ax8 = fig.add_subplot(gs[0, 2])
    ax9 = fig.add_subplot(gs[1, 2])
    ax10 = fig.add_subplot(gs[2, 2])
    ax11 = fig.add_subplot(gs[3, 2])
    ax12 = fig.add_subplot(gs[0, 3])
    ax13 = fig.add_subplot(gs[1, 3])
    ax14 = fig.add_subplot(gs[2, 3])
    ax15 = fig.add_subplot(gs[3, 3])


    all_ax = [ax0,ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14,ax15]
    default_color ="#78c679"
    our_color = "#1f77b4"
    full_line = [200 for _ in range(3600)]
    for i in range(16):
        ax = all_ax[i]
        i = str(i)
        use_color = default_color if i!="13" else our_color
        #ax.plot(data_x[i], plot_data[i], linewidth=3, color=default_color)
        ax.plot(data_x[i], plot_data[i], linewidth=3, color=use_color)
        ax.plot(full_line, linewidth=1, color="#a5b0b8")
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.set_ylim(-150,300)
        ax.set_xlim(0,15*60)
        #ax.set_facecolor('grey')


    #ax.legend()
    #ax.legend(bbox_to_anchor=(1, 1), fontsize='small')
    fig.suptitle(fig_title)

    fig.text(0.01, 0.5, 'Reward (Avg of Last {})'.format(trailing_period), va='center', rotation='vertical')
    #fig.text(0.45, 0.03, 'Episode', va='center')

    fig.text(0.09, 0.89, '300 Acts', va='center', fontsize="small")
    fig.text(0.31, 0.89, '500 Acts', va='center', fontsize="small")
    fig.text(0.53, 0.89, '700 Acts', va='center', fontsize="small")
    fig.text(0.75, 0.89, '900 Acts', va='center', fontsize="small")

    fig.text(0.93, 0.78, ' $\epsilon$\n0.2', va='center', fontsize="small")
    fig.text(0.93, 0.56, ' $\epsilon$\n0.1', va='center', fontsize="small")
    fig.text(0.93, 0.35, ' $\epsilon$\n0.05', va='center', fontsize="small")
    fig.text(0.93, 0.13, ' $\epsilon$\n0.01', va='center', fontsize="small")

    plt.tight_layout(pad=0)
    plt.subplots_adjust(top=0.87, left=0.05, right=.92)
    fig.savefig('{}.png'.format(fig_name))
    plt.show()




def plot7():
    #Experiment 7: max_steps vs min_alpha
    #see if we can cap episodes in such a way to not waste time floating around and discover landings faster
    with open("exp7.json", "r") as f:
        data = f.read()
        data = json.loads(data)
    with open("exp7_times.json","r") as f:
        data_time = json.loads(f.read())
        #print("TIME", data_time)

    cap_time = 15*60  #So x_axis will be 3600
    data_x = {}
    for i in data:
        #Cut off anything greater than an hour
        if data_time[int(i)]>cap_time:
            keep_ratio = cap_time / data_time[int(i)]
            keep_num = int(keep_ratio * len(data[i]))
            data[i] = data[i][:keep_num]
        #Now need to generate accompanying x array
        #Should range from 0 to our max_time, with number of steps equal to the length
        x_max = min(cap_time, data_time[int(i)])
        step = (1 / len(data[i])) * x_max
        x_vals = [step*j for j in range(len(data[i]))]
        data_x[i] = x_vals
        #print("MAX I", i, x_vals[-1])


    #print("KEYS", [k for k in data])
    i=0
    plot_data = {}
    keys = {}
    trailing_period = 100
    for max_steps in [300, 500, 700, 900]:
        for fin_eps in [.2, .1, .05, .01]:
            i = str(i)
            keys[i] = "{} {}".format(max_steps, fin_eps)
            data_here = []
            for j in range(trailing_period, len(data[i])):
            #Want to plot
                using = data[str(i)][j-trailing_period:j]
                avg = sum(using)/len(using)
                data_here.append(avg)
            #Make it match the length
            data_here = [data_here[0] for _ in range(trailing_period)] + data_here
            plot_data[i] = data_here
            i = int(i)+1

    plot_results7(plot_data, data_x, keys, trailing_period, "experiment2", "Solving Progress in 15 Minutes")


#plot1()
#plot7()
#plot_replay()
