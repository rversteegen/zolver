import re
import matplotlib.pyplot as plt
import scipy.optimize
import numpy as np

"""
Graph token generation rate, CPU/GPU utilitisation and other performance counters from a notebook log.
"""

SHOW_CPUSTATS = True#False
SHOW_CPUTOTALS = True#False

OLD_NB = False

OLD_NB, logfile = True, "logs/deepseek-new-baselinev18.log" #slowdown from 1000s, then 60% from 2400s
#OLD_NB, logfile = True, "logs/deepseek-new-baselinev41.log" # slowdown from 2800s, more from 3800s
#OLD_NB, logfile = True, "logs/deepseek-new-baselinev48.log"  # 97min gap, then slowdown
#logfile = "logs/deepseek-new-baselinev53.log"   # 1xT4 No slowdown, but still 200% CPU usage

#logfile = "logs/deepseek-new-baselinev55.log"  # 1xT4  No MHz, initial ramping up slowdown, later high ctx_switches
# Also, slowdown despite proc CPU at nearly 100%

#logfile = "logs/deepseek-new-baselinev57.log"  # 2xT4 (balanced) pegged afer 3000s, but variable genrate, 50%GPU
#logfile = "logs/deepseek-new-baselinev60.log" # P100, constant MHz, 505 slowdown after 3000s
#logfile = "logs/vllmv0.log"

data = []
token_gen_rate = []
gentimes = []
threadtimes = []
proctimes = []

# Outputs of psutil
scpustats = []   # psutil.cpu_stats()
scputimes = []   # psutil.cpu_times()
scpu_logtimes = []  # Log time

temps = []
MHz = []
GB = []
memutil = []
gpuutil = []
totalgaps = 0.0

with open(logfile) as infile:
    lasttime = 0
    for line in infile:
        linetime = float(line.split('s')[0])

        if OLD_NB:
            m = re.search(r"<<<<<GEN (\d+) tokens \((\d+) total\) in ([0-9.]+)s", line)
            if m:
                toks = int(m.group(1))
                totaltoks = int(m.group(2))
                time = float(m.group(3))
                #if time > 200:
                #    continue
                data.append([toks, totaltoks, time])
        else:
            # eg
            # 218.8s 45 <<<<<GEN 146 tokens (299 total) in 8.89+0.06s (16.31 tok/s) (36.1/37.3s CPU) (cuda0: 13.34GB,100%) (cuda1: 0.00GB,0%) (cuda0: 64°C,100%) (cuda1: 45°C,0%)
            m = re.search(r"<<<<<GEN (\d+) tokens \((\d+) total\) in ([0-9.]+)\+[0-9.]+s \(([0-9.]+) tok/s", line)
            if m:
                toks = int(m.group(1)) - 1
                totaltoks = int(m.group(2))
                time = float(m.group(3))
                gap = linetime - (lasttime + time)
                assert gap >= -1e2
                if lasttime:  # Not first GEN
                    totalgaps += gap
                    # These big gaps are just due to OOMs causing missing GENs
                    # if gap > 15:
                    #     print("gap", gap, ":")
                    #     print(line)
                speed = float(m.group(4))
                data.append([toks, totaltoks, time])
                token_gen_rate.append(speed)


                matches = re.findall(r"\(cuda.: (\d+)°C,(\d+)%(,(\d+)MHz)?", line)
                temps.append([float(m[0]) for m in matches])
                gpuutil.append([float(m[1]) for m in matches])
                if matches[0][3]:
                    MHz.append([float(m[3]) for m in matches])

        if m:
            m = re.search(r"\(([0-9.]+)/([0-9.]+)s CPU\)", line)
            threadtimes.append(float(m.group(1)))  # The thread
            proctimes.append(float(m.group(2)))  # The process

            #if time > 200:
            #    continue
            matches = re.findall(r"\(cuda.: ([0-9.]+)GB(,(\d+))?", line)
            GB.append([float(m[0]) for m in matches])
            if matches[0][2]:
                memutil.append([float(m[2]) for m in matches])

        if m:
            # eg "218.8s 45 <<<<<GEN 146 tokens..."
            gentimes.append(linetime)
            lasttime = linetime

        if "scputimes" in line:
            m = re.search(r"user=([0-9.]+).*system=([0-9.]+).*iowait=([0-9.]+).*softirq=([0-9.]+)", line)
            scputimes.append([float(x) for x in m.groups()])
            scpu_logtimes.append(linetime)
        if "scpustats" in line:
            m = re.search(r"ctx_switches=(\d+), interrupts=(\d+), soft_interrupts=(\d+)", line)
            scpustats.append([float(x) for x in m.groups()])


print("totalgaps", totalgaps)

dat = np.array(data).T
print("shape", dat.shape)

def residuals(regression_param):
    "For least squares regression, "
    t_prompt, t_gen = regression_param  # Time per token in the prompt or generated
    time = dat[2]
    genlen = dat[0]
    promptlen = dat[1] - dat[0]
    ret = (time - promptlen * t_prompt - genlen * t_gen)
    #print("RET", ret.sum())
    return ret

def gen_speed(regression_param):
    "Estimate the "
    t_prompt, t_gen = regression_param
    time = dat[2]
    genlen = dat[0]
    promptlen = dat[1] - dat[0]
    ret = genlen / (time - promptlen * t_prompt)
    return ret

#residuals, [0.01, 0.05]))

soln, _ = scipy.optimize.leastsq(residuals, [0.01, 0.05])
print('leastsq:', soln)


if SHOW_CPUTOTALS or SHOW_CPUSTATS:
    fig, (ax1, ax3, ax5, ax6) = plt.subplots(4,1)
    ax6.set_xlabel("Time (sec)")
else:
    fig, (ax1, ax3, ax5) = plt.subplots(3,1)
    ax5.set_xlabel("Time (sec)")

#l1
#soln = [0.00250588, 0.05673307]
#l2
#soln = [0.00209891, 0.06174432]

gentimes = np.array(gentimes)

# Use negative widths to align bars on their right edges
#bartimes = np.array([0] + gentimes)
#barwidths = -(bartimes[1:] - bartimes[:-1])

barwidths = -dat[2]  # times


if token_gen_rate == []:   # V48
    token_gen_rate = gen_speed(soln)

ax1.set_ylabel("Tokens/sec generated")
ax1.bar(gentimes, token_gen_rate, width = barwidths, align='edge', color = 'grey', edgecolor='black', linewidth=1, )
#plt.plot(gentimes, token_gen_rate)
# ax1.set_ylim(top = max(token_gen_rate) * 1.4)  # Leave some space above
# ax1.set_ylim(bottom = 0,top = 23)  # Leave some space above

# ax2 = ax1.twinx()
# ax2.plot(gentimes, GB, color="blue")
# ax2.set_ylabel("GPU memory used/GB", color="blue")


ax4 = ax3.twinx()

if len(temps):
    ax3.set_ylabel("GPU temperature/°C")#, color="blue")
    ax3.plot(gentimes, temps)

if len(MHz):
    ax4.set_ylabel("::::: GPU clock/MHz")
    MHz = np.array(MHz)
    #for y, color in zip(MHz.T, 
    ax4.plot(gentimes, MHz, linestyle=":")

ax5.set_ylabel("CPU/GPU Utilisation%")
#ax5.set_yticks(np.arange(0,401,step=50))
ax5.grid(axis='y')

scputimes = np.array(scputimes)
scpu_logtimes = np.array(scpu_logtimes)
threadtimes = np.array(threadtimes)
proctimes = np.array(proctimes)

def plot_rate_of_change(ax, logtimes, values, *args, **kwargs):
    # Plot average rate of change of values, eg. CPU util %, over each logging time interval
    rate = (values[1:] - values[:-1]) / (logtimes[1:] - logtimes[:-1])
    ax.step(logtimes[1:], rate, *args, where='pre', **kwargs)

plot_rate_of_change(ax5, gentimes, 100 * threadtimes, ':', label='Main thread CPU%')
plot_rate_of_change(ax5, gentimes, 100 * proctimes, ':', label='Python process CPU%')

if len(scputimes): #not OLD_NB:
    plot_rate_of_change(ax5, scpu_logtimes, 100 * scputimes[:,0], '--k', label='User CPU%')#, linewidth=2)
    plot_rate_of_change(ax5, scpu_logtimes, 100 * scputimes[:,1], '--b', label='System CPU%')#, linewidth=2)

if len(gpuutil):
    gpuutil = np.array(gpuutil)
    for gpu in range(gpuutil.shape[1]):
        ax5.plot(gentimes, gpuutil[:,gpu], label=f"GPU{gpu}%")
    # Always equal to gpu utilitization%
    # memutil = np.array(memutil)
    # ax5.plot(gentimes, memutil[:,0], label="GPU0mem")
    # ax5.plot(gentimes, memutil[:,1], label="GPU1mem")
ax5.legend(loc='best')

if SHOW_CPUSTATS:
    ax6.set_ylabel("CPU counter increase/sec")
    scpustats = np.array(scpustats)
    if len(scpustats):
        for column, label in zip(range(scpustats.shape[1]), "ctx_switches interrupts soft_interrupts".split()):
            plot_rate_of_change(ax6, scpu_logtimes, 100 * scpustats[:,column], label=label, linewidth=2)
            #ax6.step(scpu_logtimes, scpustats[:,column], where='pre', label=label)
    ax6.legend()

elif SHOW_CPUTIMES:
    # Plot the total CPU times rather than rates of change
    ax6.set_ylabel("CPU seconds")
    ax6.plot(gentimes, threadtimes, '--', label="Main thread")
    ax6.plot(gentimes, proctimes, '--', label="Python process")
    if len(scputimes):
        for column, label in zip(range(scputimes.shape[1]), "user system iowait softirq".split()):
            ax6.plot(scpu_logtimes, scputimes[:,column], label=label)
    ax6.legend()

# ax = fig.add_subplot(projection='3d')
# for a,b,c in data:
#     ax.scatter(a,b,c)
# #plt.plot(data)
plt.show()
