import re
import matplotlib.pyplot as plt
import scipy.optimize
import numpy as np

"""
Graph token generation rate, CPU/GPU utilitisation and other performance counters from a notebook log.
"""

# Generation stats and CPU/GPU counters are on log lines starting with this word
# e.g.
# 222.4s 40 <<<<<GEN 287 tokens (393 total) in 13.87+2.54s (20.62 tok/s) (39.9/41.1s CPU) (cuda0: 13.49GB,52%) (cuda0: 44°C,52%,1328MHz)  (GPU0:Pstate=P0 [])
START_WORD = "<<<<<GEN"

NCPU = 4  # Number of cores

# Show ctx_switches, etc
SHOW_CPUSTATS = True#False
# else:
SHOW_CPUTOTALS = True#False

logfile = "logs/deepseek-new-baselinev18.log" #slowdown from 1000s, then 60% from 2400s
#logfile = "logs/deepseek-new-baselinev21.log" #
#logfile = "logs/deepseek-new-baselinev35.log" # T4 small slowdown after 500s
#logfile = "logs/deepseek-new-baselinev36.log" # T4 No slowdown in 2h
#logfile = "logs/deepseek-new-baselinev41.log" # slowdown from 2800s, more from 3800s
#logfile = "logs/deepseek-new-baselinev48.log"  # 97min gap, then slowdown

# NEWNB

#logfile = "logs/deepseek-new-baselinev53.log"   # 1xT4 No slowdown, but still 200% CPU usage

logfile = "logs/deepseek-new-baselinev55.log"  # 1xT4  No MHz, initial ramping up slowdown, later high ctx_switches
# Also, slowdown despite proc CPU at nearly 100%

#logfile = "logs/deepseek-new-baselinev57.log"  # 2xT4 (balanced) pegged afer 3000s, but variable genrate, 50%GPU
#logfile = "logs/deepseek-new-baselinev60.log" # P100, constant MHz, 505 slowdown after 3000s
#logfile = "logs/vllm-deepseekv2.log"
#logfile = "logs/vllm-deepseekv3.log"
#logfile = "logs/vllmv0.log"

# Data from GEN log lines
token_data = []  # Tuples (tokens_generated, tokens_processed (omit ones already KV cached), time_to_generate)
token_gen_rate = []  # tokens/s generated, if recorded (otherwise calculated)
gen_logtimes = []
threadtimes = []
proctimes = []
temps = []
MHz = []
GB = []
memutil = []
gpuutil = []
totalgaps = 0.0

result = []
too_slow = []
traceback = []

# Outputs of psutil
scpustats = []   # psutil.cpu_stats()
scputimes = []   # psutil.cpu_times()
scpu_logtimes = []  # Log time

lasttime = 0
last_total_tokens = 0

with open(logfile) as infile:
    for line in infile:
        try:
            linetime = float(line.split('s')[0])
        except:
            # Garbage from tqdm
            pass

        if "PROMPT" in line:
            last_total_tokens = 0

        if "too slow" in line:
            too_slow.append(lasttime)
        if "###<Result" in line:
            result.append(lasttime)
        if "Traceback (" in line:
            traceback.append(lasttime)


        if START_WORD in line:

            ## First try the old logging format I used
            # eg:
            # 20984.9s 32133 <<<<<GEN 36 tokens (488 total) in 3.4s (10.5 tok/s) (18319.6/18339.9s CPU) (cuda0: 14.27GB)
            m = re.search(START_WORD + r" (\d+) tokens \((\d+) total\) in ([0-9.]+)s", line)
            if m:
                toks = int(m.group(1))
                totaltoks = int(m.group(2))
                processedtoks = totaltoks - last_total_tokens
                last_total_tokens = totaltoks
                time = float(m.group(3))
                #if time > 200:
                #    continue
                token_data.append([toks, processedtoks, time])
            else:
                ## Newer logging format

                # eg
            # 218.8s 45 <<<<<GEN 146 tokens (299 total) in 8.89+0.06s (16.31 tok/s) (36.1/37.3s CPU) (cuda0: 13.34GB,100%) (cuda1: 0.00GB,0%) (cuda0: 64°C,100%) (cuda1: 45°C,0%)
                m = re.search(START_WORD + r" (\d+) tokens \((\d+) total\) in ([0-9.]+)\+[0-9.]+s \(([0-9.]+) tok/s", line)
                assert m
                toks = int(m.group(1)) - 1
                totaltoks = int(m.group(2)) - last_total_tokens
                time = float(m.group(3))
                gap = linetime - (lasttime + time)
                assert gap >= -1e2
                if lasttime:  # Not first GEN
                    totalgaps += gap
                rate = float(m.group(4))
                token_data.append([toks, totaltoks, time])
                token_gen_rate.append(rate)

                # GPU temp, util%, clock
                matches = re.findall(r"\(cuda.: (\d+)°C,(\d+)%(,(\d+)MHz)?", line)
                temps.append([float(m[0]) for m in matches])
                gpuutil.append([float(m[1]) for m in matches])
                if matches[0][3]:
                    MHz.append([float(m[3]) for m in matches])

            # CPU time
            m = re.search(r"\(([0-9.]+)/([0-9.]+)s CPU\)", line)
            threadtimes.append(float(m.group(1)))  # The thread
            proctimes.append(float(m.group(2)))  # The process

            # GPU memory
            matches = re.findall(r"\(cuda.: ([0-9.]+)GB(,(\d+))?", line)
            GB.append([float(m[0]) for m in matches])
            if matches[0][2]:
                memutil.append([float(m[2]) for m in matches])

            gen_logtimes.append(linetime)
            lasttime = linetime

        # CPU time and counters
        if "scputimes" in line:
            m = re.search(r"user=([0-9.]+).*nice=([0-9.]+).*system=([0-9.]+).*idle=([0-9.]+).*iowait=([0-9.]+).*softirq=([0-9.]+)", line)
            scputimes.append([float(x) for x in m.groups()])
            scpu_logtimes.append(linetime)
        if "scpustats" in line:
            m = re.search(r"ctx_switches=(\d+), interrupts=(\d+), soft_interrupts=(\d+)", line)
            scpustats.append([float(x) for x in m.groups()])


print("Total gaps", totalgaps, "s")

dat = np.array(token_data).T

for (toks, processedtoks, gentime), logtime in zip(token_data, gen_logtimes):
    if gentime > 200:
        print(f"{gentime}s generation at {logtime}")


if token_gen_rate == []:   # V48
    # Tokens/s generation rate wasn't recorded, so estimate it, subtracting off the
    # time for expanding the KV cache using linear regression.

    def residuals(regression_params):
        "For least squares regression, residuals between predicted and actual generation times"
        t_prompt, t_gen = regression_params  # Time per token in the prompt or generated
        time = dat[2]
        genlen = dat[0]
        promptlen = dat[1] - dat[0]
        ret = (time - promptlen * t_prompt - genlen * t_gen)
        #print("RET", ret.sum())
        return ret

    def gen_rate(regression_params):
        "Estimate the "
        t_prompt, t_gen = regression_params
        time = dat[2]
        genlen = dat[0]
        promptlen = dat[1] - dat[0]
        ret = genlen / (time - promptlen * t_prompt)
        return ret

    regression_params, _ = scipy.optimize.leastsq(residuals, [0.01, 0.05])
    print('leastsq:', regression_params)

    token_gen_rate = gen_rate(regression_params)



starttime = 0 # min(gen_logtimes + scpu_logtimes)
endtime = max(gen_logtimes + scpu_logtimes)
gen_logtimes = np.array(gen_logtimes)

if SHOW_CPUTOTALS or SHOW_CPUSTATS:
    fig, (ax1, ax3, ax5, ax6) = plt.subplots(4,1)
    ax6.set_xlabel("Time (sec)")
    ax6.set_xlim(left = starttime, right = endtime)
else:
    fig, (ax1, ax3, ax5) = plt.subplots(3,1)
    ax5.set_xlabel("Time (sec)")
ax1.set_xlim(left = starttime, right = endtime)
ax3.set_xlim(left = starttime, right = endtime)
ax5.set_xlim(left = starttime, right = endtime)

# Use negative widths to align bars on their right edges
barwidths = -dat[2]  # times

ax1.set_ylabel("Tokens/sec generated")
ax1.bar(gen_logtimes, token_gen_rate, width = barwidths, align='edge', color = 'grey', edgecolor='black', linewidth=1, )
ax1.set_ylim(bottom = 0, top = min(35, max(token_gen_rate)))
ax1.grid(axis='y')


# ax2 = ax1.twinx()
# ax2.plot(gen_logtimes, GB, color="blue")
# ax2.set_ylabel("GPU memory used/GB", color="blue")

ax4 = ax3.twinx()

if len(temps):
    ax3.set_ylabel("GPU temperature/°C")#, color="blue")
    ax3.plot(gen_logtimes, temps)

if len(MHz):
    ax4.set_ylabel("::::: GPU clock/MHz")
    MHz = np.array(MHz)
    #for y in MHz.T:
    ax4.plot(gen_logtimes, MHz, linestyle=":")

ax5.set_ylabel("CPU/GPU Utilisation%")
#ax5.set_yticks(np.arange(0,401,step=25))
ax5.grid(axis='y')

scputimes = np.array(scputimes)
scpu_logtimes = np.array(scpu_logtimes)
threadtimes = np.array(threadtimes)
proctimes = np.array(proctimes)

def plot_rate_of_change(ax, logtimes, values, *args, subtract = 0, **kwargs):
    # Plot average rate of change of values, eg. CPU util %, over each logging time interval
    rate = (values[1:] - values[:-1]) / (logtimes[1:] - logtimes[:-1])
    if subtract:
        rate = subtract - rate
    ax.step(logtimes[1:], rate, *args, where='pre', **kwargs)

plot_rate_of_change(ax5, gen_logtimes, 100 * threadtimes, ':', label='Main thread CPU%')
plot_rate_of_change(ax5, gen_logtimes, 100 * proctimes, ':', label='Python process CPU%')

if len(scputimes):
    plot_rate_of_change(ax5, scpu_logtimes, 100 * scputimes[:,0], '--k', label='User CPU%')#, linewidth=2)
    plot_rate_of_change(ax5, scpu_logtimes, 100 * scputimes[:,1], '--r', label='Nice CPU%')#, linewidth=2)
    plot_rate_of_change(ax5, scpu_logtimes, 100 * scputimes[:,2], '--b', label='System CPU%')#, linewidth=2)
    plot_rate_of_change(ax5, scpu_logtimes, 100 * scputimes[:,3], '-g', label='Total CPU%', subtract=100*NCPU)

ax5.scatter(result, [100] * len(result))# 'p')
ax5.scatter(traceback, [150] * len(traceback))# 'p')
ax5.scatter(too_slow, [200] * len(too_slow))# 'p')


if len(gpuutil):
    gpuutil = np.array(gpuutil)
    for gpu in range(gpuutil.shape[1]):
        ax5.plot(gen_logtimes, gpuutil[:,gpu], label=f"GPU{gpu}%")
    # Always equal to gpu utilitization%
    # memutil = np.array(memutil)
    # ax5.plot(gen_logtimes, memutil[:,0], label="GPU0mem")
    # ax5.plot(gen_logtimes, memutil[:,1], label="GPU1mem")
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
    ax6.plot(gen_logtimes, threadtimes, '--', label="Main thread")
    ax6.plot(gen_logtimes, proctimes, '--', label="Python process")
    if len(scputimes):
        for column, label in zip(range(scputimes.shape[1]), "user nice system idle iowait softirq".split()):
            ax6.plot(scpu_logtimes, scputimes[:,column], label=label)
    ax6.legend()

plt.show()
