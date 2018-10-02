import os,re
from matplotlib import pyplot as plt

epoch = []
it = []
d = []
g = []
idx = []
with open('./report/mnist.log', 'r') as f:
    for i, line in enumerate(f.readlines()):
        # pattern = r'INFO:root:Epoch/Iter:(?P<epoch>)/(?P<iter>), Dloss: (?P<d>), Gloss: (?P<g>)'
        pattern = r'INFO:root:Epoch/Iter:(\d+)/(\d+), Dloss: ([\d.]+), Gloss: ([\d.]+)'
        match = re.match(pattern, line)
        # print(line)
        idx.append(i/10)
        epoch.append(int(match.group(1)))
        it.append(int(match.group(2)))
        d.append(float(match.group(3)))
        g.append(float(match.group(4)))

plt.plot(idx, g, label='G')
plt.plot(idx, d, label='D')
plt.xlabel('# of epoch')
plt.ylabel('loss')
plt.legend()
plt.show()