import scipy as sp
import matplotlib.pyplot as plt


def error(f, x, y):
    return sp.sum((f(x) - y) ** 2)


data = sp.genfromtxt("web_traffic.tsv", delimiter="\t")

print(data[:10])
print(data.shape)

x = data[:,0]
y = data[:,1]

print(sp.sum(sp.isnan(y)))

# ndim 1
fp1, residuals, rank, sv, rcond = sp.polyfit(x,y,1, full=True)
f1 = sp.poly1d(fp1)

print("Model parameters: %s" % fp1)
print(residuals)

fx = sp.linspace(0,x[-1],1000)
plt.plot(fx, f1(fx), linewidth=4)

# ndim 2
f2p = sp.polyfit(x, y, 2)
print(f2p)
f2 = sp.poly1d(f2p)
print(error(f2,x,y))

plt.plot(fx, f2(fx), linewidth=4)

plt.legend(["d=%i " % w for w in range(1, 3, 1)], loc="upper left")

plt.scatter(x,y)
plt.title("Web traffic over the last month")
plt.xlabel("Time")
plt.ylabel("Hits/hour")
plt.xticks([w*7*24 for w in range(10)],['week %i '%w for w in range(10)])
plt.autoscale(tight=True)
plt.grid()
plt.show()
