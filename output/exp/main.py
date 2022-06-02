import matplotlib.pyplot as plt

if __name__ == "__main__":
    l = []
    for i in range(607):
        l.append(int(input()))

    plt.boxplot(l)
    plt.show()

    sum = 0
    up_40 = 0
    for i in l:
        sum += abs(i)
        if abs(i)>=40:
            up_40 += 1
    print(sum/len(l))
    print(up_40)