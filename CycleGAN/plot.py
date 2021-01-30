import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np

def draw_plot(data):
    y = np.zeros(150)
    for datum in data:
        d = int(np.clip((datum // 0.1) + 75, a_max=149, a_min=0))
        y[d] += 1
    return y

if __name__ == '__main__':
    with open('./dataset2/transfer_latent_vector_1.pkl', 'rb') as f:
        data1 = pkl.load(f)
    with open('./dataset2/transfer_latent_vector_2.pkl', 'rb') as f:
        data2 = pkl.load(f)
    with open('./dataset2/latent_vector_1.pkl', 'rb') as f:
        data3 = pkl.load(f)
    with open('./dataset2/latent_vector_2.pkl', 'rb') as f:
        data4 = pkl.load(f)
    data1 = data1.transpose()
    data2 = data2.transpose()
    data3 = data3.transpose()
    data4 = data4.transpose()
    for i in range(1):
        # y1 = draw_plot(data1[i])
        # y2 = draw_plot(data2[i])
        # y3 = draw_plot(data3[i])
        # y4 = draw_plot(data4[i])
        plt.plot(data1[0][:1000], data1[1][:1000],'.', color='red')
        plt.plot(data3[0][:1000], data3[1][:1000],'.', color='blue')
        plt.show()
        plt.plot(data2[0][:1000], data2[1][:1000],'.', color='red')
        plt.plot(data4[0][:1000], data4[1][:1000],'.', color='blue')
        plt.show()
        plt.plot(data1[2][:1000], data1[3][:1000],'.', color='red')
        plt.plot(data3[2][:1000], data3[3][:1000],'.', color='blue')
        plt.show()
        plt.plot(data2[2][:1000], data2[3][:1000],'.', color='red')
        plt.plot(data4[2][:1000], data4[3][:1000],'.', color='blue')
        plt.show()
        # plt.plot(y2, color='green')
        # plt.plot(y3, color='blue')
        # plt.plot(y4, color='cyan')
