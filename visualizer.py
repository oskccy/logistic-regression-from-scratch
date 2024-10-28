import matplotlib.pyplot as plt


def loss_visualizer(vec_x, vec_y):

    fig, ax = plt.subplots()
    ax.plot(vec_x, vec_y)
    ax.set_title(
        'Loss Function in Relation to Epochs (Error)')
    plt.show()
