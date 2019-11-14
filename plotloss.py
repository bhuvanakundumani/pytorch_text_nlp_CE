import matplotlib.pyplot as plt 

def plot_function(train_loss_plot, valid_loss_plot):
    epoch_count = range(1,len(train_loss_plot)+1)
    plt.plot(epoch_count,  train_loss_plot, 'r--')
    plt.plot(epoch_count, valid_loss_plot, 'b-')
    plt.legend(['Training Loss', 'Valid Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('model_files/images/Loss_graph.png')
    plt.show()
    