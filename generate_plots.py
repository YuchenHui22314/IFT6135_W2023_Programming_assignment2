import os
import json
import matplotlib.pyplot as plt
import numpy as np

def get_logger(model):
  logger = dict()
  logger['train_time'] = [0]
  logger['eval_time'] = [0]
  logger['train_losses'] = []
  logger['eval_accs'] = []
  logger['eval_losses'] = []
  logger['parameters'] = sum([p.numel() for p in model.back_bone.parameters() if p.requires_grad])
  return logger


def plots(experiment_index):
    path = f"F:\\yuchenxi\\UDEM\\diro\\IFT6135\\assignment2\\IFT6135_W2023_Programming_assignment2\\log\\{experiment_index}\\args.json"
    ## load args.json
    with open(path, 'r') as f:
        args = json.load(f)
    
    #plot learning curves (train loss and validation loss/accuracy) over training iterations
    # "tran_losses" and "eval_accs" are lists of length iterations
    x = list(range(len(args['train_losses'])))
    if experiment_index != 6:
        x = [i*5 for i in x]
    else:
        x = [i*703 for i in x]

    train_losses = args['train_losses']
    eval_accs = [v/100 for v in args['eval_accs']]

    fig, ax = plt.subplots()
    ax.set_xlabel("iterations")
    ax.set_ylabel("loss/accuracy")

    ax.plot(x,train_losses, label = "training loss")
    ax.plot(x,eval_accs, label = "validation accuracy(0-1)")
    ax.set_title(f"learning curves for experiment {experiment_index}")

    ax.legend()
    fig.savefig(os.path.join(f"F:\\yuchenxi\\UDEM\\diro\\IFT6135\\assignment2\\IFT6135_W2023_Programming_assignment2\\log", f'experiment{experiment_index}.png'))


for i in range(1,7):
    plots(i)
    # fig, axs = plt.subplots(2, 2,figsize=(14,12))  # a figure with a 2x2 grid of Axes
    # # axs = [[obj,obj],[obj,obj]]



    # ax_top_left_corner = axs[0][0]
    # ax_top_right_corner = axs[0][1]
    # ax_bottom_left_corner = axs[1][0]
    # ax_bottom_right_corner = axs[1][1]

    # # set attributs of each axes (ax)
    # ## ---sinx
    # x = np.linspace(-3,3,100)
    # y = np.sin(x)
    # ax_top_left_corner.plot(x,y,label = "sinx")
    # ##
    # ax_top_left_corner.set_xlabel("lol")
    # ax_top_left_corner.set_ylabel("lol ya fan")
    # ax_top_left_corner.set_title("qibushi")
    # #------ 左下角哈
    # ax_bottom_left_corner.set_title("qibushi")
    # #-------
    # noise = np.random.randn(100)*0.1
    # ax_top_left_corner.plot(x,y+noise, label = "sinx+noise")
    # ax_top_right_corner.plot(x,y+noise, label = "sinx+noise")

    # ##---legend(显示label的内容）
    # ax_top_left_corner.legend()

    ##grid
    #ax_top_right_corner.grid()

# !!! eliminate overlap （好像要放到最后。。
# plt.tight_layout()








def generate_plots(list_of_dirs, legend_names, save_path):
    """ Generate plots according to log 
    :param list_of_dirs: List of paths to log directories
    :param legend_names: List of legend names
    :param save_path: Path to save the figs
    """
    assert len(list_of_dirs) == len(legend_names), "Names and log directories must have same length"
    data = {}
    for logdir, name in zip(list_of_dirs, legend_names):
        json_path = os.path.join(logdir, 'results.json')
        assert os.path.exists(os.path.join(logdir, 'results.json')), f"No json file in {logdir}"
        with open(json_path, 'r') as f:
            data[name] = json.load(f)
    
    for yaxis in ['train_accs', 'valid_accs', 'train_losses', 'valid_losses']:
        fig, ax = plt.subplots()
        for name in data:
            ax.plot(data[name][yaxis], label=name)
        ax.legend()
        ax.set_xlabel('epochs')
        ax.set_ylabel(yaxis.replace('_', ' '))
        fig.savefig(os.path.join(save_path, f'{yaxis}.png'))