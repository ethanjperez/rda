import json
import matplotlib.pyplot as plt
import os
from termcolor import colored

plot_title_base = 'CLEVR: '
plot_legend_title = ''
arg_feat_name = None
files_resumed = {
}
files_desc = {
    'film.pt.json': 'Best Model',
}
files = [
    'film.pt.json'
]

legend = [filename[:-len('.pt.json')] for filename in files]
legend[0] = 'Best'

exp_base_dir = os.path.join(os.environ('BASE_DIR'), 'exp/')
img_base_dir = 'img/plots/'

baseline_description = 'Baseline'
ours_description = 'New Method'
baseline_file = files[0]
baseline_legend = legend[0]
with open(exp_base_dir + baseline_file) as json_data:
    b = json.load(json_data)

plot_title = plot_title_base + 'Validation Accuracy'
plt.title(plot_title)
print(colored(plot_title, 'magenta'))
for i in range(len(files)):
    with open(exp_base_dir + files[i]) as json_data:
        d = json.load(json_data)
        print(colored(legend[i], 'cyan'))
        print([round(acc, 4) for acc in d['val_accs']])
        plt.plot(d['val_accs_ts'], d['val_accs'], label=legend[i])
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
plt.legend(title=plot_legend_title)
plt.savefig(img_base_dir + plot_title)
plt.show()

if arg_feat_name is not None:
    x = []
    y = []
    plot_title = plot_title_base + 'Best Validation Accuracy'
    plt.title(plot_title)
    print(plot_title)
    for i in range(len(files)):
        with open(exp_base_dir + files[i]) as json_data:
            d = json.load(json_data)
            x.append(d['args'][arg_feat_name])
            y.append(d['best_val_acc'])
    print(x)
    print(y)
    plt.plot(x, y)
    plt.xlabel(plot_legend_title)
    plt.ylabel('Accuracy')
    plt.savefig(img_base_dir + plot_title)
    plt.show()

for i in range(len(files)):
    with open(exp_base_dir + files[i]) as json_data:
        d = json.load(json_data)
        plot_title = plot_title_base + 'Training vs Validation - ' + legend[i] + ' ' + plot_legend_title
        plt.title(plot_title)
        print(colored(plot_title, 'magenta'))

        # Calculate original run stats
        train_accs = [round(acc, 4) for acc in d['train_accs']]
        val_accs = [round(acc, 4) for acc in d['val_accs']]
        overfit = [round(d['train_accs'][j] - d['val_accs'][j], 4) for j in range(len(d['val_accs']))]

        # Calculate baseline run stats
        baseline_train_accs = [round(acc, 4) for acc in b['train_accs']]
        baseline_val_accs = [round(acc, 4) for acc in b['val_accs']]

        # Add in continuation run stats
        if files[i] in files_resumed:
            for filename in files_resumed[files[i]]:
                with open(exp_base_dir + filename) as json_data_r:
                    d_r = json.load(json_data_r)
                    train_accs += [round(acc, 4) for acc in d_r['train_accs']]
                    val_accs += [round(acc, 4) for acc in d_r['val_accs']]
                    overfit += [round(d_r['train_accs'][j] - d_r['val_accs'][j], 4)
                                  for j in range(len(d_r['val_accs']))]

        best_val_acc = max(val_accs)
        best_val_idx = val_accs.index(best_val_acc)
        best_train_acc = train_accs[best_val_idx]
        best_overfit = overfit[best_val_idx]

        print(colored('Best @ Epoch ' + str(best_val_idx + 1) + '/' + str(len(val_accs)), 'yellow'))
        print(colored('Train:  ', 'cyan'))
        print(colored(best_train_acc, 'red'))
        print(train_accs)
        print(colored('Val:    ', 'cyan'))
        print(colored(best_val_acc, 'red'))
        print(val_accs)
        print(colored('Overfit:', 'cyan'))
        print(colored(best_overfit, 'red'))
        print(overfit)
        print('')

        label_desc = ours_description
        if files[i] in files_desc:
            label_desc = files_desc[files[i]]

        plot_cutoff = 1000
        if i > 0:
            plt.plot(range(len(train_accs[:plot_cutoff])), train_accs[:plot_cutoff], label=label_desc + ': Train')
            plt.plot(range(len(val_accs[:plot_cutoff])), val_accs[:plot_cutoff], label=label_desc + ': Val')
        plt.plot(range(len(baseline_train_accs[:plot_cutoff])), baseline_train_accs[:plot_cutoff], label='Train' + baseline_description)
        plt.plot(range(len(baseline_val_accs[:plot_cutoff])), baseline_val_accs[:plot_cutoff], label='Val' + baseline_description)

        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')

    plt.savefig(img_base_dir + plot_title)
    plt.show()

# plt.legend()
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
#
# plt.savefig(img_base_dir + plot_title)
# plt.show()
