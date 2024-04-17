import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

xticklabels = ['Exit 1', 'Exit 2', 'Exit 3', 'Exit 4']
color_list = ["pink", "red", "teal", "blue", "orange", "yellow", "magenta","green","aqua"]

def draw_side_by_side_box_plot(data, colors, xticks, title, x_title, y_title, output_dir, label_format = '{:,.0f}', yticks = []):
    fig, ax = plt.subplots(figsize=(15, 10))
    box_dict = ax.boxplot(data, patch_artist=True,  showmeans=True)
    for item in ['boxes', 'fliers', 'medians', 'means']:
        for sub_item, color in zip(box_dict[item], colors):
            plt.setp(sub_item, color=color)
    for item in ['whiskers', 'caps']:
        for sub_items, color in zip(zip(box_dict[item][::2],box_dict[item][1::2]), colors):
            plt.setp(sub_items, color=color)

    plt.xlabel(x_title, fontdict = {'fontsize' : 15})
    plt.xticks(xticks, fontsize = 15)
    ax.set_xticklabels(xticklabels)
    if (len(yticks) > 0):
        plt.yticks(yticks, fontsize = 15)
    else:
        plt.yticks(fontsize = 15)
    plt.ylabel(y_title, fontdict = {'fontsize' : 15})
    plt.title(title, weight = 'bold', fontdict = {'fontsize' : 20})
    ticks_loc = plt.gca().get_yticks().tolist()
    plt.gca().yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    plt.gca().set_yticklabels([label_format.format(x) for x in ticks_loc])
    plt.tight_layout()
    plt.savefig(output_dir)
    plt.close()

def draw_side_by_side_violin_plot(data, colors, xticks, title, x_title, y_title, output_dir, label_format = '{:,.0f}', yticks = []):
    fig, ax = plt.subplots(figsize=(15, 10))
    violin = ax.violinplot(data, showmeans=True, quantiles=[[0.25, 0.75] for _ in range(4)])
    for i in range(4):
        violin['bodies'][i].set_facecolor(colors[i])
    violin['cbars'].set_edgecolor('gray')
    violin['cmaxes'].set_edgecolor('gray')
    violin['cmins'].set_edgecolor('gray')
    violin['cmeans'].set_edgecolor('black')

    plt.xlabel(x_title, fontdict = {'fontsize' : 15})
    plt.xticks(xticks, fontsize = 15)
    ax.set_xticklabels(xticklabels)
    if (len(yticks) > 0):
        plt.yticks(yticks, fontsize = 15)
    else:
        plt.yticks(fontsize = 15)
    plt.ylabel(y_title, fontdict = {'fontsize' : 15})
    plt.title(title, weight = 'bold', fontdict = {'fontsize' : 20})
    ticks_loc = plt.gca().get_yticks().tolist()
    plt.gca().yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    plt.gca().set_yticklabels([label_format.format(x) for x in ticks_loc])
    plt.tight_layout()
    plt.savefig(output_dir)
    plt.close()

# def draw_scatter_graph_entry(x, y, s, color, xticks, title, x_title, y_title, output_dir, label_format = '{:,.0f}', yticks = [], yticklabels = []):
#     plt.figure(figsize=(15, 10))
#     plt.scatter(x, y, s=s, color = color)

#     plt.xlabel(x_title, fontdict = {'fontsize' : 15})
#     plt.xticks(xticks, fontsize = 15)
#     plt.gca().set_xticklabels(xticklabels)
#     if (len(yticks) > 0):
#         plt.yticks(yticks, fontsize = 15)
#     else:
#         plt.yticks(fontsize = 15)
#     plt.ylabel(y_title, fontdict = {'fontsize' : 15})
#     plt.title(title, weight = 'bold', fontdict = {'fontsize' : 20})
#     ticks_loc = plt.gca().get_yticks().tolist()
#     plt.gca().yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
#     plt.gca().set_yticklabels([label_format.format(x) for x in ticks_loc] if len(yticklabels) == 0 else yticklabels)
#     plt.tight_layout()
#     plt.savefig(output_dir)
#     plt.close()

def draw_pie_chart(data, colors, title, output_dir, label_format = '%.2f%%'):
    plt.figure(figsize=(15, 10))
    plt.pie(data, colors = colors, labels=xticklabels, autopct=label_format, startangle=90, counterclock=False, textprops={'fontsize': 15})
    plt.title(title, weight = 'bold', fontdict = {'fontsize' : 20})
    plt.tight_layout()
    plt.savefig(output_dir)
    plt.close()

def draw_bar_graph(data, labels, colors, title, x_title, y_title, output_dir, label_format = '{:,.0f}', yticks = [], bar_width = 0.25, xtick_labels = []):
    fig, ax = plt.subplots(figsize=(15, 10))
    xtick_labels = xtick_labels if len(xtick_labels) > 0 else xticklabels
    index = [i for i in range(len(xtick_labels))]

    for i in range(len(labels)):
        plt.bar([j + i * bar_width for j in index], data[i], bar_width, alpha = 0.4, color = colors[i], label= labels[i])

    plt.xlabel(x_title, fontdict = {'fontsize' : 15})
    plt.xticks([i + bar_width for i in range(len(xtick_labels))], fontsize = 15)
    ax.set_xticklabels(xtick_labels)
    if (len(yticks) > 0):
        plt.yticks(yticks, fontsize = 15)
    else:
        plt.yticks(fontsize = 15)
    plt.ylabel(y_title, fontdict = {'fontsize' : 15})
    plt.title(title, weight = 'bold', fontdict = {'fontsize' : 20})
    ticks_loc = plt.gca().get_yticks().tolist()
    plt.gca().yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    plt.gca().set_yticklabels([label_format.format(x) for x in ticks_loc])
    plt.legend(fontsize="15")
    plt.tight_layout()
    plt.savefig(output_dir)
    plt.close()

def draw_histogram(data, bins, color, title, x_title, y_title, output_dir, label_format = '{:,.0f}', xticks = [], xticklabels = []):
    plt.figure(figsize=(15, 10))
    plt.hist(data, color = color, bins = bins, alpha = 0.7)
    plt.xlabel(x_title, fontdict = {'fontsize' : 15})
    if (len(xticks) > 0):
        plt.xticks(xticks, fontsize = 15)
    else:
        plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.ylabel(y_title, fontdict = {'fontsize' : 15})
    plt.title(title, weight = 'bold', fontdict = {'fontsize' : 20})
    ticks_loc = plt.gca().get_xticks().tolist()
    plt.gca().xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    plt.gca().set_xticklabels([label_format.format(x) for x in ticks_loc] if len(xticklabels) == 0 else xticklabels)
    ticks_loc = plt.gca().get_yticks().tolist()
    plt.gca().yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in ticks_loc])
    plt.tight_layout()
    plt.savefig(output_dir)
    plt.close()