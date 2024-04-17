import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

xticklabels = ["Exit 1", "Exit 2", "Exit 3", "Exit 4"]
color_list = ["pink", "red", "teal", "blue", "orange", "yellow", "magenta","green","aqua"]

bbox_ratio_distribution_color = "red"
bbox_size_distribution_color = "blue"
exit_stage_colors = ["orange", "yellow", "magenta", "aqua"]
mAP_size_colors = ["pink", "teal", "green"]
alpha = 0.6

figsize = (15, 10)
title_fontsize = 20
entry_fontsize = 15

def draw_side_by_side_box_plot(data, colors, xticks, title, x_title, y_title, output_dir, label_format = "{:,.0f}", yticks = []):
    fig, ax = plt.subplots(figsize=figsize)
    box_dict = ax.boxplot(data, patch_artist=True, showmeans=False, whiskerprops = dict(linestyle='--'))
    for i in range(len(xticks)):
        box_dict["boxes"][i].set_edgecolor("black")
        box_dict["boxes"][i].set_facecolor(colors[i])
        box_dict["boxes"][i].set_alpha(alpha)
        box_dict["boxes"][i]
    for item in ["whiskers"]:
        for sub_items in zip(zip(box_dict[item][::2],box_dict[item][1::2])):
            plt.setp(sub_items, color="black")
    for item in ["caps"]:
        for sub_items in zip(zip(box_dict[item][::2],box_dict[item][1::2])):
            plt.setp(sub_items, color="gray")
    for median in box_dict['medians']:
        median.set_color('black')

    plt.xlabel(x_title, fontdict = {"fontsize" : entry_fontsize})
    plt.xticks(xticks, fontsize = entry_fontsize)
    ax.set_xticklabels(xticklabels)
    if (len(yticks) > 0):
        plt.yticks(yticks, fontsize = entry_fontsize)
    else:
        plt.yticks(fontsize = entry_fontsize)
    plt.ylabel(y_title, fontdict = {"fontsize" : entry_fontsize})
    plt.title(title, weight = "bold", fontdict = {"fontsize" : title_fontsize})
    ticks_loc = plt.gca().get_yticks().tolist()
    plt.gca().yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    plt.gca().set_yticklabels([label_format.format(x) for x in ticks_loc])
    plt.tight_layout()
    plt.savefig(output_dir)
    plt.close()

def draw_side_by_side_violin_plot(data, colors, xticks, title, x_title, y_title, output_dir, label_format = "{:,.0f}", yticks = []):
    fig, ax = plt.subplots(figsize=figsize)
    violin = ax.violinplot(data, showmeans=True, quantiles=[[0.25, 0.75] for _ in range(len(xticks))])
    for i in range(len(xticks)):
        violin["bodies"][i].set_facecolor(colors[i])
        violin["bodies"][i].set_alpha(alpha)
    violin["cbars"].set_edgecolor("gray")
    violin["cmaxes"].set_edgecolor("gray")
    violin["cmins"].set_edgecolor("gray")
    violin["cmeans"].set_edgecolor("black")

    plt.xlabel(x_title, fontdict = {"fontsize" : entry_fontsize})
    plt.xticks(xticks, fontsize = entry_fontsize)
    ax.set_xticklabels(xticklabels)
    if (len(yticks) > 0):
        plt.yticks(yticks, fontsize = entry_fontsize)
    else:
        plt.yticks(fontsize = entry_fontsize)
    plt.ylabel(y_title, fontdict = {"fontsize" : entry_fontsize})
    plt.title(title, weight = "bold", fontdict = {"fontsize" : title_fontsize})
    ticks_loc = plt.gca().get_yticks().tolist()
    plt.gca().yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    plt.gca().set_yticklabels([label_format.format(x) for x in ticks_loc])
    plt.tight_layout()
    plt.savefig(output_dir)
    plt.close()

# def draw_scatter_graph_entry(x, y, s, color, xticks, title, x_title, y_title, output_dir, label_format = "{:,.0f}", yticks = [], yticklabels = []):
#     plt.figure(figsize=figsize)
#     plt.scatter(x, y, s=s, color = color)

#     plt.xlabel(x_title, fontdict = {"fontsize" : entry_fontsize})
#     plt.xticks(xticks, fontsize = entry_fontsize)
#     plt.gca().set_xticklabels(xticklabels)
#     if (len(yticks) > 0):
#         plt.yticks(yticks, fontsize = entry_fontsize)
#     else:
#         plt.yticks(fontsize = entry_fontsize)
#     plt.ylabel(y_title, fontdict = {"fontsize" : entry_fontsize})
#     plt.title(title, weight = "bold", fontdict = {"fontsize" : title_fontsize})
#     ticks_loc = plt.gca().get_yticks().tolist()
#     plt.gca().yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
#     plt.gca().set_yticklabels([label_format.format(x) for x in ticks_loc] if len(yticklabels) == 0 else yticklabels)
#     plt.tight_layout()
#     plt.savefig(output_dir)
#     plt.close()

def draw_pie_chart(data, colors, title, output_dir, label_format = "%.2f%%"):
    plt.figure(figsize=figsize)
    n = plt.pie(data, colors = colors, labels=xticklabels, autopct=label_format, startangle=90, counterclock=False, textprops={"fontsize": entry_fontsize})
    for i in range(len(n[0])):
        n[0][i].set_alpha(alpha)
    plt.title(title, weight = "bold", fontdict = {"fontsize" : title_fontsize})
    plt.tight_layout()
    plt.savefig(output_dir)
    plt.close()

def draw_bar_graph(data, labels, colors, title, x_title, y_title, output_dir, label_format = "{:,.0f}", yticks = [], bar_width = 0.25, xtick_labels = []):
    fig, ax = plt.subplots(figsize=figsize)
    xtick_labels = xtick_labels if len(xtick_labels) > 0 else xticklabels
    index = [i for i in range(len(xtick_labels))]

    for i in range(len(labels)):
        plt.bar([j + i * bar_width for j in index], data[i], bar_width, alpha = alpha, color = colors[i], label= labels[i])

    plt.xlabel(x_title, fontdict = {"fontsize" : entry_fontsize})
    plt.xticks([i + bar_width for i in range(len(xtick_labels))], fontsize = entry_fontsize)
    ax.set_xticklabels(xtick_labels)
    if (len(yticks) > 0):
        plt.yticks(yticks, fontsize = entry_fontsize)
    else:
        plt.yticks(fontsize = entry_fontsize)
    plt.ylabel(y_title, fontdict = {"fontsize" : entry_fontsize})
    plt.title(title, weight = "bold", fontdict = {"fontsize" : title_fontsize})
    ticks_loc = plt.gca().get_yticks().tolist()
    plt.gca().yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    plt.gca().set_yticklabels([label_format.format(x) for x in ticks_loc])
    plt.legend(fontsize=f"{entry_fontsize}")
    plt.tight_layout()
    plt.savefig(output_dir)
    plt.close()

def draw_histogram(data, bins, color, title, x_title, y_title, output_dir, label_format = "{:,.0f}", xticks = [], xticklabels = []):
    plt.figure(figsize=figsize)
    plt.hist(data, color = color, bins = bins, alpha = alpha)
    plt.xlabel(x_title, fontdict = {"fontsize" : entry_fontsize})
    if (len(xticks) > 0):
        plt.xticks(xticks, fontsize = entry_fontsize)
    else:
        plt.xticks(fontsize = entry_fontsize)
    plt.yticks(fontsize = entry_fontsize)
    plt.ylabel(y_title, fontdict = {"fontsize" : entry_fontsize})
    plt.title(title, weight = "bold", fontdict = {"fontsize" : title_fontsize})
    ticks_loc = plt.gca().get_xticks().tolist()
    plt.gca().xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    plt.gca().set_xticklabels([label_format.format(x) for x in ticks_loc] if len(xticklabels) == 0 else xticklabels)
    ticks_loc = plt.gca().get_yticks().tolist()
    plt.gca().yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    plt.gca().set_yticklabels(["{:,.0f}".format(x) for x in ticks_loc])
    plt.tight_layout()
    plt.savefig(output_dir)
    plt.close()