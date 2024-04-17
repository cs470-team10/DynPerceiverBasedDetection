import os
import re
from tools.cs470.graph_helper.graphs import *
from tools.cs470.anaylsis_helper.formatter import *

def graph_path(i, output_dir, config_entry, title):
    formatted_title = re.sub('[^0-9a-zA-Z]+', '_', title).lower()
    if config_entry is not None:
        formatted_title = str(i) + "_" + formatted_title
        os.makedirs(f"{output_dir}/graphs/{formatted_title}", exist_ok=True)
        return f"{output_dir}/graphs/{formatted_title}/" + file_name(config_entry, name = re.sub('[^0-9a-zA-Z]+', '_', title).lower(), posfix=".jpg")
    else:
        return f"{output_dir}/graphs/{formatted_title}.jpg"
        

def graph_title(config_entry, title):
    return title + (("\n" + formatting_config_entry(config_entry)) if config_entry is not None else "")

def draw_graph(output_dir, config_entry, small_total, medium_total, large_total, image_ids, exit_stages, bbox_size_1s, bbox_size_2s, bbox_ratios):
    x_title = "Early Exit Stages"
    exit_stages = exit_stages[config_entry['index']]
    xtick_labels = ['Large', 'Medium', 'Small']
    
    ## Bbox size Histogram
    exp_title = "Bbox Size Histogram"
    path = graph_path(0, output_dir, None, exp_title)
    title = graph_title(None, exp_title)
    if (not os.path.exists(path)):
        draw_histogram(bbox_size_1s, 100, bbox_size_distribution_color, title, 'Bbox Size(pixel^2)', 'Number of Images', path)

    ## Bbox ratio Histogram
    exp_title = "Bbox Ratio Histogram"
    path = graph_path(0, output_dir, None, exp_title)
    title = graph_title(None, exp_title)
    if (not os.path.exists(path)):
        draw_histogram([i * 100 for i in bbox_ratios], 100, bbox_ratio_distribution_color, title, 'Bbox Ratio(%)', 'Number of Images', path, label_format='{:,.0f}%', xticks=[i * 10 for i in range(11)])

    graph_number = 1

    ## Exit Stage Pie Chart
    exp_title = "Exit Stage Pie Chart"
    path = graph_path(graph_number, output_dir, config_entry, exp_title)
    title = graph_title(config_entry, exp_title)
    data = [0 for _ in xtick_labels]
    for i in range(len(image_ids)):
        data[exit_stages[i] - 1] += 1
    color = exit_stage_colors
    draw_pie_chart(data, color, title, path)

    graph_number += 1

    ## mAP size per Exit Stage Bar Graph
    exp_title = "mAP Size per Exit Stage Bar Graph"
    path = graph_path(graph_number, output_dir, config_entry, exp_title)
    title = graph_title(config_entry, exp_title)
    y_title = "Percentage(%)"
    labels = ['Large', 'Medium', 'Small']
    data = []
    color = [mAP_size_colors[len(mAP_size_colors) - 1 - i] for i in range(len(mAP_size_colors))]
    for label in labels:
        entry = [0 for _ in xtick_labels]
        if (label == 'Small'):
            divide = small_total
            target = 'small'
        elif (label == 'Medium'):
            divide = medium_total
            target = 'medium'
        else:
            divide = large_total
            target = 'large'
        for j in range(len(image_ids)):
            if (bbox_size_2s[j] == target):
                entry[exit_stages[j] - 1] += 1
        data.append([(j * 100.0 / divide if divide != 0 else 0) for j in entry])

    draw_bar_graph(data, labels, color, title, x_title, y_title, path, label_format='{:,.0f}%', yticks=[i * 10 for i in range(11)])

    graph_number += 1

    ## Exit Stage per mAP Size Bar Graph
    exp_title = "Exit Stage per mAP Size Bar Grpah"
    path = graph_path(graph_number, output_dir, config_entry, exp_title)
    title = graph_title(config_entry, exp_title)
    y_title = "Percentage(%)"
    labels = ['Exit 1', 'Exit 2', 'Exit 3']
    bar_width = 0.25
    data = []
    color = exit_stage_colors
    for label in labels:
        entry = []
        if (label == 'Exit 1'):
            exit_stage = 1
        elif (label == 'Exit 2'):
            exit_stage = 2
        else:
            exit_stage = 3
        for xtick_label in xtick_labels:
            total = 0
            if (xtick_label == 'Small'):
                divide = small_total
                target = 'small'
            elif (xtick_label == 'Medium'):
                divide = medium_total
                target = 'medium'
            else:
                divide = large_total
                target = 'large'
            for j in range(len(image_ids)):
                if (bbox_size_2s[j] == target and exit_stages[j] == exit_stage):
                    total += 1
            entry.append(total * 100.0 / divide if divide != 0 else 0)
        data.append(entry)
    draw_bar_graph(data, labels, color, title, "Image Size", y_title, path, label_format='{:,.0f}%', yticks=[i * 10 for i in range(11)], bar_width=bar_width, xtick_labels=xtick_labels)

    graph_number += 1

    ## mAP size per Exit Stage Scatter
    exp_title = "mAP Size per Exit Stage Scatter"
    path = graph_path(graph_number, output_dir, config_entry, exp_title)
    title = graph_title(config_entry, exp_title)
    y_title = "mAP Size"
    x = [1, 1, 1, 2, 2, 2, 3, 3, 3]
    y = [1, 2, 3, 1, 2, 3, 1, 2, 3]
    s = []

    for x_index in [1, 2, 3]:
        for y_index in [1, 2, 3]:
            total = 0
            if (y_index == 1):
                target = 'small'
                divide = small_total
            elif (y_index == 2):
                target = 'medium'
                divide = medium_total
            elif (y_index == 3):
                target = 'large'
                divide = large_total
            for i in range(len(image_ids)):
                if (bbox_size_2s[i] == target and exit_stages[i] == x_index):
                    total += 1
            if (divide == 0):
                s.append(0)
            else:
                s.append(100000.0 * (total * 1.0 / divide))
    color = [mAP_size_colors[i - 1] for i in y]
    draw_scatter_graph_entry(x, y, s, color, [1,2,3], title, x_title, y_title, path, '{:,.0f}', yticks=[1,2,3], yticklabels=['Small', 'Medium', 'Large'])

    graph_number += 1

    ## Bbox size per Exit Stage Box Plot
    exp_title = "Bbox size per Exit Stage Box Plot"
    path = graph_path(graph_number, output_dir, config_entry, exp_title)
    title = graph_title(config_entry, exp_title)
    y_title = "Bbox Size(pixel^2)"

    data = [[],[],[]]

    for i in range(len(image_ids)):
        exit_stage = exit_stages[i]
        data[exit_stage - 1].append(bbox_size_1s[i])

    color = exit_stage_colors
    draw_side_by_side_box_plot(data, color, [1,2,3], title, x_title, y_title, path, '{:,.0f}')

    graph_number += 1

    ## Bbox size per Exit Stage Violin Plot
    exp_title = "Bbox size per Exit Stage Violin Plot"
    path = graph_path(graph_number, output_dir, config_entry, exp_title)
    title = graph_title(config_entry, exp_title)
    y_title = "Bbox Size(pixel^2)"

    data = [[], [], []]

    for i in range(len(image_ids)):
        exit_stage = exit_stages[i]
        data[exit_stage - 1].append(bbox_size_1s[i])

    color = exit_stage_colors
    draw_side_by_side_violin_plot(data, color, [1, 2, 3], title, x_title, y_title, path, '{:,.0f}')

    graph_number += 1

    ## Bbox ratio per Exit Stage Box Plot
    exp_title = "Bbox ratio per Exit Stage Box Plot"
    path = graph_path(graph_number, output_dir, config_entry, exp_title)
    title = graph_title(config_entry, exp_title)
    y_title = "Bbox Ratio(%)"

    data = [[], [], []]

    for i in range(len(image_ids)):
        exit_stage = exit_stages[i]
        data[exit_stage - 1].append(bbox_ratios[i] * 100)

    color = exit_stage_colors
    draw_side_by_side_box_plot(data, color, [1, 2, 3], title, x_title, y_title, path, '{:,.0f}%', yticks=[i * 10 for i in range(11)])

    graph_number += 1

    ## Bbox ratio per Exit Stage Violin Plot
    exp_title = "Bbox ratio per Exit Stage Violin Plot"
    path = graph_path(graph_number, output_dir, config_entry, exp_title)
    title = graph_title(config_entry, exp_title)
    y_title = "Bbox Ratio(%)"

    data = [[], [], []]

    for i in range(len(image_ids)):
        exit_stage = exit_stages[i]
        data[exit_stage - 1].append(bbox_ratios[i] * 100)

    color = exit_stage_colors
    draw_side_by_side_violin_plot(data, color, [1, 2, 3], title, x_title, y_title, path, '{:,.0f}%', yticks=[i * 10 for i in range(11)])

    graph_number += 1