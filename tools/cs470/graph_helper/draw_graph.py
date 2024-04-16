import os
import re
from tools.cs470.graph_helper.graphs import *
from tools.cs470.anaylsis_helper.formatter import *

def graph_path(i, output_dir, config_entry, title):
    formatted_title = str(i) + "_" + re.sub('[^0-9a-zA-Z]+', '_', title).lower()
    os.makedirs(f"{output_dir}/graphs/{formatted_title}", exist_ok=True)
    return f"{output_dir}/graphs/{formatted_title}/" + file_name(config_entry, name = re.sub('[^0-9a-zA-Z]+', '_', title).lower(), posfix=".jpg")

def graph_title(config_entry, title):
    return title + "\n" + formatting_config_entry(config_entry)

def draw_graph(output_dir, config_entry, small_total, medium_total, large_total, image_ids, exit_stages, bbox_size_1s, bbox_size_2s, bbox_ratios):
    x_title = "Early Exit Stages"
    index = config_entry['index']
    
    graph_number = 1
    color_index = 4

    ## Exit Stage Pie Chart
    exp_title = "Exit Stage Pie Chart"
    path = graph_path(graph_number, output_dir, config_entry, exp_title)
    title = graph_title(config_entry, exp_title)
    data = [0, 0, 0, 0]
    for i in range(len(image_ids)):
        data[exit_stages[index][i] - 1] += 1
    color = [color_list[(color_index + i) % len(color_list)] for i in range(4)]
    draw_pie_chart(data, color, title, path)

    graph_number += 1
    color_index += 4

    ## Bbox size per Exit Stage
    exp_title = "Bbox Size per Exit Stage"
    path = graph_path(graph_number, output_dir, config_entry, exp_title)
    title = graph_title(config_entry, exp_title)
    y_title = "Bbox Size(pixel^2)"
    x = exit_stages[index]
    y = [i * 1 for i in bbox_size_1s]
    s = [80 for i in x]
    color = [color_list[(color_index + i - 1) % len(color_list)] for i in x]
    draw_scatter_graph_entry(x, y, s, color, [1,2,3,4], title, x_title, y_title, path, '{:,.0f}')

    graph_number += 1
    color_index += 4

    ## Bbox ratio per Exit Stage
    exp_title = "Bbox Ratio per Exit Stage"
    path = graph_path(graph_number, output_dir, config_entry, exp_title)
    title = graph_title(config_entry, exp_title)
    y_title = "Bbox Ratio(%)"
    x = exit_stages[index]
    y = [i * 100 for i in bbox_ratios]
    s = [80 for i in x]
    color = [color_list[(color_index + i - 1) % len(color_list)] for i in x]
    draw_scatter_graph_entry(x, y, s, color, [1,2,3,4], title, x_title, y_title, path, '{:,.0f}%', yticks=[i * 10 for i in range(11)])

    graph_number += 1
    color_index += 4

    ## mAP size per Exit Stage
    exp_title = "mAP Size per Exit Stage"
    path = graph_path(graph_number, output_dir, config_entry, exp_title)
    title = graph_title(config_entry, exp_title)
    y_title = "mAP Size"
    x = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
    y = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
    s = []

    for x_index in [1, 2, 3, 4]:
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
                if (bbox_size_2s[i] == target and exit_stages[index][i] == x_index):
                    total += 1
            if (divide == 0):
                s.append(0)
            else:
                s.append(100000.0 * (total * 1.0 / divide))
    color = [color_list[(color_index + i - 1) % len(color_list)] for i in y]
    draw_scatter_graph_entry(x, y, s, color, [1,2,3,4], title, x_title, y_title, path, '{:,.0f}', yticks=[1,2,3], yticklabels=['small', 'medium', 'large'])

    graph_number += 1
    color_index += 4

    ## Bbox size per Exit stage Side-by-Side Box Plot
    exp_title = "Bbox size per Exit stage Side-by-Side Box Plot"
    path = graph_path(graph_number, output_dir, config_entry, exp_title)
    title = graph_title(config_entry, exp_title)
    y_title = "Bbox Size(pixel^2)"

    data = [[],[],[],[]]

    for i in range(len(image_ids)):
        exit_stage = exit_stages[index][i]
        data[exit_stage - 1].append(bbox_size_1s[i])

    color = [color_list[(color_index + i) % len(color_list)] for i in range(4)]
    draw_side_by_side_box_plot(data, color, [1,2,3,4], title, x_title, y_title, path, '{:,.0f}')

    graph_number += 1
    color_index += 4

    ## Bbox size per Exit stage Side-by-Side Violin Plot
    exp_title = "Bbox size per Exit stage Side-by-Side Violin Plot"
    path = graph_path(graph_number, output_dir, config_entry, exp_title)
    title = graph_title(config_entry, exp_title)
    y_title = "Bbox Size(pixel^2)"

    data = [[],[],[],[]]

    for i in range(len(image_ids)):
        exit_stage = exit_stages[index][i]
        data[exit_stage - 1].append(bbox_size_1s[i])

    color = [color_list[(color_index + i) % len(color_list)] for i in range(4)]
    draw_side_by_side_violin_plot(data, color, [1,2,3,4], title, x_title, y_title, path, '{:,.0f}')

    graph_number += 1
    color_index += 4

    ## Bbox ratio per Exit stage Side-by-Side Box Plot
    exp_title = "Bbox ratio per Exit stage Side-by-Side Box Plot"
    path = graph_path(graph_number, output_dir, config_entry, exp_title)
    title = graph_title(config_entry, exp_title)
    y_title = "Bbox Ratio(%)"

    data = [[],[],[],[]]

    for i in range(len(image_ids)):
        exit_stage = exit_stages[index][i]
        data[exit_stage - 1].append(bbox_ratios[i] * 100)

    color = [color_list[(color_index + i) % len(color_list)] for i in range(4)]
    draw_side_by_side_box_plot(data, color, [1,2,3,4], title, x_title, y_title, path, '{:,.0f}%', yticks=[i * 10 for i in range(11)])

    graph_number += 1
    color_index += 4

    ## Bbox ratio per Exit stage Side-by-Side Violin Plot
    exp_title = "Bbox ratio per Exit stage Side-by-Side Violin Plot"
    path = graph_path(graph_number, output_dir, config_entry, exp_title)
    title = graph_title(config_entry, exp_title)
    y_title = "Bbox Ratio(%)"

    data = [[],[],[],[]]

    for i in range(len(image_ids)):
        exit_stage = exit_stages[index][i]
        data[exit_stage - 1].append(bbox_ratios[i] * 100)

    color = [color_list[(color_index + i) % len(color_list)] for i in range(4)]
    draw_side_by_side_violin_plot(data, color, [1,2,3,4], title, x_title, y_title, path, '{:,.0f}%', yticks=[i * 10 for i in range(11)])

    graph_number += 1
    color_index += 4