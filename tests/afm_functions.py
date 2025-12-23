import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd
import plotly.express as px
from PIL import Image, ImageDraw, ImageFont
import io

def import_data(file):
    numbers = pd.read_csv(file,delimiter=';')
    numbers = numbers.dropna(axis=1, how='all')
    data = np.array(numbers)
    data = data[2:,:]
    data_clean = {}
    counter = 1
    for i in range(np.shape(data)[1]):
        if i % 2 == 0:
            data_clean[f"position{counter}"] = data[:,i]
        else:
            data_clean[f"height{counter}"] = data[:,i]
            counter += 1
    for key in data_clean:
        data_clean[key] = [float(value) for value in data_clean[key]]
    return data_clean

def threshold_averaging(data, threshold):
    averaged_data = data.copy()
    keys = list(averaged_data.keys())
    for i in range(1, len(keys), 2):
        values = averaged_data[keys[i]]
        current_step = []
        steps = []
        for k in range(0,len(values)):
            if abs(values[k] - values[k - 1]) <= threshold:
                current_step.append(values[k])
            else:
                if current_step:
                    steps.append(current_step)
                current_step = [values[k]]
        if current_step:
            steps.append(current_step)
        averaged_values = []
        for step in steps:
            avg = float(np.mean(step))
            averaged_values.extend([avg] * len(step))
        averaged_data[keys[i]] = averaged_values
    return averaged_data

def plotly_graph(file):
    keys = list(file.keys())
    counter = 1
    records = []
    for i in range(0, len(keys), 2):
        x = file[keys[i]]
        y = file[keys[i+1]]
        profile = f"Profile {counter}"
        for xi, yi in zip(x, y):
            records.append({"Position": xi, "Height": yi, "Profile": profile})
        counter += 1
    df = pd.DataFrame(records)
    fig = px.line(df, x="Position", y="Height", color="Profile", markers=False)
    return fig, df

def graph(file, filename=" "):
    fig, ax = plt.subplots()
    keys = list(file.keys())
    counter = 1
    for i in range(0, len(keys), 2):
        x = [val * 1e9 for val in file[keys[i]]]
        y = [val * 1e9 for val in file[keys[i + 1]]]
        line, = ax.plot(x, y, label=f"Profile {counter}")
        # line_color = line.get_color()
        # for k in range(len(y)):
        #     if k == 1 or y[k - 1] != y[k - 2]:
        #         ax.scatter(x[k - 1], y[k - 1], color=line_color, s=7)
        counter += 1
    ax.legend()
    ax.set_title(f"{filename} AFM Profiles")
    ax.set_xlabel("Position (nm)")
    ax.set_ylabel("Height (nm)")
    return fig

def scatter_data(file, profile):
    x_data = []
    y_data = []
    x_key = f"position{profile}"
    y_key = f"height{profile}"
    x_data = list([val * 1e9 for val in file[x_key]])
    y_data = list([val * 1e9 for val in file[y_key]])
    clean_x = [x for x in x_data if not np.isnan(x)]
    clean_y = [y for y in y_data if not np.isnan(y)]
    return clean_x, clean_y

def select_nearest_points_with_sliders(x_data, y_data, result_dict, profile, name):
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)
    index1 = widgets.IntSlider(value=0, min=0, max=len(x_data)-1, description='Point 1')
    index2 = widgets.IntSlider(value=1, min=0, max=len(x_data)-1, description='Point 2')
    confirm_button = widgets.Button(description="Confirm Selection", button_style='success')
    output = widgets.Output()
    confirm_output = widgets.Output()
    state = {
        'fig': None,
        'height': None,
    }
    def plot_selected_points(change=None):
        with output:
            clear_output(wait=True)
            fig, ax = plt.subplots()
            ax.scatter(x_data, y_data, color='blue', s = 7)
            ax.plot(x_data, y_data, color='blue', label=f"Profile {profile}")
            i1, i2 = index1.value, index2.value
            x1, y1 = x_data[i1], y_data[i1]
            x2, y2 = x_data[i2], y_data[i2]
            height_diff = abs(y2 - y1)
            ax.scatter([x1, x2], [y1, y2], color='red', zorder=5)
            ax.axvline(x=x1, color='black', linestyle='--')
            ax.axvline(x=x2, color='black', linestyle='--')
            ax.set_xlabel("Position (nm)")
            ax.set_ylabel("Height (nm)")
            ax.set_title(f"{name} Profile {profile} Height")
            ax.legend()
            plt.show()
            state['fig'] = fig
            state['height'] = height_diff
    def on_confirm_clicked(b):
        with confirm_output:
            clear_output(wait=True)
            print("✅ Confirmed. Results stored.")
        # Store result in the provided dictionary
        result_dict[profile] = {
            'fig': state['fig'],
            'height': state['height']
        }
    index1.observe(plot_selected_points, names='value')
    index2.observe(plot_selected_points, names='value')
    confirm_button.on_click(on_confirm_clicked)
    display(widgets.HBox([index1, index2]))
    display(output)
    display(confirm_button)
    display(confirm_output)
    plot_selected_points()

def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return Image.open(buf)

def resize_keep_aspect(img, target_width=None, target_height=None):
    w, h = img.size
    if target_width and not target_height:
        scale = target_width / w
    elif target_height and not target_width:
        scale = target_height / h
    elif target_width and target_height:
        scale = min(target_width / w, target_height / h)
    else:
        return img
    new_size = (int(w * scale), int(h * scale))
    return img.resize(new_size, Image.LANCZOS)

def create_report(title, image_paths, main_fig, small_figs, notes):
    width = 1200
    background_color = "white"
    initial_height = 5000  # large enough; cropped later

    report = Image.new("RGBA", (width, initial_height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(report)

    try:
        font_path = "C:/Windows/Fonts/times.ttf"
        font = ImageFont.truetype(font_path, 40)
        med_font = ImageFont.truetype(font_path, 24)
        small_font = ImageFont.truetype(font_path, 18)
    except IOError:
        font = ImageFont.load_default()
        med_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    y = 20
    padding = 20

    # --- Title ---
    draw.text((width // 2 - draw.textlength(title, font=font) // 2, y), title, font=font, fill="black")
    y += 60

    # --- First row: first 2 PNG images ---
    img_width = (width - 3 * padding) // 2
    img_height = 300
    for i in range(2):
        img = Image.open(image_paths[i]).resize((img_width, img_height))
        report.paste(img, (padding + i * (img_width + padding), y))
    y += img_height + 20

    # --- Second row: third PNG + main matplotlib ---
    img3 = Image.open(image_paths[2]).resize((img_width, img_height))
    main_img = fig_to_image(main_fig)
    main_img = resize_keep_aspect(main_img, target_width=img_width, target_height=img_height)
    report.paste(img3, (padding, y))
    report.paste(main_img, (padding + img_width + padding, y))
    y += max(img_height, main_img.height) + 30

    # --- Small matplotlib figures in rows of 3 ---
    max_small_width = (width - 4 * padding) // 3
    max_small_height = 280
    col = 0
    current_row_max_height = 0
    x_start = padding
    for fig in small_figs:
        small_img = fig_to_image(fig)
        small_img = resize_keep_aspect(small_img, target_width=max_small_width, target_height=max_small_height)
        x = x_start + col * (max_small_width + padding)
        if col == 3:
            y += current_row_max_height + padding
            col = 0
            x = x_start
            current_row_max_height = 0
        report.paste(small_img, (x, y))
        current_row_max_height = max(current_row_max_height, small_img.height)
        col += 1
    y += current_row_max_height + 40

    # --- Notes section ---
    notes_box_top = y - 10
    notes_box_left = padding - 10
    notes_box_right = width - padding + 10
    draw.text((padding, y), "Notes:", font=med_font, fill="black")
    y += 40

    for note in notes:
        draw.text((padding + 20, y), f"- {note}", font=small_font, fill="black")
        y += 35  # slightly more spacing for clarity

    y += 10  # extra bottom padding for notes

    # --- Draw notes box with larger border ---
    draw.rectangle(
        [notes_box_left, notes_box_top, notes_box_right, y],
        outline="black", width=2
    )

    # --- Outer border ---
    draw.rectangle([0, 0, width - 1, y + 10], outline="black", width=3)

    # --- Crop and show ---
    report = report.crop((0, 0, width, y + 20))
    display(report)

def properties_graph(file, profile):
    fig, ax = plt.subplots()
    x_data = []
    y_data = []
    x_key = f"position{profile}"
    y_key = f"height{profile}"
    x_data = list([val * 1e9 for val in file[x_key]])
    y_data = list([val * 1e9 for val in file[y_key]])
    line, = ax.plot(x_data, y_data, label=f"Profile {profile}")
    line_color = line.get_color()
    for k in range(len(y_data)):
        if k == 0 or (k > 1 and y_data[k - 1] != y_data[k - 2]):
            scatter_x = x_data[k]
            scatter_y = y_data[k]
            ax.scatter(scatter_x, scatter_y, color=line_color, s=7)
    ax.legend()
    ax.set_xlabel("Position [nm]")
    ax.set_ylabel("Height [nm]")
    fig.tight_layout()
    clean_x = [x for x in x_data if not np.isnan(x)]
    clean_y = [y for y in y_data if not np.isnan(y)]
    return fig, ax, clean_x, clean_y

def select_nearest_points(fig, ax, x_data, y_data):
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)
    selected_points = []
    selected_heights = []
    def onclick(event):
        if event.inaxes:
            x_click, y_click = event.xdata, event.ydata
            distances = np.sqrt((x_data - x_click)**2 + (y_data - y_click)**2)
            index_min = np.argmin(distances)
            nearest_x = x_data[index_min]
            nearest_y = y_data[index_min]
            selected_points.append((nearest_x, nearest_y))
            selected_heights.append(nearest_y)
            ax.axvline(x=nearest_x, color='black', linestyle='--', linewidth=1)
            fig.canvas.draw()
            if len(selected_points) >= 2:
                fig.canvas.mpl_disconnect(cid)
                plt.close(fig) 
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show(block=True)
    height = max(selected_heights) - min(selected_heights)
    return selected_points, height

def height_graph(file, profile, points, height, ax=None):
    height = round(height, 2)
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    x_data = []
    y_data = []
    x_key = f"position{profile}"
    y_key = f"height{profile}"
    x_data = list([val * 1e9 for val in file[x_key]])
    y_data = list([val * 1e9 for val in file[y_key]])
    line, = ax.plot(x_data, y_data, label=f"Profile {profile}")
    ax.set_xlabel("Position [nm]")
    ax.set_ylabel("Height [nm]")
    heights = [points[0][1], points[1][1]]
    y_min = min(heights)
    y_max = max(heights)
    ax.axhline(y=y_min, color='black', linestyle='--')
    ax.axhline(y=y_max, color='black', linestyle='--') 
    ax.scatter(points[0][0], points[0][1], color='#1f77b4', s=10)
    ax.scatter(points[1][0], points[1][1], color='#1f77b4', s=10)
    ax.vlines(x=max(points[0][0], points[1][0]),ymin = y_min, ymax=y_max, color='r')
    ax.annotate('', xy=(max(points[0][0], points[1][0]), y_min), xytext=(max(points[0][0], points[1][0]), y_min + 0.1),
            arrowprops=dict(arrowstyle='->', color='r', lw=2))
    ax.annotate('', xy=(max(points[0][0], points[1][0]), y_max), xytext=(max(points[0][0], points[1][0]), y_max - 0.1),
            arrowprops=dict(arrowstyle='->', color='r', lw=2))
    line_x = max(points[0][0], points[1][0])
    y_text = (y_min + y_max) / 2
    xlim = ax.get_xlim()
    if line_x > xlim[1] - 0.1 * (xlim[1] - xlim[0]):
        text_x = line_x - 15
        ha = 'right'
    else:
        text_x = line_x + 15
        ha = 'left'
    ax.text(text_x, y_text, f"{height} [nm]", va='center', ha=ha, fontsize=10, color='r')
    return fig

def subplot_heights(data, repeats=False, repeat_profiles=None):
    profiles = len(data.keys()) // 2
    n_cols = 3
    results = []
    for i in range(profiles):
        temp_fig, temp_ax, x_data, y_data = properties_graph(data, i+1)
        points, height = select_nearest_points(temp_fig, temp_ax, x_data, y_data)
        plt.close(temp_fig)
        results.append((i, points, height))
        if repeats == True:
            if (i+1) in [x for x, y in repeat_profiles]:
                for x, y in repeat_profiles:
                    if x == (i+1):
                        corresponding_y = y
                for k in range(corresponding_y-1):
                    temp_fig, temp_ax, x_data, y_data = properties_graph(data, i+1)
                    points, height = select_nearest_points(temp_fig, temp_ax, x_data, y_data)
                    plt.close(temp_fig)
                    results.append((i, points, height))
    total_plots = len(results)
    n_rows = (total_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    axes = axes.flatten()
    for subplot_index, (i, points, height) in enumerate(results):
        ax = axes[subplot_index]
        height_graph(data, i+1, points, height, ax=ax)
        ax.set_title(f"Profile {i+1}")
    for j in range(total_plots, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()

