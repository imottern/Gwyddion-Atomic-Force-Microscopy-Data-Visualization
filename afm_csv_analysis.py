import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px



class afmData:
    def __init__(self, source, name=" "):
        if type(source) == str:
            csv_path = source
            self.file_path = csv_path
            self.data = self.load_data()
        elif type(source) == pd.DataFrame:
            self.data = source
        self.name = name
    matrial_thicknesses = {"crsbr": 7.959e-10, "f3gt": 8e-10, "hbn":6.617e-10, "graphene":3.45e-10, "ptte2": 2.4e-10, "tairte":7e-10}
    def __str__(self):
        return str(self.data.info())
    def load_data(self):
        """
        takes Gwyddion csv and converts to clean dataframe
        Parameters: csv_path - path to csv file
        Returns: dataframe with position and height columns
        """
        data = pd.read_csv(self.file_path, delimiter=';', header=2)
        data = data.dropna(axis=1, how='all')
        new_columns = []
        counter = 1
        for i in range(data.shape[1]):
            if i % 2 == 0:
                new_columns.append(f"position{counter}")
            else:
                new_columns.append(f"height{counter}")
                counter += 1
        data.columns = new_columns
        data = data.astype(float)
        return data
    def get_df(self):
        """
        Returns the dataframe of the afmData object
        Parameters: none
        Returns: raw dataframe
        """ 
        return self.data
    def select_profiles(self, profiles):
        """
        selects specific profiles from the data based on given indices
        Parameters: profiles - list of profiles users wants to select
        Returns: new class object with only selected profiles
        """
        profiles = list(profiles)
        abridged_data = pd.DataFrame()
        for i in profiles:
            abridged_data[f'position{i}'] = self.data[f'position{i}']
            abridged_data[f'height{i}'] = self.data[f'height{i}']
        return afmData(abridged_data, name=self.name + " Abridged")
    def threshold_averaging(self, threshold):
        """
        averages profiles by checking if consecutive heights are within a given threshold
        Paranmeters: threshold - maximum height difference to be considered part of the same step
        Returns: new class object with threshold averaged profiles
        """
        averaged_data = self.data.copy()
        profiles = list(averaged_data.columns)
        for i in range(1, len(profiles), 2):
            values = averaged_data[profiles[i]]
            current_step = [values[0]]
            steps = []
            for k in range(1,len(values)):
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
            averaged_data[profiles[i]] = averaged_values
        return afmData(averaged_data, name=self.name + " Threshold Averaged")
    def zero_minimums(self):
        """
        lowers all profiles so that all minimum heights are at zero
        Parameters: none
        Returns: dataframe with all profiles zeroed at minimum heights
        """
        zeroed_data = self.data.copy()
        profiles = zeroed_data.columns
        for i in range(1, len(profiles), 2):
            min_height = zeroed_data[profiles[i]].min()
            zeroed_data[profiles[i]] = zeroed_data[profiles[i]] - min_height
        return afmData(zeroed_data, name=self.name)
    def smoothing_data(self, window_size):
        """
        smooths data using a moving average with specified size
        Parameters: window_size - size of the moving average window
        Returns: dataframe with smoothed height profiles
        """
        smoothed_data = self.data.copy()
        profiles = smoothed_data.columns
        for i in range(1, len(profiles), 2):
            smoothed_data[profiles[i]] = smoothed_data[profiles[i]].rolling(window=window_size, center=True, min_periods=1).mean()
        return afmData(smoothed_data, name=self.name + " Smoothed")
    def plotly_graph(self):
        """
        Creates a plotly graph of the data
        Parameters: none
        Returns: plotly figure object
        """
        fig = None
        profiles = list(self.data.columns)
        for i in range(1, (len(profiles)//2)+1):
            if fig is None:
                fig = px.line(self.data, x=f'position{profiles[i][-1]}', y=f'height{profiles[i][-1]}', labels={f'Profile{profiles[i][-1]}'}, markers=False)
                fig.data[0].name = f'Profile{profiles[i][-1]}'  
                fig.data[0].showlegend = True 
            else:
                fig.add_scatter(x=self.data[f'position{profiles[i][-1]}'], y=self.data[f'height{profiles[i][-1]}'], name=f'Profile{"".join(c for c in profiles[i] if c.isdigit())}')
        fig.update_layout(title_text=f'{self.name} Height Profiles', xaxis_title='Position [nm]', yaxis_title='Height [nm]')
        return fig
    def graph(self, save=False, raw=None, show_steps=False):
        """
        creates a matplotlib graph of the data
        Parameters: save - boolean if user wants save the graph as a png file
                    raw - optional data for a transparent overlay on the graph for comparison 
        Returns: none
        """
        fig, ax = plt.subplots()
        profiles = list(self.data.columns)
        for i in range(0, len(profiles), 2):
            label = f"Profile {"".join(c for c in profiles[i] if c.isdigit())}"
            x = [val * 1e9 for val in self.data[profiles[i]]]
            y = [val * 1e9 for val in self.data[profiles[i + 1]]]
            line, = ax.plot(x, y, label= label)
            color = line.get_color() 
            if raw is not None:
                raw_data = raw.data
                raw_x = [val * 1e9 for val in raw_data[profiles[i]]]
                raw_y = [val * 1e9 for val in raw_data[profiles[i + 1]]]
                ax.plot(raw_x, raw_y, alpha=0.4, color=color)
            if show_steps:
                steps = self.find_steps()
                if label in steps:
                    s = steps[label]
                    # Plateau lines
                    p1_x = np.array(s["plateau1_range"]) * 1e9
                    p2_x = np.array(s["plateau2_range"]) * 1e9
                    p1_y = s["plateau1_mean"] * 1e9
                    p2_y = s["plateau2_mean"] * 1e9
                    ax.hlines(
                        p1_y, p1_x[0], p1_x[1],
                        colors=color, linestyles="dotted", linewidth=3
                    )
                    ax.hlines(
                        p2_y, p2_x[0], p2_x[1],
                        colors=color, linestyles="dotted", linewidth=3
                    )
                    x_step = s["x_step"] * 1e9
                    ax.annotate(
                        "",
                        xy=(x_step, p2_y),
                        xytext=(x_step, p1_y),
                        arrowprops=dict(
                            arrowstyle="<->",
                            color=color,
                            linewidth=1.8
                        )
                    )
                    ax.text(
                        x_step,
                        (p1_y + p2_y) / 2,
                        f"{s['step_height'] * 1e9:.2f} nm",
                        color=color,
                        fontsize=9,
                        ha="left",
                        va="center",
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7)
                    )
        ax.legend()
        ax.set_title(f"{self.name} Height Profiles")
        ax.set_xlabel("Position [nm]")
        ax.set_ylabel("Height [nm]")
        ax.grid()
        if save:
            plt.savefig(f"{self.name} Height Profiles.png")
        plt.show() 
    def find_steps(self, threshold_factor=0.4, min_plateau_fraction=0.05):
        """
        finds steps in the data by checking for significant changes in height
        Parameters: threshold_factor - factor to the standard deviation of the derivative to determine the threshold for significant changes
                    min_plateau_fraction - minimum fraction of the profile length that a plateau must cover to be considered valid
        Returns: step heights
        """
        step_dict = {}
        profiles = list(self.data.columns)
        for i in range(0, len(profiles) - 1, 2):
            x = self.data[profiles[i]]
            z = self.data[profiles[i + 1]]
            dx = np.gradient(x)
            dz = np.gradient(z)
            with np.errstate(divide="ignore", invalid="ignore"):
                derivative = np.where(dx != 0, dz / dx, 0.0)
            deriv_std = 1.4826 * np.nanmedian(np.abs(derivative))
            if deriv_std == 0 or np.isnan(deriv_std):
                plateau_mask = np.ones_like(derivative, dtype=bool)
            else:
                thresh = threshold_factor * deriv_std
                plateau_mask = np.abs(derivative) < thresh

            mask = plateau_mask.astype(int)
            edges = np.flatnonzero(np.diff(mask)) + 1

            if plateau_mask[0]:
                edges = np.r_[0, edges]
            if plateau_mask[-1]:
                edges = np.r_[edges, len(plateau_mask)]

            segments = list(zip(edges[::2], edges[1::2]))

            min_len = max(int(min_plateau_fraction * len(x)), 1)
            segments = [(s, e) for s, e in segments if e - s >= min_len]

            if len(segments) < 2:
                continue

            idx_step = int(np.argmax(np.abs(derivative)))
            x_step = float(x[idx_step])

            left = [se for se in segments if (se[0] + se[1]) / 2 < idx_step]
            right = [se for se in segments if (se[0] + se[1]) / 2 >= idx_step]

            seg_left = max(left, key=lambda se: se[1] - se[0], default=None)
            seg_right = max(right, key=lambda se: se[1] - se[0], default=None)
            if seg_left is None and len(segments) >= 2:
                seg_left = segments[0]
            if seg_right is None and len(segments) >= 2:
                seg_right = segments[-1]

            if not (seg_left and seg_right) or seg_left == seg_right:
                continue

            s1, e1 = seg_left
            s2, e2 = seg_right

            plateau1_mean = float(np.mean(z[s1:e1]))
            plateau2_mean = float(np.mean(z[s2:e2]))
            step_height = plateau2_mean - plateau1_mean

            plateau1_range = (float(x[s1]), float(x[e1 - 1]))
            plateau2_range = (float(x[s2]), float(x[e2 - 1]))
            profile_num = "".join(c for c in profiles[i] if c.isdigit())
            step_dict[f"Profile {profile_num}"] = {
                "step_height": step_height,
                "x_step": x_step,
                "plateau1_mean": plateau1_mean,
                "plateau2_mean": plateau2_mean,
                "plateau1_range": plateau1_range,
                "plateau2_range": plateau2_range,
            }

        return step_dict
    def find_height(self):
        """
        Finds the the step of a profile using the find_steps function
        Parameters: profile - optional parameter to specify a specific profile
        Returns: step height integer
        """
        steps = self.find_steps()
        profiles = list(self.data.columns)

        height = []
        for i in range(0, len(profiles), 2):
            profile_num = "".join(c for c in profiles[i] if c.isdigit())
            key = f"Profile {profile_num}"

            height.append(
                steps[key]["step_height"] if key in steps else None
            )

        return height









# to-do:
# - add function to manually select steps
# - add function to calculate step height
# - update graphing functions to include step heights
# - add function to calulate layers



