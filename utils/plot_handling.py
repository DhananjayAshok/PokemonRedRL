import seaborn as sns
from utils.parameter_handling import load_parameters
from utils.log_handling import log_error
import matplotlib.pyplot as plt
import os

class Plotter:
    def __init__(self, parameters=None):
        self.parameters = load_parameters(parameters)
        self.size_params = {}
        sns.set_style("whitegrid")
        self.default_plt_params = plt.rcParams.copy()
        if self.parameters["figure_force_save"]:
            self.set_size_default() # If we're saving figures then this is likely not the final paper viz, so we dont bother with font sizes
        else:
            self.set_size_parameters()

    def set_size_parameters(self, scaler=1, font_size=16, labels_font_size=19, xtick_font_size=19, ytick_font_size=15, legend_font_size=16, title_font_size=20):
        if font_size is None:
            font_size = self.default_plt_params["font.size"]
        plt.rcParams.update({'font.size': font_size * scaler})
        if labels_font_size is None:
            labels_font_size = self.default_plt_params["axes.labelsize"]
        plt.rcParams.update({'axes.labelsize': labels_font_size * scaler})
        if xtick_font_size is None:
            xtick_font_size = self.default_plt_params["xtick.labelsize"]
        plt.rcParams.update({'xtick.labelsize': xtick_font_size * scaler})
        if ytick_font_size is None:
            ytick_font_size = self.default_plt_params["ytick.labelsize"]
        plt.rcParams.update({'ytick.labelsize': ytick_font_size * scaler})
        if title_font_size is None:
            title_font_size = self.default_plt_params["axes.titlesize"]
        plt.rcParams.update({'axes.titlesize': title_font_size * scaler})
        self.size_params["font_size"] = font_size * scaler
        self.size_params["labels_font_size"] = labels_font_size * scaler
        self.size_params["xtick_font_size"] = xtick_font_size * scaler
        self.size_params["ytick_font_size"] = ytick_font_size * scaler
        self.size_params["title_font_size"] = title_font_size * scaler
        self.size_params["legend_font_size"] = legend_font_size * scaler
        return
    
    def set_size_default(self, scaler=1):
        self.set_size_parameters(scaler=scaler, font_size=None, labels_font_size=None, xtick_font_size=None, ytick_font_size=None, legend_font_size=None, title_font_size=None)
        return

    def set_size_parameters_from_dict(self, size_params):
        """
        Set the size parameters from a dictionary. This trusts that the dictionary is correct and does not check for errors.
        """
        self.set_size_parameters(**size_params)
        return

    def get_size_input_number(self, key_name):
        while True:
            got = input(f"Enter the size for {key_name} (current value is {self.size_params[key_name]}): ")
            if got.strip() == "":
                return self.size_params[key_name]
            try:
                got = float(got)
                if got <= 0:
                    print(f"Got {got}, but it must be greater than 0")
                    continue
                return got
            except ValueError:
                print(f"Got {got}, but it must be a number")
                continue

    def test_sizes(self):
        if self.parameters["figure_force_save"]:
            self.parameters["logger"].warn("Parameters currently sets figure_force_save to True. This suggests you are running in an env without a display, but you cannot test sizes iteratively this way. This code may behave weirdly...")
        
        done = False
        while not done:
            print(f"Plot with sizes: ")
            print(f"{self.size_params}")
            plt.show()
            keepgoing = input("Do you want to keep trying different sizes? (only y will keep going):")
            if keepgoing.lower().strip() == "y":
                for key in self.size_params:
                    self.size_params[key] = self.get_size_input_number(key)
                self.set_size_parameters_from_dict(self.size_params)
            else:
                done = True
                break
        return

    def show(self, save_path=None, data_df=None, figure_force_save=None):
        if figure_force_save is None:
            figure_force_save = self.parameters["figure_force_save"]
        if figure_force_save and save_path is None:
            log_error("Figure force save is enabled, but no save path was provided", self.parameters)
            return
        if figure_force_save:
            figure_path = self.parameters["figure_dir"] + f"/{save_path}"
            figure_dir = os.path.dirname(figure_path)
            if not os.path.exists(figure_dir):
                os.makedirs(figure_dir)
            if data_df is not None:
                data_df.to_csv(f"{figure_path}.csv", index=False)
            plt.savefig(f"{figure_path}.pdf")
            plt.clf()
        else:
            plt.show()

    
