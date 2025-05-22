import os
import yaml
import torch
import time
import numpy as np

# import 9f model
from .model.pytorch.model import Model

# import xarray as xr
# import seaborn as sns
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# import matplotlib.pyplot as plt


class Emulator:
    def __init__(
        self,
        model_config_pth,
        model_checkpoint_pth,
        model_z_dict_pth,
        latent_var_pth="./emulators/Climate_online/data/latent_vars.pth",
    ):
        start_time = time.time()
        print(f"Initializing...")

        # Load model configuration
        with open(model_config_pth) as f:
            supervisor_config = yaml.safe_load(f)

        self.device = supervisor_config.get("model").get("device")
        self.model = Model(**supervisor_config.get("model")).to(self.device)

        # Load model weights
        self.model.load_state_dict(torch.load(model_checkpoint_pth))
        self.model.eval()

        # Load latent variables
        self.z_dict = torch.load(model_z_dict_pth)
        for k in self.z_dict.keys():
            self.z_dict[k] = self.z_dict[k].to(self.device)

        # Check if latent variables have been saved previously
        if os.path.exists(latent_var_pth):
            print(f"Loading pre-sampled latent variables from {latent_var_pth}")
            self.pre_sampled_zs = torch.load(latent_var_pth)
            for k in self.pre_sampled_zs.keys():
                self.pre_sampled_zs[k] = self.pre_sampled_zs[k].to(self.device)
        else:
            print("Sampling latent variables for the first time and saving them.")
            self.pre_sampled_zs = {}
            self.pre_sampled_zs["l9_zs"] = self.model.sample_z(
                self.z_dict["l9_z_mu"], self.z_dict["l9_z_cov"], 1
            ).squeeze(
                0
            )  # Sample once for level 9

            for i in range(1, 9):
                self.pre_sampled_zs[f"l{i}_zs"] = self.model.sample_z(
                    self.z_dict[f"l{i}_z_mu"], self.z_dict[f"l{i}_z_cov"], 1
                ).squeeze(
                    0
                )  # Sample once for lower levels

            # Save the sampled latent variables to a file
            torch.save(self.pre_sampled_zs, latent_var_pth)

        print(f"Total time used: {time.time() - start_time:.2f} s")

    def pred(self, x_pred):
        # Convert to tensor for prediction if numpy
        if isinstance(x_pred, np.ndarray):
            x_pred = torch.from_numpy(x_pred).to(self.device)

        # Assure same dtype
        x_pred = x_pred.to(self.z_dict["l9_z_mu"].dtype)
        buffer_dict = {}
        with torch.no_grad():
            # Use pre-sampled latent variables
            l9_zs = self.pre_sampled_zs["l9_zs"].unsqueeze(0).expand(x_pred.size(0), -1)
            l9_output_mu, _ = self.model.z_to_y(x_pred, l9_zs, level=9)

            agg_list = []
            for i in range(1, 9):
                buffer_dict[f"l{i}_zs"] = (
                    self.pre_sampled_zs[f"l{i}_zs"]
                    .unsqueeze(0)
                    .expand(x_pred.size(0), -1)
                )
                buffer_dict[f"l{i}_res_pred_mu"], buffer_dict[f"l{i}_res_pred_cov"] = (
                    self.model.z_to_y(x_pred, buffer_dict[f"l{i}_zs"], level=i)
                )
                buffer_dict[f"l{i}_res_pred_mu"] = self.model.resizer(
                    buffer_dict[f"l{i}_res_pred_mu"].reshape(
                        (
                            buffer_dict[f"l{i}_res_pred_mu"].shape[0],
                            self.model.fid_lats[i - 1],
                            -1,
                        )
                    )
                ).flatten(start_dim=1)
                agg_list.append(buffer_dict[f"l{i}_res_pred_mu"])

            ensemble_agg_mu_output = torch.mean(torch.stack(agg_list), axis=0)
            model_pred = ensemble_agg_mu_output + l9_output_mu

        return model_pred.cpu().numpy()
