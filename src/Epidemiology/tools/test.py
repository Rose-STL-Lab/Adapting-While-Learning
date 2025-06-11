from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import yaml
from lib import utils
from lib.utils import load_graph_data
from model.pytorch.dcrnn_supervisor import DCRNNSupervisor, torch
import random
import numpy as np
import os

def main(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.safe_load(f)

    graph_pkl_filename = supervisor_config["data"].get("graph_pkl_filename")
    sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)

    i = 0
    np.random.seed(i)
    random.seed(i) 
    torch.manual_seed(i)
    torch.cuda.manual_seed(i)
    torch.cuda.manual_seed_all(i)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    max_itr = 1  # 12
    data = utils.load_dataset(**supervisor_config.get("data"))

    supervisor = DCRNNSupervisor(
        random_seed=i, iteration=0, max_itr=max_itr, adj_mx=adj_mx, **supervisor_config
    )
    supervisor._data = data
    supervisor.load_model(0, 0, 277)
    for i in range(5):
        mae_metric, rmse_metric, _ = supervisor.evaluate(dataset="test")
        print(mae_metric, rmse_metric)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_filename",
        default="data/model/dcrnn_cov.yaml",
        type=str,
        help="Configuration filename for restoring the model.",
    )
    parser.add_argument(
        "--use_cpu_only", default=False, type=bool, help="Set to true to only use cpu."
    )
    args = parser.parse_args()
    main(args)
