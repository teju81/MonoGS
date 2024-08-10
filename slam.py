import os
import sys
import time
from argparse import ArgumentParser
from datetime import datetime

import torch
import torch.multiprocessing as mp
import yaml
from munch import munchify

import wandb
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.system_utils import mkdir_p
from gui import gui_utils, slam_gui
from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.eval_utils import eval_ate, eval_rendering, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import FakeQueue
from utils.slam_backend import BackEnd
from utils.slam_frontend import FrontEnd


class SLAM:
    def __init__(self, config, save_dir=None):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        self.config = config
        self.save_dir = save_dir
        model_params = munchify(config["model_params"])
        opt_params = munchify(config["opt_params"])
        pipeline_params = munchify(config["pipeline_params"])
        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,
            opt_params,
            pipeline_params,
        )

        self.live_mode = self.config["Dataset"]["type"] == "realsense"
        self.monocular = self.config["Dataset"]["sensor_type"] == "monocular"
        self.use_spherical_harmonics = self.config["Training"]["spherical_harmonics"]
        self.use_gui = self.config["Results"]["use_gui"]
        if self.live_mode:
            self.use_gui = True
        self.eval_rendering = self.config["Results"]["eval_rendering"]

        model_params.sh_degree = 3 if self.use_spherical_harmonics else 0

        self.gaussians = {}
        self.params_gui = {}

        bg_color = [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        num_tum_cameras = self.config["Dataset"]["num_tum_cameras"]
        frontend_queues = [mp.Queue() for _ in range(num_tum_cameras)]
        backend_queue = mp.Queue()

        q_main2vis = [mp.Queue() if self.use_gui else FakeQueue() for _ in range(num_tum_cameras)]
        q_vis2main = [mp.Queue() if self.use_gui else FakeQueue() for _ in range(num_tum_cameras)]

        self.config["Results"]["save_dir"] = save_dir
        self.config["Training"]["monocular"] = self.monocular


        self.frontend = [FrontEnd(self.config) for _ in range(num_tum_cameras)]
        self.slam_gui = {}
        self.backend = BackEnd(self.config)

        gui_processes=[]


        for i in range(num_tum_cameras):
            self.gaussians[i] = GaussianModel(model_params.sh_degree, config=self.config)
            self.gaussians[i].init_lr(6.0)

            self.gaussians[i].training_setup(opt_params)

            self.frontend[i].frontend_id = i
            self.frontend[i].dataset = load_dataset(model_params, model_params.source_path, config=config)
            if i == 0:
                self.frontend[i].dataset.reverse_dataset()
            self.frontend[i].background = self.background #torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            self.frontend[i].pipeline_params = self.pipeline_params
            self.frontend[i].frontend_queue = frontend_queues[i]
            self.frontend[i].backend_queue = backend_queue
            self.frontend[i].q_main2vis = q_main2vis[i]
            self.frontend[i].q_vis2main = q_vis2main[i]
            self.frontend[i].set_hyperparams()
            Log(f"Initialized paramaters for Camera ID: {self.frontend[i].frontend_id}", tag_msg="INIT", tag="MonoGS")


            self.backend.gaussians[i] = self.gaussians[i]


            if self.use_gui:
                self.params_gui[i] = gui_utils.ParamsGUI(
                    pipe=self.pipeline_params,
                    background=self.background,
                    gaussians=self.gaussians[i],
                    frontend_id = i,
                    q_main2vis=q_main2vis[i],
                    q_vis2main=q_vis2main[i],
                )
                gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui[i],))

                gui_process.start()
                gui_processes.append(gui_process)
                time.sleep(5)


        self.backend.background = self.background
        self.backend.cameras_extent = 6.0
        self.backend.pipeline_params = self.pipeline_params
        self.backend.opt_params = self.opt_params
        self.backend.frontend_queues = frontend_queues
        self.backend.backend_queue = backend_queue
        self.backend.live_mode = self.live_mode

        self.backend.set_hyperparams()


        backend_process = mp.Process(target=self.backend.run)


        backend_process.start()
        #self.frontend.run()
        frontend_processes = []
        for i in range(num_tum_cameras):
            Log(f"started FrontEnd with ID: {self.frontend[i].frontend_id}", tag_msg="MULTITHREAD START", tag="MonoGS")
            frontend_process = mp.Process(target=self.frontend[i].run)
            frontend_process.start()
            time.sleep(1)
            frontend_processes.append(frontend_process)


        backend_process.join()

        for i in range(num_tum_cameras):
            frontend_processes[i].join()
            Log(f"Front End #{i} Stopped and joined the main thread")
            if self.use_gui:
                q_main2vis.put(gui_utils.GaussianPacket(finish=True))
                gui_processes[i].join()
                Log(f"GUI #{i} Stopped and joined the main thread")
        Log("Backend stopped and joined the main thread")


    def run(self):
        pass


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str)
    parser.add_argument("--eval", action="store_true")

    args = parser.parse_args(sys.argv[1:])

    mp.set_start_method("spawn")

    with open(args.config, "r") as yml:
        config = yaml.safe_load(yml)

    config = load_config(args.config)
    save_dir = None

    if args.eval:
        Log("Running MonoGS in Evaluation Mode")
        Log("Following config will be overriden")
        Log("\tsave_results=True")
        config["Results"]["save_results"] = True
        Log("\tuse_gui=False")
        config["Results"]["use_gui"] = False
        Log("\teval_rendering=True")
        config["Results"]["eval_rendering"] = True
        Log("\tuse_wandb=True")
        config["Results"]["use_wandb"] = True

    if config["Results"]["save_results"]:
        mkdir_p(config["Results"]["save_dir"])
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        path = config["Dataset"]["dataset_path"].split("/")
        save_dir = os.path.join(
            config["Results"]["save_dir"], path[-3] + "_" + path[-2], current_datetime
        )
        tmp = args.config
        tmp = tmp.split(".")[0]
        config["Results"]["save_dir"] = save_dir
        mkdir_p(save_dir)
        with open(os.path.join(save_dir, "config.yml"), "w") as file:
            documents = yaml.dump(config, file)
        Log("saving results in " + save_dir)
        run = wandb.init(
            project="MonoGS",
            name=f"{tmp}_{current_datetime}",
            config=config,
            mode=None if config["Results"]["use_wandb"] else "disabled",
        )
        wandb.define_metric("frame_idx")
        wandb.define_metric("ate*", step_metric="frame_idx")

    slam = SLAM(config, save_dir=save_dir)

    slam.run()
    wandb.finish()

    # All done
    Log("Done.")
