import random
import time

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_mapping


class BackEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_tum_cameras = config["Dataset"]["num_tum_cameras"]
        self.gaussians = {}
        self.pipeline_params = None
        self.opt_params = None
        self.background = None
        self.cameras_extent = None
        self.frontend_queues = None
        self.backend_queue = None
        self.live_mode = False

        self.pause = False
        self.device = "cuda"
        self.dtype = torch.float32
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.last_sent = [0]*self.num_tum_cameras
        self.occ_aware_visibility = [{} for i in range(self.num_tum_cameras)]
        self.viewpoints = [{} for i in range(self.num_tum_cameras)]
        self.current_windows = [[]]*self.num_tum_cameras
        self.initialized = not self.monocular
        self.keyframe_optimizers = None

    def set_hyperparams(self):
        self.save_results = self.config["Results"]["save_results"]

        self.init_itr_num = self.config["Training"]["init_itr_num"]
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = self.config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = (
            self.cameras_extent * self.config["Training"]["init_gaussian_extent"]
        )
        self.mapping_itr_num = self.config["Training"]["mapping_itr_num"]
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        self.gaussian_th = self.config["Training"]["gaussian_th"]
        self.gaussian_extent = (
            self.cameras_extent * self.config["Training"]["gaussian_extent"]
        )
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.size_threshold = self.config["Training"]["size_threshold"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = (
            self.config["Dataset"]["single_thread"]
            if "single_thread" in self.config["Dataset"]
            else False
        )

    def add_next_kf(self, frontend_id, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None):
        self.gaussians[frontend_id].extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map
        )

    def reset(self):
        self.iteration_count = 0
        self.last_sent = [0]*self.num_tum_cameras
        self.occ_aware_visibility = [{} for i in range(self.num_tum_cameras)]
        self.viewpoints = [{} for i in range(self.num_tum_cameras)]
        self.current_windows = [[]]*self.num_tum_cameras
        self.initialized = not self.monocular
        self.keyframe_optimizers = None

        # remove all gaussians
        for i in range(self.num_tum_cameras):
            self.gaussians[i].prune_points(self.gaussians[i].unique_kfIDs >= 0)
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

    def initialize_map(self, cur_frame_idx, viewpoint, frontend_id):
        for mapping_iteration in range(self.init_itr_num):
            self.iteration_count += 1
            render_pkg = render(
                viewpoint, self.gaussians[frontend_id], self.pipeline_params, self.background
            )
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )
            loss_init = get_loss_mapping(
                self.config, image, depth, viewpoint, opacity, initialization=True
            )
            loss_init.backward()

            with torch.no_grad():
                self.gaussians[frontend_id].max_radii2D[visibility_filter] = torch.max(
                    self.gaussians[frontend_id].max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians[frontend_id].add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                if mapping_iteration % self.init_gaussian_update == 0:
                    self.gaussians[frontend_id].densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )

                if self.iteration_count == self.init_gaussian_reset or (
                    self.iteration_count == self.opt_params.densify_from_iter
                ):
                    self.gaussians[frontend_id].reset_opacity()

                self.gaussians[frontend_id].optimizer.step()
                self.gaussians[frontend_id].optimizer.zero_grad(set_to_none=True)

        self.occ_aware_visibility[frontend_id][cur_frame_idx] = (n_touched > 0).long()
        Log("Initialized map")
        return render_pkg

    def map(self, current_windows, frontend_id, prune=False, iters=1):
        Log("", tag_msg="Mapping", tag="BackEnd")
        if len(current_windows[frontend_id]) == 0:
            return

        viewpoint_stack = [self.viewpoints[frontend_id][kf_idx] for kf_idx in current_windows[frontend_id]]
        random_viewpoint_stack = []
        frames_to_optimize = self.config["Training"]["pose_window"]

        current_window_set = set(current_windows[frontend_id])
        for cam_idx, viewpoint in self.viewpoints[frontend_id].items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)

        for _ in range(iters):
            self.iteration_count += 1
            self.last_sent[frontend_id] += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []

            keyframes_opt = []

            for cam_idx in range(len(current_windows[frontend_id])):
                viewpoint = viewpoint_stack[cam_idx]
                keyframes_opt.append(viewpoint)
                render_pkg = render(
                    viewpoint, self.gaussians[frontend_id], self.pipeline_params, self.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )

                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)

            for cam_idx in torch.randperm(len(random_viewpoint_stack))[:2]:
                viewpoint = random_viewpoint_stack[cam_idx]
                render_pkg = render(
                    viewpoint, self.gaussians[frontend_id], self.pipeline_params, self.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)

            scaling = self.gaussians[frontend_id].get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()
            loss_mapping.backward()
            gaussian_split = False
            ## Deinsifying / Pruning Gaussians
            with torch.no_grad():
                self.occ_aware_visibility[frontend_id] = {}
                for idx in range((len(current_windows[frontend_id]))):
                    kf_idx = current_windows[frontend_id][idx]
                    n_touched = n_touched_acm[idx]
                    self.occ_aware_visibility[frontend_id][kf_idx] = (n_touched > 0).long()

                # # compute the visibility of the gaussians
                # # Only prune on the last iteration and when we have full window
                if prune:
                    if len(current_windows[frontend_id]) == self.config["Training"]["window_size"]:
                        prune_mode = self.config["Training"]["prune_mode"]
                        prune_coviz = 3
                        self.gaussians[frontend_id].n_obs.fill_(0)
                        for window_idx, visibility in self.occ_aware_visibility[frontend_id].items():
                            self.gaussians[frontend_id].n_obs += visibility.cpu()
                        to_prune = None
                        if prune_mode == "odometry":
                            to_prune = self.gaussians[frontend_id].n_obs < 3
                            # make sure we don't split the gaussians, break here.
                        if prune_mode == "slam":
                            # only prune keyframes which are relatively new
                            sorted_window = sorted(current_windows[frontend_id], reverse=True)
                            mask = self.gaussians[frontend_id].unique_kfIDs >= sorted_window[2]
                            if not self.initialized:
                                mask = self.gaussians[frontend_id].unique_kfIDs >= 0
                            to_prune = torch.logical_and(
                                self.gaussians[frontend_id].n_obs <= prune_coviz, mask
                            )
                        if to_prune is not None and self.monocular:
                            self.gaussians[frontend_id].prune_points(to_prune.cuda())
                            for idx in range((len(current_windows[frontend_id]))):
                                current_idx = current_windows[frontend_id][idx]
                                self.occ_aware_visibility[frontend_id][current_idx] = (
                                    self.occ_aware_visibility[frontend_id][current_idx][~to_prune]
                                )
                        if not self.initialized:
                            self.initialized = True
                            Log("Initialized SLAM")
                        # # make sure we don't split the gaussians, break here.
                    return False

                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians[frontend_id].max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians[frontend_id].max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians[frontend_id].add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                update_gaussian = (
                    self.iteration_count % self.gaussian_update_every
                    == self.gaussian_update_offset
                )
                if update_gaussian:
                    self.gaussians[frontend_id].densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                    gaussian_split = True

                ## Opacity reset
                if (self.iteration_count % self.gaussian_reset) == 0 and (
                    not update_gaussian
                ):
                    Log("Resetting the opacity of non-visible Gaussians")
                    self.gaussians[frontend_id].reset_opacity_nonvisible(visibility_filter_acm)
                    gaussian_split = True

                self.gaussians[frontend_id].optimizer.step()
                self.gaussians[frontend_id].optimizer.zero_grad(set_to_none=True)
                self.gaussians[frontend_id].update_learning_rate(self.iteration_count)
                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)
                # Pose update
                for cam_idx in range(min(frames_to_optimize, len(current_windows[frontend_id]))):
                    viewpoint = viewpoint_stack[cam_idx]
                    if viewpoint.uid == 0:
                        continue
                    update_pose(viewpoint)
        return gaussian_split

    def push_to_frontend(self, tag=None, frontend_id=None):
        if frontend_id is not None:
            frontend_list = [frontend_id]
        else:
            frontend_list = list(range(self.num_tum_cameras))

        for i in frontend_list:
            self.last_sent[i] = 0
            keyframes = []
            for kf_idx in self.current_windows[i]:
                kf = self.viewpoints[i][kf_idx]
                keyframes.append((kf_idx, kf.R.clone(), kf.T.clone()))
            if tag is None:
                tag = "sync_backend"

            msg = [tag, clone_obj(self.gaussians[i]), self.occ_aware_visibility[i], keyframes]
            self.frontend_queues[i].put(msg)

    def run(self):
        Log("Slam Backend running............")
        self.reset()
        while True:
            if self.backend_queue.empty():
                if self.pause:
                    time.sleep(0.01)
                    continue

                if sum([len(self.current_windows[i]) for i in range(self.num_tum_cameras)]) == 0:
                    time.sleep(0.01)
                    continue

                for i in range(self.num_tum_cameras):
                    if len(self.current_windows[i]) == 0:
                        continue
                    self.map(self.current_windows,i)
                    if self.last_sent[i] >= 10:
                        self.map(self.current_windows, i, prune=True, iters=10)
                        self.push_to_frontend("sync_backend",i)
            else:
                data = self.backend_queue.get()
                Log(f"Message Rxd from frontend...", tag_msg=data[0], tag="BackEnd")

                if data[0] == "stop":
                    break
                elif data[0] == "pause":
                    self.pause = True
                elif data[0] == "unpause":
                    self.pause = False
                elif data[0] == "init":
                    frontend_id = data[1]
                    print(f"frontend id: {frontend_id}")
                    cur_frame_idx = data[2]
                    viewpoint = data[3]
                    depth_map = data[4]
                    Log("", tag_msg="Resetting the system", tag="BackEnd")
                    #self.reset()

                    self.viewpoints[frontend_id][cur_frame_idx] = viewpoint
                    self.add_next_kf(
                        frontend_id, cur_frame_idx, viewpoint, depth_map=depth_map, init=True
                    )
                    self.initialize_map(cur_frame_idx, viewpoint, frontend_id)
                    self.push_to_frontend("init", frontend_id)

                elif data[0] == "keyframe":
                    frontend_id = data[1]
                    cur_frame_idx = data[2]
                    viewpoint = data[3]
                    current_window = data[4]
                    depth_map = data[5]

                    self.viewpoints[frontend_id][cur_frame_idx] = viewpoint
                    self.current_windows[frontend_id] = current_window
                    self.add_next_kf(frontend_id, cur_frame_idx, viewpoint, depth_map=depth_map)

                    opt_params = []
                    frames_to_optimize = self.config["Training"]["pose_window"]
                    iter_per_kf = self.mapping_itr_num if self.single_thread else 10
                    if not self.initialized:
                        if (
                            len(self.current_windows[frontend_id])
                            == self.config["Training"]["window_size"]
                        ):
                            frames_to_optimize = (
                                self.config["Training"]["window_size"] - 1
                            )
                            iter_per_kf = 50 if self.live_mode else 300
                            Log("Performing initial BA for initialization")
                        else:
                            iter_per_kf = self.mapping_itr_num
                    for cam_idx in range(len(self.current_windows[frontend_id])):
                        if self.current_windows[frontend_id][cam_idx] == 0:
                            continue
                        viewpoint = self.viewpoints[frontend_id][self.current_windows[frontend_id][cam_idx]]
                        if cam_idx < frames_to_optimize:
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_rot_delta],
                                    "lr": self.config["Training"]["lr"]["cam_rot_delta"]
                                    * 0.5,
                                    "name": "rot_{}".format(viewpoint.uid),
                                }
                            )
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_trans_delta],
                                    "lr": self.config["Training"]["lr"][
                                        "cam_trans_delta"
                                    ]
                                    * 0.5,
                                    "name": "trans_{}".format(viewpoint.uid),
                                }
                            )
                        opt_params.append(
                            {
                                "params": [viewpoint.exposure_a],
                                "lr": 0.01,
                                "name": "exposure_a_{}".format(viewpoint.uid),
                            }
                        )
                        opt_params.append(
                            {
                                "params": [viewpoint.exposure_b],
                                "lr": 0.01,
                                "name": "exposure_b_{}".format(viewpoint.uid),
                            }
                        )
                    self.keyframe_optimizers = torch.optim.Adam(opt_params)

                    self.map(self.current_windows, frontend_id, iters=iter_per_kf)
                    self.map(self.current_windows, frontend_id, prune=True)
                    self.push_to_frontend("keyframe", frontend_id)
                else:
                    pass
                    #Log(f"Message rxd from frontend with {data[0]}", tag_msg="OOV_MSG", tag="BackEnd")
                    #raise Exception("Unprocessed data", data)
        while not self.backend_queue.empty():
            self.backend_queue.get()
        for i in range(self.num_tum_cameras):
            while not self.frontend_queues[i].empty():
                self.frontend_queues[i].get()
        return
