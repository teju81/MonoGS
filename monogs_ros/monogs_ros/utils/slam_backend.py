import random
import time
import numpy as np
import torch
import torch.multiprocessing as mp
from torch import nn

from tqdm import tqdm
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray, Int32MultiArray, MultiArrayDimension


from monogs_interfaces.msg import F2B, B2F, B2LC, CameraMsg, Gaussian, Keyframe, OccAwareVisibility 
from monogs_ros.utils.camera_utils import Camera
from monogs_ros.gaussian_splatting.utils.graphics_utils import getProjectionMatrix2

import yaml
from monogs_ros.utils.config_utils import load_config
from monogs_ros.utils.dataset import load_dataset

from munch import munchify
from monogs_ros.gaussian_splatting.scene.gaussian_model import GaussianModel
from monogs_ros.utils.ros_utils import (
    convert_ros_array_message_to_tensor, 
    convert_ros_multi_array_message_to_tensor, 
    convert_tensor_to_ros_message, 
    convert_numpy_array_to_ros_message, 
    convert_ros_multi_array_message_to_numpy, 
)

from monogs_ros.gaussian_splatting.gaussian_renderer import render
from monogs_ros.gaussian_splatting.utils.loss_utils import l1_loss, ssim
from monogs_ros.utils.logging_utils import Log
from monogs_ros.utils.multiprocessing_utils import clone_obj
from monogs_ros.utils.pose_utils import update_pose
from monogs_ros.utils.slam_utils import get_loss_mapping
import pydbow3 as bow

class BackEnd(Node):
    def __init__(self, config):
        super().__init__('backend_node')
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
        self.monocular = False
        self.iteration_count = 0
        self.last_sent = [0]*self.num_tum_cameras
        self.occ_aware_visibility = [{} for i in range(self.num_tum_cameras)]
        self.viewpoints = [{} for i in range(self.num_tum_cameras)]
        self.current_windows = [[]]*self.num_tum_cameras
        self.initialized = not self.monocular
        self.keyframe_optimizers = None

        self.vocab = bow.Vocabulary()
        self.vocab.load("/root/code/monogs_ros_ws/src/monogs_ros/monogs_ros/utils/orbvoc.dbow3")

        self.queue_size_ = 100
        self.b2f_msg_counter = 0
        self.b2lc_msg_counter = 0
        self.b2f_publisher = self.create_publisher(B2F, '/Back2Front', self.queue_size_)
        self.b2lc_publisher = self.create_publisher(B2LC, '/Back2LoopClosure', self.queue_size_)

        self.f2b_subscriber = self.create_subscription(F2B, '/Front2Back', self.f2b_listener_callback, self.queue_size_)
        self.f2b_subscriber  # prevent unused variable warning


    def f2b_listener_callback(self, f2b_msg):
        self.get_logger().info('I heard from frontend: "%s"' % f2b_msg.msg)
 
        frontend_id, cur_frame_idx, viewpoint, depth_map, current_window = self.convert_from_f2b_ros_msg(f2b_msg)

        #Log(f"Message Rxd from frontend with frontend id = {frontend_id}...", tag_msg=f2b_msg.msg, tag="BackEnd")

        if f2b_msg.msg == "stop":
            pass #Need to signal a stop from here
        elif f2b_msg.msg == "pause":
            self.pause = True
        elif f2b_msg.msg == "unpause":
            self.pause = False
        elif f2b_msg.msg == "init":
            Log("", tag_msg="Resetting the system", tag="BackEnd")
            self.add_next_kf(
                frontend_id, cur_frame_idx, viewpoint, depth_map=depth_map, init=True
            )
            self.viewpoints[frontend_id][cur_frame_idx] = viewpoint
            self.initialize_map(cur_frame_idx, viewpoint, frontend_id)
            self.push_to_frontend("init", frontend_id)
            self.publish_message_to_loopclosure_node(frontend_id, viewpoint, self.occ_aware_visibility[frontend_id])

        elif f2b_msg.msg == "keyframe":
            self.current_windows[frontend_id] = current_window
            self.add_next_kf(frontend_id, cur_frame_idx, viewpoint, depth_map=depth_map)
            self.viewpoints[frontend_id][cur_frame_idx] = viewpoint

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

            self.keyframe_map_update(frontend_id, iter_per_kf)
            self.push_to_frontend("keyframe", frontend_id)
            self.publish_message_to_loopclosure_node(frontend_id, self.viewpoints[frontend_id][cur_frame_idx], self.occ_aware_visibility[frontend_id])
        else:
            pass
            #Log(f"Message rxd from frontend with {data[0]}", tag_msg="OOV_MSG", tag="BackEnd")
            #raise Exception("Unprocessed data", data)

    def get_viewpoint_from_cam_msg(self, cam_msg):

        cur_frame_idx = cam_msg.uid
        device = cam_msg.device

        gt_pose = torch.eye(4, device=device)
        gt_pose[:3,:3] = convert_ros_multi_array_message_to_tensor(cam_msg.rot_gt, device)
        gt_pose[:3,3] = convert_ros_array_message_to_tensor(cam_msg.trans_gt, device)

        gt_color = convert_ros_multi_array_message_to_tensor(cam_msg.original_image, device)
        gt_depth = convert_ros_multi_array_message_to_numpy(cam_msg.depth)
        fx = cam_msg.fx
        fy = cam_msg.fy
        cx = cam_msg.cx
        cy = cam_msg.cy
        fovx = cam_msg.fovx
        fovy = cam_msg.fovy
        image_width = cam_msg.image_width
        image_height = cam_msg.image_height


        projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            W=image_width,
            H=image_height,
        ).transpose(0, 1)
        projection_matrix = projection_matrix.to(device=device)

        viewpoint = Camera(
            cur_frame_idx,
            gt_color,
            gt_depth,
            gt_pose,
            projection_matrix,
            fx,
            fy,
            cx,
            cy,
            fovx,
            fovy,
            image_height,
            image_width,
            device=device,
        )

        viewpoint.uid = cam_msg.uid
        viewpoint.R = convert_ros_multi_array_message_to_tensor(cam_msg.rot, device)
        viewpoint.T = convert_ros_array_message_to_tensor(cam_msg.trans, device)
        viewpoint.cam_rot_delta = nn.Parameter(convert_ros_array_message_to_tensor(cam_msg.cam_rot_delta, device).requires_grad_(True))
        viewpoint.cam_trans_delta = nn.Parameter(convert_ros_array_message_to_tensor(cam_msg.cam_trans_delta, device).requires_grad_(True))
        viewpoint.exposure_a = nn.Parameter(torch.tensor(cam_msg.exposure_a, requires_grad=True, device=device))
        viewpoint.exposure_b = nn.Parameter(torch.tensor(cam_msg.exposure_b, requires_grad=True, device=device))
        viewpoint.projection_matrix = convert_ros_multi_array_message_to_tensor(cam_msg.projection_matrix, device)
        viewpoint.R_gt = convert_ros_multi_array_message_to_tensor(cam_msg.rot_gt, device)
        viewpoint.T_gt = convert_ros_array_message_to_tensor(cam_msg.trans_gt, device)

        viewpoint.original_image = convert_ros_multi_array_message_to_tensor(cam_msg.original_image, device)
        viewpoint.depth = convert_ros_multi_array_message_to_numpy(cam_msg.depth)
        viewpoint.fx = cam_msg.fx
        viewpoint.fy = cam_msg.fy
        viewpoint.cx = cam_msg.cx
        viewpoint.cy = cam_msg.cy
        viewpoint.FoVx = cam_msg.fovx
        viewpoint.FoVy = cam_msg.fovy
        viewpoint.image_width = cam_msg.image_width
        viewpoint.image_height = cam_msg.image_height
        viewpoint.device = cam_msg.device

        #viewpoint.keypoints = cam_msg.keypoints
        viewpoint.descriptors = convert_ros_multi_array_message_to_numpy(cam_msg.descriptors)
        # viewpoint.BowList = cam_msg.BowList
        # viewpoint.PlaceRecognitionQueryUID = cam_msg.PlaceRecognitionQueryUID
        # viewpoint.PlaceRecognitionWords = cam_msg.PlaceRecognitionWords
        # viewpoint.PlaceRecognitionScore = cam_msg.PlaceRecognitionScore
        viewpoint.sim_score = cam_msg.sim_score


        return viewpoint


    def convert_from_f2b_ros_msg(self, f2b_msg):
        frontend_id = f2b_msg.frontend_id
        if f2b_msg.msg == "pause" or f2b_msg.msg == "unpause":
            cur_frame_idx = None
            viewpoint = None
            depth_map = None
            current_window = None
        else:
            cur_frame_idx = f2b_msg.cur_frame_idx

            viewpoint = self.get_viewpoint_from_cam_msg(f2b_msg.viewpoint)

            depth_map = convert_ros_multi_array_message_to_numpy(f2b_msg.depth_map)
            current_window = convert_ros_array_message_to_tensor(f2b_msg.current_window, self.device).tolist()

        return frontend_id, cur_frame_idx, viewpoint, depth_map, current_window


    def publish_message_to_frontend(self, tag, gaussian, oav_dict, keyframes):
        b2f_msg = self.convert_to_b2f_ros_msg(tag, gaussian, oav_dict, keyframes)

        self.b2f_publisher.publish(b2f_msg)
        #self.get_logger().info('Publishing: %s - Hello World from backend: %d' % (b2f_msg.msg, self.b2f_msg_counter))
        self.b2f_msg_counter += 1

    def publish_message_to_loopclosure_node(self, frontend_id, viewpoint, oav_dict):
        b2lc_msg = self.convert_to_b2lc_ros_msg(frontend_id, viewpoint, oav_dict)

        self.b2lc_publisher.publish(b2lc_msg)
        self.get_logger().info('Publishing: Hello World from backend: %d' % (self.b2lc_msg_counter))
        self.b2lc_msg_counter += 1

    def convert_to_b2f_ros_msg(self, tag, gaussian, oav_dict, keyframes):
        b2f_msg = B2F()

        b2f_msg.msg = tag

        #Gaussian part of the message
        b2f_msg.gaussian.active_sh_degree = 0

        b2f_msg.gaussian.max_sh_degree = gaussian.max_sh_degree
        # np_arr = np.random.rand(2,3)
        # tensor_msg = torch.from_numpy(np_arr)
        b2f_msg.gaussian.xyz = convert_tensor_to_ros_message(gaussian._xyz)
        b2f_msg.gaussian.features_dc = convert_tensor_to_ros_message(gaussian._features_dc)
        b2f_msg.gaussian.features_rest = convert_tensor_to_ros_message(gaussian._features_rest)
        b2f_msg.gaussian.scaling = convert_tensor_to_ros_message(gaussian._scaling)
        b2f_msg.gaussian.rotation = convert_tensor_to_ros_message(gaussian._rotation)
        b2f_msg.gaussian.opacity = convert_tensor_to_ros_message(gaussian._opacity)
        b2f_msg.gaussian.max_radii2d = gaussian.max_radii2D.tolist()
        b2f_msg.gaussian.xyz_gradient_accum = convert_tensor_to_ros_message(gaussian.xyz_gradient_accum)
        b2f_msg.gaussian.unique_kfids = gaussian.unique_kfIDs.tolist()
        b2f_msg.gaussian.n_obs = gaussian.n_obs.tolist()

        for k,v in oav_dict.items():
            oav_msg = OccAwareVisibility()
            oav_msg.kf_idx = k
            oav_msg.vis_array = v.tolist()
            b2f_msg.occ_aware_visibility.append(oav_msg)

        for i in range(len(keyframes)):
            keyframe_msg = Keyframe()
            keyframe_msg.kf_idx = keyframes[i][0]
            keyframe_msg.rot = convert_tensor_to_ros_message(keyframes[i][1])
            keyframe_msg.trans = keyframes[i][2].tolist()
            b2f_msg.keyframes.append(keyframe_msg)

        return b2f_msg

    def get_camera_msg_from_viewpoint(self, viewpoint):

        if viewpoint is None:
            return None

        viewpoint_msg = CameraMsg()
        viewpoint_msg.uid = viewpoint.uid
        viewpoint_msg.device = viewpoint.device
        viewpoint_msg.rot = convert_tensor_to_ros_message(viewpoint.R)
        viewpoint_msg.trans = viewpoint.T.tolist()
        viewpoint_msg.rot_gt = convert_tensor_to_ros_message(viewpoint.R_gt)
        viewpoint_msg.trans_gt = viewpoint.T_gt.tolist()
        viewpoint_msg.original_image = convert_tensor_to_ros_message(viewpoint.original_image)
        viewpoint_msg.depth = convert_numpy_array_to_ros_message(viewpoint.depth)
        viewpoint_msg.fx = viewpoint.fx
        viewpoint_msg.fy = viewpoint.fy
        viewpoint_msg.cx = viewpoint.cx
        viewpoint_msg.cy = viewpoint.cy
        viewpoint_msg.fovx = viewpoint.FoVx
        viewpoint_msg.fovy = viewpoint.FoVy
        viewpoint_msg.image_width = viewpoint.image_width
        viewpoint_msg.image_height = viewpoint.image_height
        viewpoint_msg.cam_rot_delta = viewpoint.cam_rot_delta.tolist()
        viewpoint_msg.cam_trans_delta = viewpoint.cam_trans_delta.tolist()
        viewpoint_msg.exposure_a = viewpoint.exposure_a.item()
        viewpoint_msg.exposure_b = viewpoint.exposure_b.item()
        viewpoint_msg.projection_matrix = convert_tensor_to_ros_message(viewpoint.projection_matrix)

        #viewpoint_msg.keypoints = viewpoint.keypoints
        viewpoint_msg.descriptors = convert_numpy_array_to_ros_message(viewpoint.descriptors)
        # viewpoint_msg.BowList = viewpoint.BowList
        # viewpoint_msg.PlaceRecognitionQueryUID = viewpoint.PlaceRecognitionQueryUID
        # viewpoint_msg.PlaceRecognitionWords = viewpoint.PlaceRecognitionWords
        # viewpoint_msg.PlaceRecognitionScore = viewpoint.PlaceRecognitionScore
        viewpoint_msg.sim_score = viewpoint.sim_score

        return viewpoint_msg

    def convert_to_b2lc_ros_msg(self, frontend_id, viewpoint, oav_dict):
        b2lc_msg = B2LC()

        b2lc_msg.frontend_id = frontend_id

        if viewpoint is not None:
            b2lc_msg.viewpoint = self.get_camera_msg_from_viewpoint(viewpoint)

        for k,v in oav_dict.items():
            oav_msg = OccAwareVisibility()
            oav_msg.kf_idx = k
            oav_msg.vis_array = v.tolist()
            b2lc_msg.occ_aware_visibility.append(oav_msg)

        return b2lc_msg

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

    def ComputeBoW(self, viewpoint):

        # Compute BOW Vector

        descriptors_list = [viewpoint.descriptors[i].reshape(1,-1) for i in range(viewpoint.descriptors.shape[0])]
        
        # Compute BoW and Feature vectors
        viewpoint.dbow3_vec_dict = self.vocab.transform(descriptors_list)

    def add_next_kf(self, frontend_id, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None):
        self.gaussians[frontend_id].extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map
        )

        self.ComputeBoW(viewpoint)

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
        # # remove everything from the queues
        # while not self.backend_queue.empty():
        #     self.backend_queue.get()

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
        #Log("", tag_msg="Mapping", tag="BackEnd")
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
                #self.occ_aware_visibility[frontend_id] = {}
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
                        for idx in range((len(current_windows[frontend_id]))):
                            current_idx = current_windows[frontend_id][idx]
                            self.gaussians[frontend_id].n_obs += self.occ_aware_visibility[frontend_id][current_idx].cpu()

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
                    #Log("Resetting the opacity of non-visible Gaussians")
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

            self.publish_message_to_frontend(tag, self.gaussians[i], self.occ_aware_visibility[i], keyframes)

    def periodic_map_update(self, frontend_id):

        # Update the local Gaussian map for the given frontend id by running a single iteration without pruning
        self.map(self.current_windows,frontend_id)

        #After running the above update atleast 10 times, update the Gaussian over 10 iterations and prune
        if self.last_sent[frontend_id] >= 10:
            self.map(self.current_windows, frontend_id, prune=True, iters=10)

        self.update_oav(frontend_id)


    def keyframe_map_update(self, frontend_id, iter_per_kf):
        self.map(self.current_windows, frontend_id, iters=iter_per_kf)
        self.map(self.current_windows, frontend_id, prune=True)

        self.update_oav(frontend_id)

    def update_oav(self, frontend_id):
        # Number of Gaussians and their locations would have changed - update occ_aware_visibility
        # To Do: Run this only when things have changed. Need to bring in a check for the same.
        with torch.no_grad():
            kfid_list = self.occ_aware_visibility[frontend_id].keys()
            for kfid in kfid_list:
                render_pkg = render(self.viewpoints[frontend_id][kfid], self.gaussians[frontend_id], self.pipeline_params, self.background)
                n_touched = render_pkg["n_touched"]
                self.occ_aware_visibility[frontend_id][kfid] = (n_touched > 0).long()

    def run(self):

        if self.pause:
            time.sleep(0.01)
            return

        if sum([len(self.current_windows[i]) for i in range(self.num_tum_cameras)]) == 0:
            time.sleep(0.01)
            return

        for i in range(self.num_tum_cameras):
            if len(self.current_windows[i]) == 0:
                return
            self.periodic_map_update(i)
            self.push_to_frontend("sync_backend",i)

        return

def main():
    base_config_file = "/root/code/monogs_ros_ws/src/monogs_ros/monogs_ros/configs/rgbd/tum/base_config.yaml"

    with open(base_config_file, "r") as yml:
        base_config = yaml.safe_load(yml)
        config_file = base_config["Dataset"]["dataset_config_file"]

    config = load_config(config_file)
    model_params = munchify(config["model_params"])
    opt_params = munchify(config["opt_params"])
    pipeline_params = munchify(config["pipeline_params"])

    rclpy.init()

    backend = BackEnd(config)

    num_tum_cameras = config["Dataset"]["num_tum_cameras"]
    frontend_queues = [mp.Queue() for _ in range(num_tum_cameras)]
    gaussians = {}

    for i in range(num_tum_cameras):

        gaussians[i] = GaussianModel(model_params.sh_degree, config=config)
        gaussians[i].init_lr(6.0)

        gaussians[i].training_setup(opt_params)
        backend.gaussians[i] = gaussians[i]

    bg_color = [0, 0, 0]
    backend.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    backend.cameras_extent = 6.0
    backend.pipeline_params = pipeline_params
    backend.opt_params = opt_params
    backend.live_mode = False

    backend.set_hyperparams()
    Log("Slam Backend running............")
    backend.reset()
    try:
        while rclpy.ok():
            rclpy.spin_once(backend, timeout_sec=0.1)
            backend.run()

    finally:
        backend.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
