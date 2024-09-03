import time

import numpy as np
import torch
import torch.multiprocessing as mp
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray, Int32MultiArray, MultiArrayDimension

from monogs_interfaces.msg import F2B, B2F, Gaussian, Keyframe, OccAwareVisibility, CameraMsg, G2F, F2G
from monogs_ros.gaussian_splatting.scene.gaussian_model import GaussianModel

import yaml
from monogs_ros.utils.config_utils import load_config
from monogs_ros.utils.dataset import load_dataset

from munch import munchify

from monogs_ros.gaussian_splatting.gaussian_renderer import render
from monogs_ros.gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from monogs_ros.gui import gui_utils
from monogs_ros.utils.camera_utils import Camera
from monogs_ros.utils.eval_utils import eval_ate, save_gaussians
from monogs_ros.utils.logging_utils import Log
from monogs_ros.utils.multiprocessing_utils import clone_obj
from monogs_ros.utils.pose_utils import update_pose
from monogs_ros.utils.slam_utils import get_loss_tracking, get_median_depth


class FrontEnd(Node):
    def __init__(self, config):
        super().__init__('frontend_node')
        self.config = config
        self.background = None
        self.pipeline_params = None
        self.frontend_queue = None
        self.backend_queue = None
        self.q_main2vis = None
        self.q_vis2main = None
        self.frontend_id = 2

        self.initialized = False
        self.kf_indices = []
        self.monocular = False
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []

        self.reset = True
        self.requested_init = False
        self.requested_keyframe = 0
        self.use_every_n_frames = 1

        self.gaussians = None
        self.cameras = dict()
        self.device = "cuda:0"
        self.pause = False


        self.queue_size_ = 10
        self.f2b_publisher = self.create_publisher(F2B, '/Front2Back', self.queue_size_)
        self.msg_counter_f2b = 0

        self.b2f_subscriber = self.create_subscription(B2F, '/Back2Front', self.b2f_listener_callback, self.queue_size_)
        self.b2f_subscriber  # prevent unused variable warning

        self.f2g_publisher = self.create_publisher(F2G, '/Front2Gui', self.queue_size_)
        self.msg_counter_f2g = 0

        self.g2f_subscriber = self.create_subscription(G2F, '/Gui2Front', self.g2f_listener_callback, self.queue_size_)
        self.g2f_subscriber  # prevent unused variable warning


    def b2f_listener_callback(self, b2f_msg):
        self.get_logger().info('I heard from backend: %s' % b2f_msg.msg)
        if b2f_msg.msg == "sync_backend":
            self.sync_backend(b2f_msg)

        elif b2f_msg.msg == "keyframe":
            self.sync_backend(b2f_msg)
            self.requested_keyframe -= 1

        elif b2f_msg.msg == "init":
            self.sync_backend(b2f_msg)
            self.requested_init = False

        elif b2f_msg.msg == "stop":
            Log("Frontend Stopped.")
            # need to break


    def sync_backend(self, b2f_msg):
        Log(f"Message Rxd from backend...", tag_msg=b2f_msg.msg, tag=f"FrontEnd_{self.frontend_id}")
        self.convert_b2f_msg_from_ros_msg(b2f_msg)

    def convert_b2f_msg_from_ros_msg(self, b2f_msg):
        model_params = munchify(self.config["model_params"])
        self.gaussians = GaussianModel(model_params.sh_degree, config=self.config)
        #Gaussian part of the message
        self.gaussians.active_sh_degree = b2f_msg.gaussian.active_sh_degree
        self.gaussians.max_sh_degree = b2f_msg.gaussian.max_sh_degree

        self.gaussians._xyz = self.convert_ros_multi_array_message_to_tensor(b2f_msg.gaussian.xyz, self.device)
        self.gaussians._features_dc = self.convert_ros_multi_array_message_to_tensor(b2f_msg.gaussian.features_dc, self.device)
        self.gaussians._features_rest = self.convert_ros_multi_array_message_to_tensor(b2f_msg.gaussian.features_rest, self.device)
        self.gaussians._scaling = self.convert_ros_multi_array_message_to_tensor(b2f_msg.gaussian.scaling, self.device)
        self.gaussians._rotation = self.convert_ros_multi_array_message_to_tensor(b2f_msg.gaussian.rotation, self.device)
        self.gaussians._opacity = self.convert_ros_multi_array_message_to_tensor(b2f_msg.gaussian.opacity, self.device)
        self.gaussians.max_radii2D = self.convert_ros_array_message_to_tensor(b2f_msg.gaussian.max_radii2d, self.device)
        self.gaussians.xyz_gradient_accum = self.convert_ros_multi_array_message_to_tensor(b2f_msg.gaussian.xyz_gradient_accum, self.device)
        self.gaussians.unique_kfIDs = self.convert_ros_array_message_to_tensor(b2f_msg.gaussian.unique_kfids, self.device)
        self.gaussians.n_obs = self.convert_ros_array_message_to_tensor(b2f_msg.gaussian.n_obs, self.device)


        occ_aware_visibility = {}
        for oav_msg in b2f_msg.occ_aware_visibility:
            kf_idx = oav_msg.kf_idx
            vis_array = self.convert_ros_array_message_to_tensor(oav_msg.vis_array, self.device)
            occ_aware_visibility[kf_idx] = vis_array
        self.occ_aware_visibility = occ_aware_visibility

        keyframes = []
        for keyframe_msg in b2f_msg.keyframes:
            kf_idx = keyframe_msg.kf_idx
            kf_R = self.convert_ros_multi_array_message_to_tensor(keyframe_msg.rot, self.device)
            kf_T = self.convert_ros_array_message_to_tensor(keyframe_msg.trans, self.device)

            self.cameras[kf_idx].update_RT(kf_R.clone(), kf_T.clone())

        return


    def convert_ros_multi_array_message_to_numpy(self, ros_multiarray_msg):

        num_dims = len(ros_multiarray_msg.layout.dim)

        # Extract the dimensions from the layout
        dim0 = ros_multiarray_msg.layout.dim[0].size
        dim1 = ros_multiarray_msg.layout.dim[1].size
        if num_dims == 2:
            dims = (dim0, dim1)
        else:
            dim2 = ros_multiarray_msg.layout.dim[2].size
            dims = (dim0, dim1, dim2)

        # Convert the flat array back into a 2D numpy array
        if isinstance(ros_multiarray_msg, Float32MultiArray):
            data = np.array(ros_multiarray_msg.data, dtype=np.float32).reshape(dims)
        elif isinstance(ros_multiarray_msg, Int32MultiArray):
            data = np.array(ros_multiarray_msg.data, dtype=np.int32).reshape(dims)

        return data

    def convert_ros_multi_array_message_to_tensor(self, ros_multiarray_msg, device):

        num_dims = len(ros_multiarray_msg.layout.dim)

        # Extract the dimensions from the layout
        dim0 = ros_multiarray_msg.layout.dim[0].size
        dim1 = ros_multiarray_msg.layout.dim[1].size
        if num_dims == 2:
            dims = (dim0, dim1)
        else:
            dim2 = ros_multiarray_msg.layout.dim[2].size
            dims = (dim0, dim1, dim2)

        # Convert the flat array back into a 2D numpy array
        if isinstance(ros_multiarray_msg, Float32MultiArray):
            data = torch.reshape(torch.tensor(ros_multiarray_msg.data, dtype=torch.float32, device=device), dims)
        elif isinstance(ros_multiarray_msg, Int32MultiArray):
            data = torch.reshape(torch.tensor(ros_multiarray_msg.data, dtype=torch.int32, device=device), dims)

        return data

    def convert_ros_array_message_to_tensor(self, ros_array_msg, device):

        if all(isinstance(i, int) for i in ros_array_msg):
            data = torch.tensor(ros_array_msg, dtype=torch.int32, device=device)
        elif all(isinstance(i, float) for i in ros_array_msg):
            data = torch.tensor(ros_array_msg, dtype=torch.float32, device=device)

        return data

    def publish_message_to_backend(self, tag, cur_frame_idx=None, viewpoint=None, depth_map=None, current_window=[]):
        f2b_msg = self.convert_f2b_msg_to_ros_msg(tag, cur_frame_idx, viewpoint, depth_map, current_window)


        self.f2b_publisher.publish(f2b_msg)
        self.get_logger().info('Publishing: %s - Hello World from frontend: %d' % (f2b_msg.msg, self.msg_counter_f2b))
        self.msg_counter_f2b += 1


    def convert_f2b_msg_to_ros_msg(self, tag, cur_frame_idx, viewpoint, depth_map, current_window):
        f2b_msg = F2B()

        f2b_msg.msg = tag
        f2b_msg.frontend_id = self.frontend_id
        if cur_frame_idx is not None:
            f2b_msg.cur_frame_idx = cur_frame_idx


        if viewpoint is not None:
            f2b_msg.viewpoint = self.get_camera_msg_from_viewpoint(viewpoint)

        if depth_map is not None:
            f2b_msg.depth_map = self.convert_numpy_array_to_ros_message(depth_map)
        f2b_msg.current_window = current_window

        return f2b_msg


    def convert_tensor_to_ros_message(self, tensor_msg):
        if tensor_msg.dtype == torch.float32 or tensor_msg.dtype == torch.float64:
            ros_multiarray_msg = Float32MultiArray()
        elif tensor_msg.dtype == torch.int32 or tensor_msg.dtype == torch.int64:
            ros_multiarray_msg = Int32MultiArray()

        # If empty tensor
        if tensor_msg.shape[0] == 0:
            return ros_multiarray_msg

        num_dims = len(tensor_msg.size())

        if num_dims == 2:
            # Define the layout of the array
            dim0 = MultiArrayDimension()
            dim0.label = "dim0"
            dim0.size = tensor_msg.shape[0]
            dim0.stride = tensor_msg.shape[1] # num of columns per row

            dim1 = MultiArrayDimension()
            dim1.label = "dim1"
            dim1.size = tensor_msg.shape[1]
            dim1.stride = 1
            ros_multiarray_msg.layout.dim = [dim0, dim1]

        elif num_dims == 3:
            # Define the layout of the array
            dim0 = MultiArrayDimension()
            dim0.label = "dim0"
            dim0.size = tensor_msg.shape[0]
            dim0.stride = tensor_msg.shape[1]*tensor_msg.shape[2] # num of columns per row

            dim1 = MultiArrayDimension()
            dim1.label = "dim1"
            dim1.size = tensor_msg.shape[1]
            dim1.stride = tensor_msg.shape[2]

            dim2 = MultiArrayDimension()
            dim2.label = "dim1"
            dim2.size = tensor_msg.shape[2]
            dim2.stride = 1
            ros_multiarray_msg.layout.dim = [dim0, dim1, dim2]
        elif num_dims > 3:
            print("#Dimensions is > 3")

        ros_multiarray_msg.layout.data_offset = 0

        # Flatten the data and assign it to the message
        ros_multiarray_msg.data = tensor_msg.flatten().tolist()

        return ros_multiarray_msg


    def convert_numpy_array_to_ros_message(self, np_arr):
        if np_arr.dtype == np.float32 or np_arr.dtype == np.float64:
            ros_multiarray_msg = Float32MultiArray()
        elif np_arr.dtype == np.int32 or np_arr.dtype == np.int32:
            ros_multiarray_msg = Int32MultiArray()

        # If empty tensor
        if np_arr.shape[0] == 0:
            return ros_multiarray_msg

        num_dims = np_arr.ndim

        if num_dims == 2:
            # Define the layout of the array
            dim0 = MultiArrayDimension()
            dim0.label = "dim0"
            dim0.size = np_arr.shape[0]
            dim0.stride = np_arr.shape[1] # num of columns per row

            dim1 = MultiArrayDimension()
            dim1.label = "dim1"
            dim1.size = np_arr.shape[1]
            dim1.stride = 1
            ros_multiarray_msg.layout.dim = [dim0, dim1]

        elif num_dims == 3:
            # Define the layout of the array
            dim0 = MultiArrayDimension()
            dim0.label = "dim0"
            dim0.size = np_arr.shape[0]
            dim0.stride = np_arr.shape[1]*np_arr.shape[2] # num of columns per row

            dim1 = MultiArrayDimension()
            dim1.label = "dim1"
            dim1.size = np_arr.shape[1]
            dim1.stride = np_arr.shape[2]

            dim2 = MultiArrayDimension()
            dim2.label = "dim1"
            dim2.size = np_arr.shape[2]
            dim2.stride = 1
            ros_multiarray_msg.layout.dim = [dim0, dim1, dim2]
        elif num_dims > 3:
            print("#Dimensions is > 3")

        ros_multiarray_msg.layout.data_offset = 0

        # Flatten the data and assign it to the message
        ros_multiarray_msg.data = np_arr.flatten().tolist()

        return ros_multiarray_msg

    def set_hyperparams(self):
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = self.config["Results"]["save_results"]
        self.save_trj = self.config["Results"]["save_trj"]
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = self.config["Training"]["single_thread"]

    def add_new_keyframe(self, cur_frame_idx, depth=None, opacity=None, init=False):
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        self.kf_indices.append(cur_frame_idx)
        viewpoint = self.cameras[cur_frame_idx]
        gt_img = viewpoint.original_image.cuda()
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]
        if self.monocular:
            if depth is None:
                initial_depth = 2 * torch.ones(1, gt_img.shape[1], gt_img.shape[2])
                initial_depth += torch.randn_like(initial_depth) * 0.3
            else:
                depth = depth.detach().clone()
                opacity = opacity.detach()
                use_inv_depth = False
                if use_inv_depth:
                    inv_depth = 1.0 / depth
                    inv_median_depth, inv_std, valid_mask = get_median_depth(
                        inv_depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        inv_depth > inv_median_depth + inv_std,
                        inv_depth < inv_median_depth - inv_std,
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    inv_depth[invalid_depth_mask] = inv_median_depth
                    inv_initial_depth = inv_depth + torch.randn_like(
                        inv_depth
                    ) * torch.where(invalid_depth_mask, inv_std * 0.5, inv_std * 0.2)
                    initial_depth = 1.0 / inv_initial_depth
                else:
                    median_depth, std, valid_mask = get_median_depth(
                        depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        depth > median_depth + std, depth < median_depth - std
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    depth[invalid_depth_mask] = median_depth
                    initial_depth = depth + torch.randn_like(depth) * torch.where(
                        invalid_depth_mask, std * 0.5, std * 0.2
                    )

                initial_depth[~valid_rgb] = 0  # Ignore the invalid rgb pixels
            return initial_depth.cpu().numpy()[0]
        # use the observed depth
        initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0)
        initial_depth[~valid_rgb.cpu()] = 0  # Ignore the invalid rgb pixels
        return initial_depth[0].numpy()

    def initialize(self, cur_frame_idx, viewpoint):
        self.initialized = not self.monocular
        self.kf_indices = []
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []
        # # remove everything from the queues
        # while not self.backend_queue.empty():
        #     self.backend_queue.get()

        # Initialise the frame at the ground truth pose
        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)

        self.kf_indices = []
        depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
        self.request_init(cur_frame_idx, viewpoint, depth_map)
        self.reset = False

    def tracking(self, cur_frame_idx, viewpoint):
        Log(f"Current Frame ID: {cur_frame_idx}", tag_msg="Tracking", tag=f"FrontEnd_{self.frontend_id}")
        prev = self.cameras[cur_frame_idx - self.use_every_n_frames]
        viewpoint.update_RT(prev.R, prev.T)

        opt_params = []
        opt_params.append(
            {
                "params": [viewpoint.cam_rot_delta],
                "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                "name": "rot_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.cam_trans_delta],
                "lr": self.config["Training"]["lr"]["cam_trans_delta"],
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

        pose_optimizer = torch.optim.Adam(opt_params)
        for tracking_itr in range(self.tracking_itr_num):
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )
            pose_optimizer.zero_grad()
            loss_tracking = get_loss_tracking(
                self.config, image, depth, opacity, viewpoint
            )
            loss_tracking.backward()

            with torch.no_grad():
                pose_optimizer.step()
                converged = update_pose(viewpoint)

            if tracking_itr % 10 == 0:
                #Log("", tag_msg="Sending Gaussian Packets to GUI", tag=f"FrontEnd_{self.frontend_id}")
                gaussian_packet = gui_utils.GaussianPacket(
                        current_frame=viewpoint,
                        gtcolor=viewpoint.original_image,
                        gtdepth=viewpoint.depth
                        if not self.monocular
                        else np.zeros((viewpoint.image_height, viewpoint.image_width)),
                    )
                self.publish_message_to_gui(gaussian_packet)

            if converged:
                break

        self.median_depth = get_median_depth(depth, opacity)

        return render_pkg

    def is_keyframe(
        self,
        cur_frame_idx,
        last_keyframe_idx,
        cur_frame_visibility_filter,
        occ_aware_visibility,
    ):
        kf_translation = self.config["Training"]["kf_translation"]
        kf_min_translation = self.config["Training"]["kf_min_translation"]
        kf_overlap = self.config["Training"]["kf_overlap"]

        curr_frame = self.cameras[cur_frame_idx]
        last_kf = self.cameras[last_keyframe_idx]
        pose_CW = getWorld2View2(curr_frame.R, curr_frame.T)
        last_kf_CW = getWorld2View2(last_kf.R, last_kf.T)
        last_kf_WC = torch.linalg.inv(last_kf_CW)
        dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3])
        dist_check = dist > kf_translation * self.median_depth
        dist_check2 = dist > kf_min_translation * self.median_depth

        union = torch.logical_or(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        intersection = torch.logical_and(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        point_ratio_2 = intersection / union
        return (point_ratio_2 < kf_overlap and dist_check2) or dist_check

    def add_to_window(
        self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window
    ):
        N_dont_touch = 2
        window = [cur_frame_idx] + window
        # remove frames which has little overlap with the current frame
        curr_frame = self.cameras[cur_frame_idx]
        to_remove = []
        removed_frame = None
        for i in range(N_dont_touch, len(window)):
            kf_idx = window[i]
            # szymkiewicz–simpson coefficient
            intersection = torch.logical_and(
                cur_frame_visibility_filter, occ_aware_visibility[kf_idx]
            ).count_nonzero()
            denom = min(
                cur_frame_visibility_filter.count_nonzero(),
                occ_aware_visibility[kf_idx].count_nonzero(),
            )
            point_ratio_2 = intersection / denom
            cut_off = (
                self.config["Training"]["kf_cutoff"]
                if "kf_cutoff" in self.config["Training"]
                else 0.4
            )
            if not self.initialized:
                cut_off = 0.4
            if point_ratio_2 <= cut_off:
                to_remove.append(kf_idx)

        if to_remove:
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]
        kf_0_WC = torch.linalg.inv(getWorld2View2(curr_frame.R, curr_frame.T))

        if len(window) > self.config["Training"]["window_size"]:
            # we need to find the keyframe to remove...
            inv_dist = []
            for i in range(N_dont_touch, len(window)):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.cameras[kf_i_idx]
                kf_i_CW = getWorld2View2(kf_i.R, kf_i.T)
                for j in range(N_dont_touch, len(window)):
                    if i == j:
                        continue
                    kf_j_idx = window[j]
                    kf_j = self.cameras[kf_j_idx]
                    kf_j_WC = torch.linalg.inv(getWorld2View2(kf_j.R, kf_j.T))
                    T_CiCj = kf_i_CW @ kf_j_WC
                    inv_dists.append(1.0 / (torch.norm(T_CiCj[0:3, 3]) + 1e-6).item())
                T_CiC0 = kf_i_CW @ kf_0_WC
                k = torch.sqrt(torch.norm(T_CiC0[0:3, 3])).item()
                inv_dist.append(k * sum(inv_dists))

            idx = np.argmax(inv_dist)
            removed_frame = window[N_dont_touch + idx]
            window.remove(removed_frame)

        return window, removed_frame

    def request_keyframe(self, cur_frame_idx, viewpoint, current_window, depthmap):
        self.publish_message_to_backend("keyframe", cur_frame_idx, viewpoint, depthmap, current_window)
        # msg = [f"Camera ID: {self.frontend_id}"]
        # self.backend_queue.put(msg)
        # msg = ["keyframe", self.frontend_id, cur_frame_idx, viewpoint, current_window, depthmap]
        # self.backend_queue.put(msg)
        self.requested_keyframe += 1

    def reqeust_mapping(self, cur_frame_idx, viewpoint):
        self.publish_message_to_backend("map", cur_frame_idx, viewpoint)
        # msg = [f"Camera ID: {self.frontend_id}"]
        # self.backend_queue.put(msg)
        # msg = ["map", cur_frame_idx, viewpoint]
        # self.backend_queue.put(msg)

    def request_init(self, cur_frame_idx, viewpoint, depth_map):
        self.publish_message_to_backend("init", cur_frame_idx, viewpoint, depth_map)
        # msg = [f"Camera ID: {self.frontend_id}"]
        # self.backend_queue.put(msg)
        # msg = ["init", self.frontend_id, cur_frame_idx, viewpoint, depth_map]
        # self.backend_queue.put(msg)
        self.requested_init = True

    def cleanup(self, cur_frame_idx):
        self.cameras[cur_frame_idx].clean()
        if cur_frame_idx % 10 == 0:
            torch.cuda.empty_cache()

    def publish_message_to_gui(self, gaussian_packet):
        f2g_msg = self.convert_f2g_msg_to_ros_msg(gaussian_packet)
        #f2g_msg.msg = 'Publishing: %s - Hello World from frontend: %d' % (f2g_msg.msg, self.msg_counter_f2g)

        self.f2g_publisher.publish(f2g_msg)
        self.get_logger().info('Publishing: %s - Hello World from frontend: %d' % (f2g_msg.msg, self.msg_counter_f2g))
        self.msg_counter_f2g += 1

    def convert_f2g_msg_to_ros_msg(self, gaussian_packet):
        f2g_msg = F2G()

        f2g_msg.msg = "Sending Gaussian Packets - without Gaussians"

        f2g_msg.has_gaussians = gaussian_packet.has_gaussians

        if gaussian_packet.has_gaussians:
            f2g_msg.msg = "Sending Gaussian Packets - with Gaussians"

            f2g_msg.active_sh_degree = gaussian_packet.active_sh_degree 

            f2g_msg.max_sh_degree = gaussian_packet.max_sh_degree
            f2g_msg.get_xyz = self.convert_tensor_to_ros_message(gaussian_packet.get_xyz)
            f2g_msg.get_features = self.convert_tensor_to_ros_message(gaussian_packet.get_features)
            f2g_msg.get_scaling = self.convert_tensor_to_ros_message(gaussian_packet.get_scaling)
            f2g_msg.get_rotation = self.convert_tensor_to_ros_message(gaussian_packet.get_rotation)
            f2g_msg.get_opacity = self.convert_tensor_to_ros_message(gaussian_packet.get_opacity)

            f2g_msg.unique_kfids = gaussian_packet.unique_kfIDs.tolist()
            f2g_msg.n_obs = gaussian_packet.n_obs.tolist()


        if gaussian_packet.gtcolor is not None:
            f2g_msg.gtcolor = self.convert_tensor_to_ros_message(gaussian_packet.gtcolor)
        
        if gaussian_packet.gtdepth is not None:
            f2g_msg.gtdepth = self.convert_numpy_array_to_ros_message(gaussian_packet.gtdepth)
    

        f2g_msg.current_frame = self.get_camera_msg_from_viewpoint(gaussian_packet.current_frame)


        if gaussian_packet.keyframes is not None:
            for viewpoint in gaussian_packet.keyframes:
                f2g_msg.keyframes.append(self.get_camera_msg_from_viewpoint(viewpoint))

        f2g_msg.finish = gaussian_packet.finish


        if gaussian_packet.kf_window is not None:
            f2g_msg.kf_window.idx = list(gaussian_packet.kf_window.keys())[0]
            f2g_msg.kf_window.current_window = list(gaussian_packet.kf_window.values())[0]



        return f2g_msg

    def get_camera_msg_from_viewpoint(self, viewpoint):

        if viewpoint is None:
            return None

        viewpoint_msg = CameraMsg()
        viewpoint_msg.uid = viewpoint.uid
        viewpoint_msg.device = viewpoint.device
        viewpoint_msg.rot = self.convert_tensor_to_ros_message(viewpoint.R)
        viewpoint_msg.trans = viewpoint.T.tolist()
        viewpoint_msg.rot_gt = self.convert_tensor_to_ros_message(viewpoint.R_gt)
        viewpoint_msg.trans_gt = viewpoint.T_gt.tolist()
        viewpoint_msg.original_image = self.convert_tensor_to_ros_message(viewpoint.original_image)
        viewpoint_msg.depth = self.convert_numpy_array_to_ros_message(viewpoint.depth)
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
        viewpoint_msg.projection_matrix = self.convert_tensor_to_ros_message(viewpoint.projection_matrix)

        return viewpoint_msg

    def g2f_listener_callback(self, g2f_msg):
        self.get_logger().info('I heard from gui: %s' % g2f_msg.msg)
        self.pause = g2f_msg.pause
        if self.pause:
            self.publish_message_to_backend(tag="pause")
        else:
            self.publish_message_to_backend(tag="unpause")


    def run(self, projection_matrix):

        # if self.q_vis2main.empty():
        #     if self.pause:
        #         return
        # else:
        #     data_vis2main = self.q_vis2main.get()
        #     self.pause = data_vis2main.flag_pause
        #     if self.pause:
        #         self.backend_queue.put(["pause"])
        #         return
        #     else:
        #         self.backend_queue.put(["unpause"])


        if self.pause:
            return


        #Log(f"Current Frame ID: {self.cur_frame_idx}", tag=f"FrontEnd_{self.frontend_id}")

        if self.requested_init:
            time.sleep(0.01)
            return

        if self.single_thread and self.requested_keyframe > 0:
            time.sleep(0.01)
            return

        if not self.initialized and self.requested_keyframe > 0:
            time.sleep(0.01)
            return

        viewpoint = Camera.init_from_dataset(
            self.dataset, self.cur_frame_idx, projection_matrix
        )
        viewpoint.compute_grad_mask(self.config)

        self.cameras[self.cur_frame_idx] = viewpoint

        if self.reset:
            self.initialize(self.cur_frame_idx, viewpoint)
            self.current_window.append(self.cur_frame_idx)
            self.cur_frame_idx += 1
            return

        self.initialized = self.initialized or (
            len(self.current_window) == self.window_size
        )

        if self.gaussians is None:
            return

        # Tracking
        render_pkg = self.tracking(self.cur_frame_idx, viewpoint)


        current_window_dict = {}
        current_window_dict[self.current_window[0]] = self.current_window[1:]
        keyframes = [self.cameras[kf_idx] for kf_idx in self.current_window]

        gaussian_packet = gui_utils.GaussianPacket(
                gaussians=clone_obj(self.gaussians),
                current_frame=viewpoint,
                keyframes=keyframes,
                kf_window=current_window_dict,
            )
        self.publish_message_to_gui(gaussian_packet)

        if self.requested_keyframe > 0:
            self.cleanup(self.cur_frame_idx)
            self.cur_frame_idx += 1 # Update cur_frame_idx to be used next iteration
            return

        last_keyframe_idx = self.current_window[0]
        check_time = (self.cur_frame_idx - last_keyframe_idx) >= self.kf_interval
        curr_visibility = (render_pkg["n_touched"] > 0).long()
        create_kf = self.is_keyframe(
            self.cur_frame_idx,
            last_keyframe_idx,
            curr_visibility,
            self.occ_aware_visibility,
        )
        if len(self.current_window) < self.window_size:
            union = torch.logical_or(
                curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
            ).count_nonzero()
            intersection = torch.logical_and(
                curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
            ).count_nonzero()
            point_ratio = intersection / union
            create_kf = (
                check_time
                and point_ratio < self.config["Training"]["kf_overlap"]
            )
        if self.single_thread:
            create_kf = check_time and create_kf
        if create_kf:
            self.current_window, removed = self.add_to_window(
                self.cur_frame_idx,
                curr_visibility,
                self.occ_aware_visibility,
                self.current_window,
            )
            if self.monocular and not self.initialized and removed is not None:
                self.reset = True
                Log(
                    "Keyframes lacks sufficient overlap to initialize the map, resetting."
                )
                return
            depth_map = self.add_new_keyframe(
                self.cur_frame_idx,
                depth=render_pkg["depth"],
                opacity=render_pkg["opacity"],
                init=False,
            )
            self.request_keyframe(
                self.cur_frame_idx, viewpoint, self.current_window, depth_map
            )
        else:
            self.cleanup(self.cur_frame_idx)
        self.cur_frame_idx += 1 # Update cur_frame_idx to be used next iteration

        return

def main():
    config_file = "/monogs_ros_ws/src/monogs_ros/monogs_ros/configs/rgbd/tum/fr1_desk.yaml"

    with open(config_file, "r") as yml:
        config = yaml.safe_load(yml)

    config = load_config(config_file)
    model_params = munchify(config["model_params"])
    opt_params = munchify(config["opt_params"])
    pipeline_params = munchify(config["pipeline_params"])

    rclpy.init()

    frontend = FrontEnd(config)
    frontend.frontend_id = 0
    frontend.dataset = load_dataset(model_params, model_params.source_path, config=config)
    bg_color = [0, 0, 0]
    frontend.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    frontend.pipeline_params = pipeline_params
    frontend.set_hyperparams()
    Log(f"Initialized paramaters for Camera ID: {frontend.frontend_id}", tag_msg="INIT", tag="MonoGS")

    Log(f"Started FrontEnd", tag=f"FrontEnd_{frontend.frontend_id}")
    frontend.cur_frame_idx = 0
    projection_matrix = getProjectionMatrix2(
        znear=0.01,
        zfar=100.0,
        fx=frontend.dataset.fx,
        fy=frontend.dataset.fy,
        cx=frontend.dataset.cx,
        cy=frontend.dataset.cy,
        W=frontend.dataset.width,
        H=frontend.dataset.height,
    ).transpose(0, 1)
    projection_matrix = projection_matrix.to(device=frontend.device)
    try:
        while rclpy.ok():
            rclpy.spin_once(frontend, timeout_sec=0.1)
            if frontend.cur_frame_idx >= len(frontend.dataset):
                break
            frontend.run(projection_matrix)
    finally:
        frontend.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
