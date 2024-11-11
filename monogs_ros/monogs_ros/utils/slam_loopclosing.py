import time
import rclpy
from rclpy.node import Node
import yaml
import torch
from torch import nn


from std_msgs.msg import String, Float32MultiArray, Int32MultiArray, MultiArrayDimension
from monogs_interfaces.msg import B2LC, OccAwareVisibility
from monogs_ros.gaussian_splatting.utils.graphics_utils import getProjectionMatrix2
from monogs_ros.utils.camera_utils import Camera

from monogs_ros.utils.logging_utils import Log
from monogs_ros.utils.config_utils import load_config
from monogs_ros.utils.ros_utils import (
    convert_ros_array_message_to_tensor, 
    convert_ros_multi_array_message_to_tensor, 
    convert_tensor_to_ros_message, 
    convert_numpy_array_to_ros_message, 
    convert_ros_multi_array_message_to_numpy, 
)

class LoopClosure(Node):
    def __init__(self, config):
        super().__init__('loop_closure_node')
        self.config = config
        self.num_tum_cameras = config["Dataset"]["num_tum_cameras"]
        self.device = "cuda"

        self.occ_aware_visibility = [{} for i in range(self.num_tum_cameras)]
        self.viewpoints = [{} for i in range(self.num_tum_cameras)]
        self.kfid_list = [[] for i in range(self.num_tum_cameras)]


        self.mQueryKF = None
        self.mKFDB = None
        self.mLoopBow_List = []
        self.mbLoopClosureDetected = False

        self.Connected_KFs_dict = [{} for i in range(self.num_tum_cameras)]

        self.queue_size_ = 100
        self.msg_counter = 0
        # self.lc2b_publisher = self.create_publisher(LC2B, '/LoopClosure2Back', self.queue_size_)

        self.b2lc_subscriber = self.create_subscription(B2LC, '/Back2LoopClosure', self.b2lc_listener_callback, self.queue_size_)
        self.b2lc_subscriber  # prevent unused variable warning

    def set_hyperparams(self):
        pass

    def get_connected_key_frames(self):

        curr_visibility = self.occ_aware_visibility[self.frontend_id][self.query_kfid]
        self.Connected_KFs_dict[self.frontend_id][self.query_kfid] = {
                                                            kfid:(intersection / union)
                                                            for kfid in self.kfid_list[self.frontend_id]
                                                            for (intersection, union) in [
                                                                (
                                                                    torch.logical_and(curr_visibility, self.occ_aware_visibility[self.frontend_id][kfid]).count_nonzero(),
                                                                    torch.logical_or(curr_visibility, self.occ_aware_visibility[self.frontend_id][kfid]).count_nonzero()
                                                                )
                                                            ]
                                                            if (intersection / union) > self.config["Training"]["kf_overlap"] and kfid != self.query_kfid
        }
        print(f"Query Key Frame ID: {self.query_kfid}")

        print(f"Connected Key Frames: {self.Connected_KFs_dict[self.frontend_id][self.query_kfid]}")

        return

    def DetectCommonRegionsFromBoW(self):
        pass


    def DetectLoopClosure(self):
        print(f"Key frames List: {self.kfid_list[self.frontend_id]}")
        if len(self.kfid_list[self.frontend_id]) > 2:
            self.get_connected_key_frames()

        # numCandidates = 3
        # self.mKFDB.DetectNBestCandidates(self.mLoopBow_List, numCandidates)

        # # Check the BoW candidates if the geometric candidate list is empty
        # if len(mLoopBow_List) == 0:
        #   self.DetectCommonRegionsFromBoW()

        # self.mKFDB.AddKF2DB(self.mQueryKF)

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

    def convert_from_b2lc_ros_msg(self, b2lc_msg):
        frontend_id = b2lc_msg.frontend_id
        viewpoint = self.get_viewpoint_from_cam_msg(b2lc_msg.viewpoint)
        print(f"Keyframe ID: {viewpoint.uid}")

        kfid = viewpoint.uid

        self.viewpoints[frontend_id][kfid] = viewpoint
        self.kfid_list[frontend_id].append(kfid)

        occ_aware_visibility = {}
        for oav_msg in b2lc_msg.occ_aware_visibility:
            kf_idx = oav_msg.kf_idx
            vis_array = convert_ros_array_message_to_tensor(oav_msg.vis_array, self.device)
            occ_aware_visibility[kf_idx] = vis_array
        self.occ_aware_visibility[frontend_id] = occ_aware_visibility

        self.frontend_id = frontend_id
        self.query_kfid = kfid


    def b2lc_listener_callback(self, b2lc_msg):
        self.get_logger().info('I heard from backend')
        self.convert_from_b2lc_ros_msg(b2lc_msg)

        self.DetectLoopClosure()

    def run(self):
        pass


def main():
    base_config_file = "/root/code/monogs_ros_ws/src/monogs_ros/monogs_ros/configs/rgbd/tum/base_config.yaml"

    with open(base_config_file, "r") as yml:
        base_config = yaml.safe_load(yml)
        config_file = base_config["Dataset"]["dataset_config_file"]

    config = load_config(config_file)

    rclpy.init()

    loopclosure = LoopClosure(config)
    loopclosure.set_hyperparams()

    Log(f"Started Loop closure Node", tag=f"LoopClosure")
    try:
        while rclpy.ok():
            rclpy.spin_once(loopclosure, timeout_sec=0.1)
            loopclosure.run()
    finally:
        loopclosure.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()