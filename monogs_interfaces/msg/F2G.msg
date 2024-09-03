string msg
bool has_gaussians
uint8 active_sh_degree
uint8 max_sh_degree
std_msgs/Float32MultiArray get_xyz
std_msgs/Float32MultiArray get_scaling
std_msgs/Float32MultiArray get_rotation
std_msgs/Float32MultiArray get_opacity
std_msgs/Float32MultiArray get_features
int32[] unique_kfids
int32[] n_obs
std_msgs/Float32MultiArray gtcolor
std_msgs/Float32MultiArray gtdepth
CameraMsg current_frame
CameraMsg[] keyframes
bool finish
KFWindow kf_window

