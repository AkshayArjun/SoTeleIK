import rclpy 
from rclpy.node import Node
from teleop_interfaces.msg import TeleopDataRaw

import yaml
import os
from ament_index_python.packages import get_package_share_directory

class CalibrateNode(Node):
    def __init__(self):
        super().__init__('calibrate_node')
        
        package_path = get_package_share_directory('so100mjk')
        self.yaml_path = os.path.join(package_path, 'config', 'calibrate.yaml')
        
        self.calibrate_data = None
        self.groups_to_calibrate = ['head', 'left_wrist', 'right_wrist']

        try:
            with open(self.yaml_path, 'r') as f:
                self.calibrate_data = yaml.safe_load(f)
            if self.calibrate_data is None:
                self.get_logger().error("YAML file is empty or malformed.")
                # You might want to shut down or use default values here
                return 
            self.get_logger().info(f"Initial calibration data loaded successfully.")
        except FileNotFoundError:
            self.get_logger().error(f"Calibration file not found at: {self.yaml_path}")
            return
        
        self.reset()

        self.sub_teleop = self.create_subscription(
            TeleopDataRaw, '/meta/teleop_data_raw', self.teleop_callback, 10
        )
        
        self.get_logger().info("Calibration node started. Waiting for data...")
    def reset(self):
        for group_name in self.groups_to_calibrate:
            calib_group = self.calibrate_data['calibrate'][group_name]
            calib_group['x_offset']['min'] = 0.0
            calib_group['x_offset']['max'] = 0.0
            calib_group['y_offset']['min'] = 0.0
            calib_group['y_offset']['max'] = 0.0
            calib_group['z_offset']['min'] = 0.0
            calib_group['z_offset']['max'] = 0.0
        self.get_logger().info("Calibration data reset.")

    def teleop_callback(self, msg: TeleopDataRaw):
        if self.calibrate_data is None:
            return # Don't process if initial data failed to load

        data_changed = False
        for group in msg.cartesian_groups:
            if group.group_name in self.groups_to_calibrate:
                
                calib_group = self.calibrate_data['calibrate'][group.group_name]
                pos = group.pose.position

                # Check X-axis
                if pos.x < calib_group['x_offset']['min']:
                    calib_group['x_offset']['min'] = pos.x
                    data_changed = True
                elif pos.x > calib_group['x_offset']['max']:
                    calib_group['x_offset']['max'] = pos.x
                    data_changed = True

                # Check Y-axis
                if pos.y < calib_group['y_offset']['min']:
                    calib_group['y_offset']['min'] = pos.y
                    data_changed = True
                elif pos.y > calib_group['y_offset']['max']:
                    calib_group['y_offset']['max'] = pos.y
                    data_changed = True

                # Check Z-axis
                if pos.z < calib_group['z_offset']['min']:
                    calib_group['z_offset']['min'] = pos.z
                    data_changed = True
                elif pos.z > calib_group['z_offset']['max']:
                    calib_group['z_offset']['max'] = pos.z
                    data_changed = True

        # Only write to the file if a value has actually changed
        if data_changed:
            with open(self.yaml_path, 'w') as f:
                yaml.dump(self.calibrate_data, f, sort_keys=False)
            self.get_logger().info(f"Updated calibration data.", throttle_duration_sec=1.0)

def main(args=None):
    rclpy.init(args=args)
    node = CalibrateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user. Final calibration data saved.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()