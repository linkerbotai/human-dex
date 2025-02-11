#!/usr/bin/env python3

import rospy,os,time,sys
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
from threading import Thread

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.color_msg import ColorMsg

class MultiCameraPublisher:
    def __init__(self):
        rospy.init_node('multi_camera_publisher_desk', anonymous=True)
        
        try:
            # 多个摄像机的设备路径
            self.camera_devices = ["/dev/video0","/dev/video2"]
        except:
            ColorMsg(msg="不存在多目RGB摄像头", color="red")
            #ColorMsg(msg="RGB摄像头video号变化", color="red")
            #self.camera_devices = ["/dev/video1","/dev/video4"]
            self.camera_devices = []
        self.hdf5 = True
        # 为每个摄像机创建一个发布者
        self.publishers = []
        # CvBridge 用于 OpenCV 图像和 ROS 图像消息之间的转换
        self.bridge = CvBridge()
        # 存储摄像机视频捕获对象的列表
        self.caps = []
        # 初始化RGB相机
        self.init_camera()
        
    def init_camera(self):
        if len(self.camera_devices) > 0:
            for i in range(len(self.camera_devices)):
                if i == 0:
                    self.publishers.append(rospy.Publisher(f'/camera/rgb/image_raw/cam_top', Image, queue_size=10))
                    ColorMsg(msg="top彩色相机准备完毕", color="green")
                if i == 1:
                    self.publishers.append(rospy.Publisher(f'/camera/rgb/image_raw/cam_front', Image, queue_size=10))
                    ColorMsg(msg="front彩色相机准备完毕", color="green")
        else:
            ColorMsg(msg="没有RGB彩色相机，不能采集hdf5数据", color="green")
            self.hdf5 = False
        # 初始化所有摄像机
        for device in self.camera_devices:
            cap = cv2.VideoCapture(device)
            if not cap.isOpened():
                rospy.logerr(f"Failed to open camera {device}.")
            self.caps.append(cap)
        
        # 启动摄像机发布线程
        self.threads = []
        for i in range(len(self.caps)):
            thread = Thread(target=self.publish_camera, args=(i,))
            thread.start()
            self.threads.append(thread)

    def publish_camera(self, index):
        rate = rospy.Rate(30)  # 10 Hz
        while not rospy.is_shutdown():

            cap = self.caps[index]
            if not cap.isOpened():
                continue
            ret, frame = cap.read()
            if not ret:
                rospy.logerr(f"Failed to capture image from camera {self.camera_devices[index]}.")
                continue
            try:
                image_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                self.publishers[index].publish(image_msg)
            except CvBridgeError as e:
                rospy.logerr(f"CvBridge Error: {e}")
            rate.sleep()
    
    def release_resources(self):
        for cap in self.caps:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        multi_cam_publisher = MultiCameraPublisher()
        rospy.spin()  # Keep the node running
    except rospy.ROSInterruptException:
        pass
    finally:
        multi_cam_publisher.release_resources()
