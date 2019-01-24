#!/usr/bin/env python

import os
from os.path import join

import rospy
#from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from std_msgs.msg import Header


from std_srvs.srv import Empty, EmptyResponse
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import caffe


caffe.set_mode_gpu() 



mypath = '/patht/models/'







class CrowdDetectorNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.show_results = True
        self.pub_heat = rospy.Publisher('detector/heat', Image, queue_size=1)
        self.debug_on_service = rospy.Service('detector/debug_on', Empty, self.debug_on_callback)
        self.model_def = mypath + 'deploy.prototxt'
        self.model_weights = mypath+'crowd.caffemodel'
        self.meansub = np.load(mypath+'mean.npy')
        self.net = caffe.Net(self.model_def,self.model_weights, caffe.TEST)
        
        
    def provHeat(self, orig_image):
        resizedframe = cv2.resize(orig_image,(700,700))
        frameres = resizedframe.transpose((2,0,1))

        self.net.blobs['data'].data[0,...] = frameres
        self.net.forward()
        out = self.net.blobs['prob'].data
        heat = np.resize(out[:,1,:,:],(160,160))
        heat =  np.uint8(heat*255)
        cv2.imshow("heatmap",heat)
        cv2.waitKey(1)
        print heat.shape, orig_image.shape
        self.pub_heat.publish(self.bridge.cv2_to_imgmsg(heat, "8UC1"))
        #self.pub_heat.publish(self.bridge.cv2_to_imgmsg(orig_image, "bgr8"))





							 
    def listener(self):
        rospy.init_node('detector', anonymous=True)
        rospy.loginfo('Detector node started!')
        # Subscribe to user_camera topic to get the frames
        rospy.Subscriber('/usb_cam/image_raw', Image, self.detection_callback, queue_size=1)
        #rospy.Subscriber("/usb_cam/image_raw/compressed", CompressedImage, self.detection_callback, queue_size=1)
        rospy.spin()
							 
    def detection_callback(self, data):
        # Compressed image
        np_arr = np.fromstring(data.data, np.uint8)
        #cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        cv2.imshow("image", cv_image)
        cv2.waitKey(1)
        self.provHeat(cv_image)

    def debug_on_callback(self, req):
        self.show_results = True
        return EmptyResponse()

    def debug_off_callback(self, req):
        if self.show_results:
            cv2.destroyAllWindows()

        self.show_results = False
        return EmptyResponse()



if __name__ == '__main__':
    node = CrowdDetectorNode()
    node.listener()


