#!/usr/bin/env python3
import numpy as np
import roslib
#roslib.load_manifest('my_package')
import cv2
import rospy
import sys
#import imutils
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from epipolar_tf.msg import ImageArray

import rosbag
import random

def main():
	rospy.init_node('image_pairs')
	# image_sub = rospy.Subscriber("/camera/color/image_raw",Image,imageCallback)
	# depth_sub = rospy.Subscriber("/camera/depth/image_rect_raw",Image,depthCallback)

	rate = rospy.Rate(50)
	bag_read = rosbag.Bag('full.bag')
	bag_write = rosbag.Bag('paired.bag','w')

	topic_list = ['/camera/color/image_raw']

	count = 1

	print('Started')
	for topic, msg, t in bag_read.read_messages(topics=topic_list):
		if count == 1:
			image_msg = ImageArray()
			image_msg.header = msg.header
			image_msg.data = []
			image_msg.data.insert(0,msg)
			image1 = msg

			count += 1
		elif count == 2:
			# num = random.random()
			# if num>0.8:
			image_msg.data.insert(1,msg)
			image2 = msg

			bag_write.write('/image1',image1)
			bag_write.write('/image2',image2)
			bag_write.write('/paired_images',image_msg)

			count = 1

	bag_read.close()
	bag_write.close()
	print('Finished')

if __name__=='__main__':
	main()
