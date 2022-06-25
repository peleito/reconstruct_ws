#!/usr/bin/env python3
import numpy as np
import roslib
#roslib.load_manifest('my_package')
import cv2 as cv
import rospy
import sys
#import imutils
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from epipolar_tf.msg import ImageArray
from geometry_msgs.msg import Pose, PoseStamped
from nav_msgs.msg import Path
import tf.transformations

import rosbag
from matplotlib import pyplot as plt

bridge = CvBridge()

# K: [901.0037841796875, 0.0, 651.3675537109375, 0.0, 900.79052734375, 377.70037841796875, 0.0, 0.0, 1.0]
K = np.array([901.0037841796875, 0.0, 651.3675537109375, 0.0, 900.79052734375, 377.70037841796875, 0.0, 0.0, 1.0])
K = np.reshape(K,(3,3))
P_calib = np.eye(3,4,dtype=np.float64)
P_calib = np.matmul(K,P_calib)
scale_lambda = 0.01
# camMat = [[901.0037841796875, 0.0, 651.3675537109375],[0.0, 900.79052734375, 377.70037841796875],[0.0, 0.0, 1.0]]


def cv_convert(data):
	try:
		#print(data)
		cv1 = bridge.imgmsg_to_cv2(data.data[0],'bgr8')
		#print(cv1)
		cv2 = bridge.imgmsg_to_cv2(data.data[1],'bgr8')
		cv_image = [cv1, cv2]
	except:
		#print(e)
		cv_image = None
	
	return cv_image

def epipolar_geometry(data):
	# print('start')
	img1 = data[0]
	img2 = data[1]
	# print('sift')
	sift = cv.SIFT_create()
	# find the keypoints and descriptors with SIFT
	# print('detect')
	kp1, des1 = sift.detectAndCompute(img1,None)
	kp2, des2 = sift.detectAndCompute(img2,None)
	# FLANN parameters
	# print('flann')
	FLANN_INDEX_KDTREE = 1
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks=50)
	flann = cv.FlannBasedMatcher(index_params,search_params)
	matches = flann.knnMatch(des1,des2,k=2)
	good = []
	pts1 = []
	pts2 = []
	# ratio test as per Lowe's paper
	# print('ratio')
	for i,(m,n) in enumerate(matches):
		if m.distance < 0.8*n.distance:
			good.append(m)
			pts2.append(kp2[m.trainIdx].pt)
			pts1.append(kp1[m.queryIdx].pt)

	pts1 = np.float32(pts1)
	pts2 = np.float32(pts2)


	E, maskE = cv.findEssentialMat(pts1,pts2,K,method=cv.RANSAC,prob=0.99,threshold=1.0)
	F, maskF = cv.findFundamentalMat(pts1,pts2)
	# print(E)
	# print(F)

	# print(cv.recoverPose(E,pts1,pts2,K,maskE))
	# print(pts1)
	# print(pts2)
	# print(maskE)
	# print(E)
	# print(K)

	if E is not None:
		# [val,R,t,mask] = cv.recoverPose(E,pts1,pts2,K,maskE)

		[R, t] = decomp_E_mat(E,pts1,pts2)
		# mat = np.zeros((3,4))
		# mat[0:3,0:3] = R
		# mat[0:3,3] = [t[0,0],t[1,0],t[2,0]]
		# print(R)
		# print(t)
		# print(mat)
		
		# P = K*mat

		# R = mat[0:3,0:3]
		# t = mat[0:3,3]
	else:
		R = np.identity(3)
		t = np.array([0.0,0.0,0.0])

	# print('R: ' + str(R))
	# print('t: ' + str(t))
	t = -1*t
	R = np.linalg.inv(R)

	# print(R)
	# print(t)
	
	# return R,t
	
	# We select only inlier points
	pts1 = pts1[maskE.ravel()==1]
	pts2 = pts2[maskE.ravel()==1]
	print(np.shape(pts1))

	# Find epilines corresponding to points in right image (second image) and
	# drawing its lines on left image
	lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
	lines1 = lines1.reshape(-1,3)
	img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
	# Find epilines corresponding to points in left image (first image) and
	# drawing its lines on right image
	lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
	lines2 = lines2.reshape(-1,3)
	img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
	
	# plt.figure(1)
	plt.subplot(121),plt.imshow(img5)
	plt.subplot(122),plt.imshow(img3)
	plt.pause(0.001)

	# plt.figure(2)
	# plt.subplot(121),plt.imshow(img5)
	# plt.subplot(122),plt.imshow(img3)
	# plt.pause(0.001)
	# plt.show()
	# print('plot')

	return R,t

def drawlines(img1,img2,lines,pts1,pts2):
	''' img1 - image on which we draw the epilines for the points in img2
	lines - corresponding epilines '''
	# print(img1.shape)
	[r,c,b] = img1.shape
	# img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
	# img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)

	for r,pt1,pt2 in zip(lines,pts1,pts2):
		color = tuple(np.random.randint(0,255,3).tolist())
		x0,y0 = map(int, [0, -r[2]/r[1] ])
		x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
		img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
		c1 = tuple(pt1.astype(int))
		c2 = tuple(pt2.astype(int))
		# print(c1)
		# print(type(c1))
		img1 = cv.circle(img1,c1,5,color,-1)
		img2 = cv.circle(img2,c2,5,color,-1)

	return img1,img2

def decomp_E_mat(E, q1, q2):

	def sum_z_cal_relative_scale(R, t):
		# Get the transformation matrix
		# print('tform')
		T = form_transform(R, t)
		# print(T)
		# Make the projection matrix
		# print('matmul')
		P = np.matmul(np.concatenate((K, np.zeros((3, 1))), axis=1), T)

		# Triangulate the 3D points
		# print('triangulate')
		# print(P_calib.dtype)
		# print(P.dtype)
		# print(q1.T.dtype)
		# print(q2.T.dtype)
		hom_Q1 = cv.triangulatePoints(P_calib, P, q1.T, q2.T)
		# print(hom_Q1)
		# print(hom_Q1)
		# Also seen from cam 2
		# print('matmul')
		hom_Q2 = np.matmul(T, hom_Q1)
		# print(hom_Q2)

		# Un-homogenize
		uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
		uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]
		# print(uhom_Q1)
		# print(uhom_Q2)

		# Find the number of points there has positive z coordinate in both cameras
		sum_of_pos_z_Q1 = sum(uhom_Q1[2, :] > 0)
		sum_of_pos_z_Q2 = sum(uhom_Q2[2, :] > 0)
		# print(sum_of_pos_z_Q1)
		# print(sum_of_pos_z_Q2)

		# Form point pairs and calculate the relative scale
		# print(uhom_Q1.T[:-1] - uhom_Q1.T[1:])
		# print(uhom_Q2.T[:-1] - uhom_Q2.T[1:])
		# print(np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1)+0.0000001)
		relative_scale = np.mean(np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1)/(np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1)+0.0000001))
		# print(relative_scale)
		# print(q)

		return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale

	R1, R2, t = cv.decomposeEssentialMat(E)
	t = np.squeeze(t)

	# Make a list of the different possible pairs
	pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]

	# Check which solution there is the right one
	z_sums = []
	relative_scales = []
	# print('outside for loop')
	for R, t in pairs:
		# print('inside for loop')
		z_sum, scale = sum_z_cal_relative_scale(R, t)
		# print('after method')
		z_sums.append(z_sum)
		relative_scales.append(scale)

	# Select the pair there has the most points with positive z coordinate
	# print(relative_scales)
	right_pair_idx = np.argmax(z_sums)
	right_pair = pairs[right_pair_idx]
	relative_scale = relative_scales[right_pair_idx]
	R1, t = right_pair
	t = t * relative_scale

	t = t*scale_lambda

	return [R1, t]



def form_transform(R,t):
	T = np.identity(4)
	T[0:3,0:3] = R
	T[0:3,3] = t

	return T

def convert_pose(R,t):
	msg_stamped = PoseStamped()
	msg = Pose()

	msg.position.x = t[0]
	msg.position.y = t[1]
	msg.position.z = t[2]

	# print(R)
	rot = np.identity(4)
	rot[0:3,0:3] = R
	# print(rot)
	q = tf.transformations.quaternion_from_matrix(rot)

	msg.orientation.x = q[0]
	msg.orientation.y = q[1]
	msg.orientation.z = q[2]
	msg.orientation.w = q[3]

	msg_stamped.pose = msg
	msg_stamped.header.seq = 1
	msg_stamped.header.stamp = rospy.get_rostime()
	msg_stamped.header.frame_id = 'map'

	return msg_stamped

def global_pose(R,t):
	global last_tf
	current_tf = np.identity(4)
	current_tf[0:3,0:3] = R
	current_tf[0:3,3] = t

	# print(t)

	global_tf = np.matmul(current_tf,last_tf)
	last_tf = global_tf

	new_R = global_tf[0:3,0:3]
	new_t = global_tf[0:3,3]
	new_t = np.array([[new_t[0]],[new_t[1]],[new_t[2]]])
	return new_R,new_t

def get_path(pose):
	global path_msg
	path_msg.poses.insert(-1,pose)

def main():
	rospy.init_node('transformations')

	global last_tf
	last_tf = np.identity(4)

	global path_msg
	path_msg = Path()
	path_msg.header.seq = 1
	path_msg.header.stamp = rospy.get_rostime()
	path_msg.header.frame_id = 'map'

	
	# image_sub = rospy.Subscriber("/camera/color/image_raw",Image,imageCallback)
	# depth_sub = rospy.Subscriber("/camera/depth/image_rect_raw",Image,depthCallback)

	rate = rospy.Rate(10)
	bag_read = rosbag.Bag('paired.bag')
	bag_write = rosbag.Bag('transformation.bag','w')

	topic_list = ['/paired_images']

	count = 1
	print('Started')
	i = 1
	for topic, msg, t in bag_read.read_messages(topics=topic_list):
		# print(i)
		# print('convert')
		images = cv_convert(msg)
		#print(images)
		# print('epipole')

		R,t = epipolar_geometry(images)
		# print(np.linalg.norm(t))

		# print(R)
		# print(t)
		pose_msg = convert_pose(R,t)
		# print(pose_msg)
		bag_write.write('/local_tf',pose_msg)

		new_R,new_t = global_pose(R,t)
		global_pose_msg = convert_pose(new_R,new_t)
		bag_write.write('/global_tf',global_pose_msg)

		get_path(global_pose_msg)
		bag_write.write('/global_path',path_msg)
		# print('repeat')
		i += 1

	bag_read.close()
	bag_write.close()
	print('Finished')

if __name__=='__main__':
	main()
