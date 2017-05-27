import cv2
import  matplotlib.pyplot as plt
import numpy as np

# 生成sift/surf/orb/brisk的特征点图像
def feature_point_show(img_path, feature_type, flags=True):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if feature_type == 'sift':
        sift = cv2.xfeatures2d.SIFT_create()
    elif feature_type == 'surf':
        sift = cv2.xfeatures2d.SURF_create()
    elif feature_type=='orb':
        sift = cv2.ORB_create(5000)
    elif feature_type == 'brisk':
        sift = cv2.BRISK_create()
    else:
        print('wrong argument')
        return
    kp, des = sift.detectAndCompute(gray, None)
    if flags==True:
        img = cv2.drawKeypoints(gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    elif flags==False:
        img = cv2.drawKeypoints(gray, kp, img)
    plt.figure(feature_type)
    plt.imshow(img)
    plt.show()

# 生成sift和surf的特征点匹配图像
def sift_match_plt(img_path, feature_type):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if feature_type=='sift':
        sift=cv2.xfeatures2d.SIFT_create()
    elif feature_type=='surf':
        sift=cv2.xfeatures2d.SURF_create()
    else:
        print('wrong argument')
        return
    # rotate 30 degree
    ang = np.pi / 6
    rot_mat = np.array([[np.cos(ang), np.sin(ang), 0], [-np.sin(ang), np.cos(ang), 200]])
    img_30 = cv2.warpAffine(img, rot_mat, (600, 500))
    # find the keypoints and descriptors with SIFT/SURF
    kp1, des1 = sift.detectAndCompute(img, None)
    kp2, des2 = sift.detectAndCompute(img_30, None)
    # use bfm
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    # cv2.drawMatchesKnn expects list of lists as matches.
    img_match = cv2.drawMatchesKnn(img, kp1, img_30, kp2, matches[:100], None, flags=2)
    plt.figure(feature_type)
    plt.imshow(img_match)
    plt.show()

def orb_match_plt(img_path, feature_type):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if feature_type == 'orb':
        orb = cv2.ORB_create()
    elif feature_type == 'brisk':
        orb = cv2.BRISK_create()
    else:
        print('wrong argument')
        return
    # rotate 30 degree
    ang = np.pi / 6
    rot_mat = np.array([[np.cos(ang), np.sin(ang), 0], [-np.sin(ang), np.cos(ang), 200]])
    img_30 = cv2.warpAffine(img, rot_mat, (600, 500))
    # find the keypoints and descriptors with SIFT/SURF
    kp1, des1 = orb.detectAndCompute(img, None)
    kp2, des2 = orb.detectAndCompute(img_30, None)
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1, des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    # Draw first 10 matches.
    img_match = cv2.drawMatches(img, kp1, img_30, kp2, matches[:100], None, flags=2)
    plt.figure(feature_type)
    plt.imshow(img_match)
    plt.show()
img='H:\data\image\lena.jpg'
#
orb_match_plt(img, 'orb')
# sift_match_plt(img, 'surf')
# feature_point_show(img, 'brisk',False)