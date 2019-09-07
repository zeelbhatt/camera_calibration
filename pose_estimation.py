import cv2
im_left = cv2.imread('left02.jpg')
im_right = cv2.imread('right02.jpg')


print (im_left.shape)
print (im_right.shape)


from matplotlib import pyplot as plt

plt.subplot(121)
plt.imshow(im_left[...,::-1])
plt.subplot(122)
plt.imshow(im_right[...,::-1])
plt.show()



ret, corners = cv2.findChessboardCorners(im_left, (7,6))

#print (corners.shape)

#print (corners[0])

corners=corners.reshape(-1,2)
#print (corners.shape)
#print (corners[0])


im_left_vis=im_left.copy()
cv2.drawChessboardCorners(im_left_vis, (7,6), corners, ret) 
plt.imshow(im_left_vis)
plt.show()







 #-----------------------------------------------------------------------------CALIBRATION--------------------------------------------------------------------------------------------------------------







import numpy as np

x,y=np.meshgrid(range(7),range(6))
print ("x:\n",x)
print ("y:\n",y)


world_points=np.hstack((x.reshape(42,1),y.reshape(42,1),np.zeros((42,1)))).astype(np.float32)
print (world_points)


print (corners[0],'->',world_points[0])
print (corners[35],'->',world_points[35])


from glob import glob

_3d_points=[]
_2d_points=[]

img_paths=glob('*.jpg') #get paths of all all images
for path in img_paths:


    im=cv2.imread(path)
    
    ret, corners = cv2.findChessboardCorners(im, (7,6))
    
    if ret: #add points only if checkerboard was correctly detected:
        _2d_points.append(corners) #append current 2D points
        _3d_points.append(world_points) #3D points are always the same
        

print("img_path is ",img_paths)
print(type(img_paths))


#print(im)
print("Shape of im is  ",im.shape)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(_3d_points, _2d_points, (im.shape[1], im.shape[0]), None, None)

print ("Ret:",ret)
print ("Mtx:",mtx," ----------------------------------> [",mtx.shape,"]")
print ("Dist:",dist," ----------> [",dist.shape,"]")
print ("rvecs:",rvecs," --------------------------------------------------------> [Shape is...",rvecs[0].shape,"]")
print ("tvecs:",tvecs," -------------------------------------------------------> [",tvecs[0].shape,"]")



plt.imshow(cv2.imread('left12.jpg')[...,::-1])
plt.show()


im=cv2.imread('left12.jpg')[...,::-1]
im_undistorted=cv2.undistort(im, mtx, dist)
plt.subplot(121)
plt.imshow(im)
plt.subplot(122)
plt.imshow(im_undistorted)
plt.show()




_3d_corners = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                           [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3]])






image_index=10
cube_corners_2d,_ = cv2.projectPoints(_3d_corners,rvecs[image_index-1],tvecs[image_index-1],mtx,dist) 
#the underscore allows to discard the second output parameter (see doc)

print (cube_corners_2d.shape) #the output consists in 8 2-dimensional points





img=cv2.imread(img_paths[image_index]) #load the correct image

red=(0,0,255) #red (in BGR)
blue=(255,0,0) #blue (in BGR)
green=(0,255,0) #green (in BGR)
line_width=5

#first draw the base in red
cv2.line(img, tuple(cube_corners_2d[0][0]), tuple(cube_corners_2d[1][0]),red,line_width)
cv2.line(img, tuple(cube_corners_2d[1][0]), tuple(cube_corners_2d[2][0]),red,line_width)
cv2.line(img, tuple(cube_corners_2d[2][0]), tuple(cube_corners_2d[3][0]),red,line_width)
cv2.line(img, tuple(cube_corners_2d[3][0]), tuple(cube_corners_2d[0][0]),red,line_width)

#now draw the pillars
cv2.line(img, tuple(cube_corners_2d[0][0]), tuple(cube_corners_2d[4][0]),blue,line_width)
cv2.line(img, tuple(cube_corners_2d[1][0]), tuple(cube_corners_2d[5][0]),blue,line_width)
cv2.line(img, tuple(cube_corners_2d[2][0]), tuple(cube_corners_2d[6][0]),blue,line_width)
cv2.line(img, tuple(cube_corners_2d[3][0]), tuple(cube_corners_2d[7][0]),blue,line_width)

#finally draw the top
cv2.line(img, tuple(cube_corners_2d[4][0]), tuple(cube_corners_2d[5][0]),green,line_width)
cv2.line(img, tuple(cube_corners_2d[5][0]), tuple(cube_corners_2d[6][0]),green,line_width)
cv2.line(img, tuple(cube_corners_2d[6][0]), tuple(cube_corners_2d[7][0]),green,line_width)
cv2.line(img, tuple(cube_corners_2d[7][0]), tuple(cube_corners_2d[4][0]),green,line_width)

#cv2.line(img, tuple(start_point), tuple(end_point),(0,0,255),3) #we set the color to red (in BGR) and line width to 3
    
plt.imshow(img[...,::-1])
plt.show()






all_right_corners=[]
all_left_corners=[]
all_3d_points=[]
idx=[1, 3, 6, 12, 14] #we use only some image pairs
valid_idxs=[] #we will also keep an list of valid indices, i.e., indices for which the procedure succeeded
for i in idx:
    im_left=cv2.imread("left%02d.jpg"%i) #load left and right images 
    im_right=cv2.imread("right%02d.jpg"%i)
    
    ret_left,left_corners=cv2.findChessboardCorners(im_left,(7,6))
    ret_right,right_corners=cv2.findChessboardCorners(im_right,(7,6))
    
    
    if ret_left and ret_right: #if both extraction succeeded
        valid_idxs.append(i)
        all_right_corners.append(right_corners)
        all_left_corners.append(left_corners)
        all_3d_points.append(world_points)

print (len(all_right_corners))
print (len(all_left_corners))
print (len(all_3d_points))

print (all_right_corners[0].shape)
print (all_left_corners[0].shape)
print (all_3d_points[0].shape)

print (all_right_corners[0].reshape(-1,2)[0])


retval, _, _, _, _, R, T, E, F=cv2.stereoCalibrate(all_3d_points,  all_left_corners, all_right_corners, (im.shape[1],im.shape[0]),mtx,dist,mtx,dist,flags=cv2.cv.CV_CALIB_FIX_INTRINSIC)






selected_image=2
left_im=cv2.imread("left%02d.jpg"%valid_idxs[selected_image])
right_im=cv2.imread("right%02d.jpg"%valid_idxs[selected_image])
left_corners=all_left_corners[selected_image].reshape(-1,2)
right_corners=all_right_corners[selected_image].reshape(-1,2)

plt.figure(figsize=(10,4))
plt.subplot(121)
plt.imshow(left_im)
plt.subplot(122)
plt.imshow(right_im)
plt.show()







cv2.circle(left_im,(left_corners[0,0],left_corners[0,1]),10,(0,0,255),10)
cv2.circle(right_im,(right_corners[0,0],right_corners[0,1]),10,(0,0,255),10)

plt.figure(figsize=(10,4))
plt.subplot(121)
plt.imshow(left_im[...,::-1])
plt.subplot(122)
plt.imshow(right_im[...,::-1])
plt.show()




lines_right = cv2.computeCorrespondEpilines(all_left_corners[selected_image], 1,F)
print (lines_right.shape)
lines_right=lines_right.reshape(-1,3) #reshape for convenience
print (lines_right.shape)




def drawLine(line,image):
    a=line[0]
    b=line[1]
    c=line[2]
    
    #ax+by+c -> y=(-ax-c)/b
    #define an inline function to compute the explicit relationship
    def y(x): return (-a*x-c)/b
    
    x0=0 #starting x point equal to zero
    x1=image.shape[1] #ending x point equal to the last column of the image
    
    y0=y(x0) #corresponding y points
    y1=y(x1)
    
    #draw the line
    cv2.line(image,(x0,int(y0)),(x1,int(y1)),(0,255,255),3)#draw the image in yellow with line_width=3






drawLine(lines_right[0],right_im)

plt.figure(figsize=(10,4))
plt.subplot(121)
plt.imshow(left_im[...,::-1])
plt.subplot(122)
plt.imshow(right_im[...,::-1])
plt.show()




lines_left = cv2.computeCorrespondEpilines(all_right_corners[selected_image], 2,F)
lines_left=lines_left.reshape(-1,3)

drawLine(lines_left[0],left_im)

plt.figure(figsize=(10,4))
plt.subplot(121)
plt.imshow(left_im[...,::-1])
plt.subplot(122)
plt.imshow(right_im[...,::-1])
plt.show()





R1=cv2.cv.fromarray(np.zeros((3,3))) #output 3x3 matrix
R2=cv2.cv.fromarray(np.zeros((3,3))) #output 3x3 matrix
P1=cv2.cv.fromarray(np.zeros((3,4))) #output 3x4 matrix
P2=cv2.cv.fromarray(np.zeros((3,4))) #output 3x4 matrix

roi1,roi2=cv2.cv.StereoRectify(cv2.cv.fromarray(mtx), #intrinsic parameters of the first camera
   cv2.cv.fromarray(mtx), #intrinsic parameters of the second camera
   cv2.cv.fromarray(dist), #distortion parameters of the first camera
   cv2.cv.fromarray(dist), #distortion parameters of the second camera
   (left_im.shape[1],left_im.shape[0]), #image dimensions
   cv2.cv.fromarray(R), #Rotation matrix between first and second cameras (returned by cv2.stereoCalibrate)
   cv2.cv.fromarray(T), #Translation vector between coordinate systems of the cameras (returned by cv2.stereoCalibrate)
   R1,R2,P1,P2) #last 4 parameters point to inizialized output variables

R1=np.array(R1) #convert output back to numpy format
R2=np.array(R2)
P1=np.array(P1)
P2=np.array(P2)






map1_x,map1_y=cv2.initUndistortRectifyMap(mtx, dist, R1, P1, (left_im.shape[1],left_im.shape[0]), cv2.cv.CV_32FC1)
map2_x,map2_y=cv2.initUndistortRectifyMap(mtx, dist, R2, P2, (left_im.shape[1],left_im.shape[0]), cv2.cv.CV_32FC1)






im_left=cv2.imread('left07.jpg')
im_right=cv2.imread('right07.jpg')

im_left_remapped=cv2.remap(im_left,map1_x,map1_y,cv2.INTER_CUBIC)
im_right_remapped=cv2.remap(im_right,map2_x,map2_y,cv2.INTER_CUBIC)






out=np.hstack((im_left_remapped,im_right_remapped))

plt.figure(figsize=(10,4))
plt.imshow(out[...,::-1])
plt.show()





for i in range(0,out.shape[0],30):
    cv2.line(out,(0,i),(out.shape[1],i),(0,255,255),3)
    
plt.figure(figsize=(10,4))
plt.imshow(out[...,::-1])
plt.show()








'''
right08.jpg', 'left02.jpg', 'left08.jpg', 'left13.jpg', 'right09.jpg', 'right13.jpg', 'right07.jpg', 'right14.jpg', 'right11.jpg', 'left06.jpg', 'left07.jpg', 'right06.jpg', 'left09.jpg', 'right02.jpg', 'left14.jpg', 'right04.jpg', 'left01.jpg', 'left05.jpg', 'left04.jpg', 'right05.jpg', 'left03.jpg', 'left12.jpg', 'left11.jpg', 'right01.jpg', 'right12.jpg', 'right03.jpg'
'''











