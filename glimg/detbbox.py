import os
import numpy as np

def xywh2xyxy(bbox):
    '''
    change bbox coord from [x,y,w,h] to [xmin,ymin,xmax,ymax]
    
    param:
        bbox: bounding represented by a list: [x,y,w,h]
    return:
        bbox: bounding represented by a list: [xmin,ymin,xmax,ymax]
    '''
    assert(len(bbox)==4)
    return [bbox[0]-(bbox[2]-1)/2, bbox[1]-(bbox[3]-1)/2, bbox[0]+(bbox[2]-1)/2, bbox[1]+(bbox[3]-1)/2]

def xyxy2xywh(bbox):
    '''
    change bbox coord from [xmin,ymin,xmax,ymax] to [x,y,w,h]
    
    param
        bbox: bounding represented by a list: [xmin,ymin,xmax,ymax]
    return
        bbox: bounding represented by a list: [x,y,w,h]
    '''
    assert(len(bbox)==4)
    return [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2, bbox[2]-bbox[0]+1, bbox[3]-bbox[1]+1]

def bbox_union(bbox1, bbox2):
    '''
    return the union area of two bboxes. 
    
    param 
        bbox1: bounding represented by a list: [xmin,ymin,xmax,ymax]
        bbox2: bounding with the same representation as bbox1
    return
        bbox_union: the union area of bbox1 and bbox2
    '''
    assert(len(bbox1)==4 and len(bbox2)==4)
    intersection = bbox_intersection(bbox1, bbox2)
    area1 = (bbox1[2]-bbox1[0]+1)*(bbox1[3]-bbox1[1]+1)
    area2 = (bbox2[2]-bbox2[0]+1)*(bbox2[3]-bbox2[1]+1)
    return area1+area2-intersection
    
def bbox_intersection(bbox1, bbox2):
    '''
    return the intersection area of two bboxes
    
    param
        bbox1: bounding represented by a list: [xmin,ymin,xmax,ymax]
        bbox2: bounding with the same representation as bbox1
    return
        intersection: the intersection area of bbox1 and bbox2
    '''
    assert(len(bbox1)==4 and len(bbox2)==4)
    xmin = max(bbox1[0], bbox2[0])
    ymin = max(bbox1[1], bbox2[1])
    xmax = min(bbox1[2], bbox2[2])
    ymax = min(bbox1[3], bbox2[3])
    if xmax<xmin or ymax<ymin:
        return 0
    return (xmax-xmin+1)*(ymax-ymin+1)

def bbox_iou(bbox1, bbox2, xywh=False):
    '''
    return the IOU (Intersection Over Union) of two bboxes
    
    param 
        bbox1: bounding represented by a list: [xmin,ymin,xmax,ymax]
        bbox2: bounding with the same representation as bbox1
        xywh: flag indicates if bboxes are represented by [x,y,w,h] format
    return
        intersection: the intersection area of bbox1 and bbox2
    '''
    assert(len(bbox1)==4 and len(bbox2)==4)
    if xywh:
        bbox1 = xywh2xyxy(bbox1)
        bbox2 = xywh2xyxy(bbox2)
    inter = bbox_intersection(bbox1, bbox2)
    union = bbox_union(bbox1, bbox2)
    return inter/union

#----------------------3D BBox-----------------#
def ry_matrix(ry):
    '''
    get the rotation matrix for given yaw (rad)
    
    param:
        ry: yaw angle (rad)
    return: 
        rotation matrix
    '''
    RY = np.asarray([[+np.cos(ry), 0, +np.sin(ry)],
                    [          0, 1,          0],
                    [-np.sin(ry), 0, +np.cos(ry)]])
    return RY

def local_corners(dims, ry):
    '''
    given the dimension and yaw angle, calculate the 3D coordinates of 3D BBox eight corners. 
    The origin of the coordinate system if the object center.
    
    param:
        dims: object dimension represented by [h,w,l]
        ry: object yaw angle (rad)
    return:
        numpy array of 3D coordinates with shape (3*8)
    '''
    assert(len(dims)==3)
    h, w, l = dims
    cood_x = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    cood_y = [0,0,0,0,-h,-h,-h,-h]
    cood_z = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    corners = np.asarray([cood_x, cood_y, cood_z]) # 3*8

    RY = ry_matrix(ry)
    corners = np.dot(RY, corners)
    return corners

def global_corners(dims, ry, locat):
    '''
    given the dimension, yaw angle and object location, calculate the 3D world coordinates of 3D BBox eight corners. 
    
    param:
        dims: object dimension represented by [h,w,l]
        ry: object yaw angle (rad)
        locat: object location in world coordinate system (usually the camera coordinate system), represented as [x,y,z]
    return:
        numpy array of 3D coordinates with shape (3*8)
    '''
    locat = np.array(locat).reshape(3,1)
    local_corns = local_corners(dims, ry)
    global_corns = local_corns + locat
    return global_corns

def project_global_corns(global_corns, project_mat):
    '''
    Given the global corners and projection matrix of camera, 
    calculate the projected coordinates of 8 global corners in image.
    
    param:
        global_corns: numpy array of 3D coordinates with shape (3*8)
        project_mat: projection matrix in KITTI's format. Shape (3*4), [:,0:3]: intrinsic, [:,3]: extrinsic
    return:
        numpy array of 2D coordinates with shape (2*8)
    '''
    global_corners = np.concatenate([global_corns, np.ones((1,8))], axis=0)
    proj_points = np.dot(project_mat, global_corners)
    proj_points[0,:] = proj_points[0,:]/proj_points[2,:]
    proj_points[1,:] = proj_points[1,:]/proj_points[2,:]
    proj_points = np.transpose(proj_points[:2,:])
    return proj_points

def project_corns_from_dims(dims, ry, locat, project_mat):
    '''
    Given the object's dims and pose, as well as the projection matrix, 
    calculate the projected coordinates of 8 global corners in image.
    
    param:
        dims: object dimension represented by [h,w,l]
        ry: object yaw angle (rad)
        locat: object location in world coordinate system (usually the camera coordinate system), represented as [x,y,z]
        project_mat: projection matrix in KITTI's format. Shape (3*4), [:,0:3]: intrinsic, [:,3]: extrinsic
    return:
        numpy array of 2D coordinates with shape (2*8)
    '''
    global_corns = global_corners(dims, ry, locat)
    proj_points = project_global_corns(global_corns, project_mat)
    return proj_points