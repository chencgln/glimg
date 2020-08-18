'''
Author: glchen
Date: 2020-07-15 17:34:48
LastEditTime: 2020-08-18 14:16:16
Description: 
'''
import os
import numpy as np

def xywh2xyxy(bbox):
    '''
    descripttion: change bbox coord from [x,y,w,h] to [xmin,ymin,xmax,ymax]
    param {type} 
    return {type} 
    '''
    assert(len(bbox)==4)
    return [bbox[0]-(bbox[2]-1)/2, bbox[1]-(bbox[3]-1)/2, bbox[0]+(bbox[2]-1)/2, bbox[1]+(bbox[3]-1)/2]

def xyxy2xywh(bbox):
    '''
    descripttion: change bbox coord from [xmin,ymin,xmax,ymax] to [x,y,w,h]
    param {type} 
    return {type} 
    '''
    assert(len(bbox)==4)
    return [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2, bbox[2]-bbox[0]+1, bbox[3]-bbox[1]+1]

def bbox_union(bbox1, bbox2, intersection):
    '''
    descripttion: return the union area of two bboxes. 
    param {type} 
    return {type} 
    '''
    assert(len(bbox1)==4 and len(bbox2)==4 and intersection>=0)
    area1 = (bbox1[2]-bbox1[0]+1)*(bbox1[3]-bbox1[1]+1)
    area2 = (bbox2[2]-bbox2[0]+1)*(bbox2[3]-bbox2[1]+1)
    return area1+area2-intersection
    
def bbox_intersection(bbox1, bbox2):
    '''
    descripttion: return the intersection area of two bboxes
    param {bbox1} 
    return {type} 
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
    descripttion: return the IOU (Intersection Over Union) of two bboxes
    param {type} 
    return {type} 
    '''
    assert(len(bbox1)==4 and len(bbox2)==4)
    if xywh:
        bbox1 = xywh2xyxy(bbox1)
        bbox2 = xywh2xyxy(bbox2)
    inter = bbox_intersection(bbox1, bbox2)
    union = bbox_union(bbox1, bbox2, inter)
    return inter/union

#----------------------3D BBox-----------------#
def ry_matrix(ry):
    RY = np.asarray([[+np.cos(ry), 0, +np.sin(ry)],
                    [          0, 1,          0],
                    [-np.sin(ry), 0, +np.cos(ry)]])
    return RY

def local_corners(dims, ry):
    '''
    dims: h, w, l
    ry: rad; 0->right, 90->forward, 180->left
    return: shape(3*8)
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
    dims: h, w, l
    ry: rad; 0->right, 90->forward, 180->left
    locat: 
    return: shape(3*8)
    '''
    locat = np.array(locat).reshape(3,1)
    local_corns = local_corners(dims, ry)
    global_corns = local_corns + locat
    return global_corns

def project_global_corns(global_corns, P2):
    global_corners = np.concatenate([global_corns, np.ones((1,8))], axis=0)
    proj_points = np.dot(P2, global_corners)
    proj_points[0,:] = proj_points[0,:]/proj_points[2,:]
    proj_points[1,:] = proj_points[1,:]/proj_points[2,:]
    proj_points = np.transpose(proj_points[:2,:])
    return proj_points

def project_corns_from_dims(dims, ry, locat, P2):
    global_corns = global_corners(dims, ry, locat)
    global_corns = np.concatenate([global_corns, np.ones((1,8))], axis=0)
    proj_points = np.dot(P2, global_corns)
    proj_points[0,:] = proj_points[0,:]/proj_points[2,:]
    proj_points[1,:] = proj_points[1,:]/proj_points[2,:]
    proj_points = np.transpose(proj_points[:2,:])
    return proj_points

