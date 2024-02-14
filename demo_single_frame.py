#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test trained network on a video
"""
from __future__ import  absolute_import

# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import os
import cv2
import torch
import numpy as np
import math
import pandas as pd

# import detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, _create_text_labels
from PIL import Image


# local paths to model and image
model_name = 'ResNext-101_fold_01.pth' #'X-101_RGB_60k.pth'
image_path = './output/image_00000_RGB.png'
#image_path = './output/003_Fi_mid_20230406.jpg'
output_directory = './output/results_zoe' ### CHANGE
image_dir = '/home/Masterarbeit_KH/Bilder/'
depth_dir = '/home/ZoeDepth/output/' #'/home/BoostingMonocularDepth/output/' #'/home/ZoeDepth/output/'

# load file with field data
field_data = pd.read_excel('/home/Masterarbeit_KH/Aufnahmen (1).xlsx')

if __name__ == "__main__":
    torch.cuda.is_available()
    logger = setup_logger(name=__name__)
    
    # All configurables are listed in /repos/detectron2/detectron2/config/defaults.py        
    cfg = get_cfg()
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"))
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"))
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ()
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512 #256   # faster (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (tree)
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1  
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 5
    cfg.MODEL.MASK_ON = True
    
    cfg.OUTPUT_DIR = './output'
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model_name)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    # cfg.INPUT.MIN_SIZE_TEST = 0  # no resize at test time
    
    # set detector
    predictor_synth = DefaultPredictor(cfg)    
    
    # set metadata
    tree_metadata = MetadataCatalog.get("my_tree_dataset").set(thing_classes=["Tree"], keypoint_names=["kpCP", "kpL", "kpR", "AX1", "AX2"], keypoint_flip_map=[], keypoint_connection_rules=[("kpL", "kpR", (255,255,255))])

    ### Loop over images
    data = []
    file_list = os.listdir(image_dir)
    for filename in file_list:
        # Split filename by "_"
        file_parts = filename.split('_')
        image_path = image_dir + filename
        print(filename)

        # inference
        img = cv2.imread(image_path)
        outputs_pred = predictor_synth(img)
        v_synth = Visualizer(img[:, :, ::-1],
                        metadata=tree_metadata, 
                        scale=1,
        )
        out_synth = v_synth.draw_instance_predictions(outputs_pred["instances"].to("cpu"))

        ##########
        dt_kp = outputs_pred["instances"].pred_keypoints.to("cpu")
        focal_pixel = 3.94 * 4032 / 5.6  #700 # focal length of camera in pixel = focal length[mm] * ImageWidth[pixel] / SensorWidth[mm]
        # convert error to metric distance

        ### get depth image
        depthfile_list = os.listdir(depth_dir)
        for depthname in depthfile_list:
            if ( os.path.splitext(filename)[0] == os.path.splitext(depthname)[0] ):
                depth_path = depth_dir + depthname
                print(depth_path)

        img_pil = Image.open(depth_path)
        if (depth_path == ""):
            img_pil = Image.open(image_path)  #(depth_path) ## Depth_path to depth images
        img_pil_cv = np.array(img_pil) 

        masks = outputs_pred['instances'].pred_masks

        # measure diameter
        dt_diameter = []
        for k in range(len(dt_kp)): 
            dt_diameter.append(math.hypot(dt_kp[k][2][0] - dt_kp[k][1][0], dt_kp[k][2][1] - dt_kp[k][1][1]))

            # use felling cut depth
            # same pixel coordinate system as cv2 (origin at upper left)
            depth = img_pil_cv[int(dt_kp[k][0][1]), int(dt_kp[k][0][0])] / 256 ## / 1000 standard
            #print(type(img_pil_cv))
            #print(img_pil_cv.shape)
            print(img_pil_cv[int(dt_kp[k][0][1]), int(dt_kp[k][0][0])])

            dt_diameter[k] = dt_diameter[k] * depth *10 / focal_pixel
            print("Depth = " + str(depth) )
            #print("Diameter = " + str(list(map("{:.3f}".format,dt_diameter[k]))) + " mm" + " at coordinate (" + str(int(dt_kp[k][0][0])) + ", " + str(int(dt_kp[k][0][1])) + ")")
            print("Diameter = " + "{:.2f}".format(dt_diameter[k]) + " cm" + " at coordinate (" + str(int(dt_kp[k][0][0])) + ", " + str(int(dt_kp[k][0][1])) + ")")

            ## Location of mask around DBH area
            # Calculate middle (DBH) y-coordinate
            y_BHcoord = (int(dt_kp[k][4][1]) + int(dt_kp[k][1][1])) //2
            # Get mask region corresponding to y-coordinate
            BH_region = masks[k][y_BHcoord, :].cpu()

            # Find indices where the mask is foreground (val 1)
            fore_indi = np.where(BH_region == 1)[0]
            distance = 0
            print("Mask instance:", k)
            if len(fore_indi) > 0:
                distance = np.max(fore_indi) - np.min(fore_indi)
                print("Coordinates (x, y):", np.max(fore_indi), y_BHcoord)
                print("Distance between outmost pixels:", distance)
            ##

            # Match tree mask with tree in field data
            Tree_Nr = 0
            curr_field_data = field_data[field_data['BildNr'] == int(file_parts[0])]

            #bbox = outputs_pred['instances'].pred_boxes[k].tensor.cpu().numpy() # bounding box check

            for fd_index, fd_row in curr_field_data.iterrows():
                x1_coord = fd_row['x1']
                y1_coord = fd_row['y1']
                x2_coord = fd_row['x2']
                y2_coord = fd_row['y2']
                x3_coord = fd_row['x3']
                y3_coord = fd_row['y3']

                # Retrieve the mask region corresponding to y-coordinate
                maskrow_y1 = masks[k][y1_coord, :].cpu()
                maskrow_y2 = masks[k][y2_coord, :].cpu()
                maskrow_y3 = masks[k][y3_coord, :].cpu()

                #print(str(maskrow_y1[x1_coord] == 1))

                # Check if the x-coordinate falls within the mask region
                if ((maskrow_y1[x1_coord] == 1) & (maskrow_y2[x2_coord] == 1)) | ((maskrow_y1[x1_coord] == 1) & (maskrow_y3[x3_coord] == 1)) | ((maskrow_y2[x2_coord] == 1) & (maskrow_y3[x3_coord] == 1)):
                    Tree_Nr = fd_row['BaumNr']
                    print("Mask_id " + str(k) + " matched to Tree_Nr " + str(Tree_Nr))

                # bounding box check
                #if (bbox[0,0] <= x1_coord <= bbox[0,2] and bbox[0,1] <= y1_coord <= bbox[0,3]) & (bbox[0,0] <= x2_coord <= bbox[0,2] and bbox[0,1] <= y2_coord <= bbox[0,3]) & (bbox[0,0] <= x3_coord <= bbox[0,2] and bbox[0,1] <= y3_coord <= bbox[0,3]) :
                #    Tree_Nr = fd_row['BaumNr']
                #    print("Bounding Box " + str(k) + " matched to Tree_Nr " + str(Tree_Nr))
            ###
            data.append({
                'Img_Nr': file_parts[0],
                'Stand_Type': file_parts[1],
                'Stand_Age': file_parts[2],
                'Mask_id': k,
                'Depth[m]': depth,
                'Stem-Diameter[mm]': "{:.2f}".format(dt_diameter[k]),
                'x': int(dt_kp[k][0][0]),
                'y': int(dt_kp[k][0][1]),
                'DBH_pixel_distance': distance,
                'DBH_y_coord': y_BHcoord,
                'Tree_Nr': Tree_Nr
            })
            ###
            print("\n")

            labels = _create_text_labels(outputs_pred["instances"].pred_classes.to("cpu"), outputs_pred["instances"].scores.to("cpu"), tree_metadata.thing_classes)
            #####

            ## added for saving results
            # annotated Image
            result_image_path = os.path.join(output_directory, filename + '_result_image.png')
            cv2.imwrite(result_image_path, out_synth.get_image()[:, :, ::-1])

            # Keypoint information
            #keypoint_path = os.path.join(output_directory, filename + '_keypoints.txt')
            #with open(keypoint_path, 'w') as file:
            #    for keypoints in outputs_pred['instances'].pred_keypoints:
            #        file.write(', '.join(map(str, keypoints.tolist())))
            #        file.write('\n')

    # Save depth and diameter in Excel file
    df = pd.DataFrame(data)
    excel_file_path = os.path.join(output_directory, 'depth_diameter.csv')
    df.to_csv(excel_file_path, index=False)
    ##
    ###########
    #cv2.imshow('predictions', out_synth.get_image()[:, :, ::-1])
    #k = cv2.waitKey(0)

    cv2.destroyAllWindows()
