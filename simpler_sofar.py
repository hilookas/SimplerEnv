import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
print(BASE_DIR)
import json
import warnings
import numpy as np
from PIL import Image
from SoFar.depth.utils import depth2pcd, transform_obj_pts
from SoFar.segmentation import sam, grounding_dino as detection
from SoFar.serve.scene_graph import get_scene_graph
from SoFar.serve.utils import generate_rotation_matrix, remove_outliers
from SoFar.serve.pointso import get_model as get_pointofm_model
from SoFar.serve.chatgpt import manip_parsing, manip_spatial_reasoning
warnings.filterwarnings("ignore")
os.makedirs("output", exist_ok=True)

def sofar(image, depth, intrinsic_matrix, extrinsic_matrix, prompt):

    output_folder = "/data/workspace/SimplerEnv/output"
    
    
    image = Image.fromarray(image)
    image.save("/data/workspace/SimplerEnv/output/img_simpler.png")
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    intrinsic = [fx, fy, cx, cy]
    pcd_camera, pcd_base = depth2pcd(depth, intrinsic, extrinsic_matrix)

    print("\nStart object parsing...")
    info = manip_parsing(prompt, image)
    print(json.dumps(info, indent=2))
    object_list = list(info.keys())

    print("Start Segment Anything...")
    detection_model = detection.get_model()
    sam_model = sam.get_model()
    detections = detection.get_detections(image, object_list, detection_model, output_folder=output_folder)
    mask, ann_img, object_names = sam.get_mask(
        image, object_list, sam_model, detections, output_folder=output_folder)


    print("Generate scene graph...")
    orientation_model = get_pointofm_model()
    objects_info, objects_dict = get_scene_graph(image, pcd_base, mask, info, object_names, orientation_model,
                                                 output_folder=output_folder)
    print("objects info:")
    for node in objects_info:
        print(node)
    
    print("Start spatial reasoning...")
    response = manip_spatial_reasoning(image, prompt, objects_info)
    print(response)

    image = np.array(image)
    interact_object_id = response["interact_object_id"] - 1
    object_mask = mask[interact_object_id]
    
    # object_mask = mask[0]
    # segmented_object = pcd[object_mask]
    segmented_object = pcd_camera[object_mask]
    
    obj_pts_base = transform_obj_pts(segmented_object,extrinsic_matrix)
    
    segmented_image = image[object_mask]
    colored_object_pcd = np.concatenate((segmented_object.reshape(-1, 3), segmented_image.reshape(-1, 3)), axis=-1)
    colored_object_pcd = remove_outliers(colored_object_pcd)
    np.save(os.path.join(output_folder, f"picked_obj.npy"), colored_object_pcd)
    

    
    interact_object_id = response["interact_object_id"]
    interact_object_dict = objects_dict[interact_object_id - 1]
    init_position = interact_object_dict["center"]
    target_position = response["target_position"]
    init_orientation = interact_object_dict["orientation"]
    target_orientation = response["target_orientation"]
    
    
    if len(target_orientation) > 0 and target_orientation.keys() == init_orientation.keys():
        direction_attributes = target_orientation.keys()
        init_directions = [init_orientation[direction] for direction in direction_attributes]
        target_directions = [target_orientation[direction] for direction in direction_attributes]
        transform_matrix = generate_rotation_matrix(np.array(init_directions), np.array(target_directions)).tolist()
    else:
        transform_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    
    
    result = {
        'init_position': init_position,
        'target_position': target_position,
        'delta_position': [round(target_position[i] - init_position[i], 2) for i in range(3)],
        'init_orientation': init_orientation,
        'target_orientation': target_orientation,
        'transform_matrix': transform_matrix
    }
    print("Result:", result)
    
    return pcd_camera.reshape(-1,3), pcd_base.reshape(-1,3), colored_object_pcd[:,:3], obj_pts_base, object_mask, result['delta_position'], transform_matrix
