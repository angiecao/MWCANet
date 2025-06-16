import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import math
from multiprocessing import Pool
from tqdm import tqdm

from scipy import ndimage

def greedy_filter(masks, Lc, tau):
    filtered_masks = []
    for M in masks:
        intersection = np.sum(M & Lc)  # Compute pixel-wise intersection
        contribution = intersection / np.sum(M)  # Calculate contribution ratio
        if contribution >= tau and contribution <= 1:
            filtered_masks.append(M.astype(bool))
    return filtered_masks

def compute_IoU(merged_mask, Lc):
    """Directly compute IoU based on the pre-merged mask"""
    intersection = np.sum(merged_mask & Lc)
    union = np.sum(merged_mask | Lc)
    return intersection / union if union != 0 else 0

def IoU(mask1, mask2):
    """
    Calculate the IoU between two masks
    """
    intersection = (mask1 & mask2).sum()  # Intersection
    union = (mask1 | mask2).sum()         # Union
    return intersection / float(union) if union != 0 else 0

def generate_neighbor(current_solution):
    """
    Generate a neighboring solution by randomly adding or removing a mask
    """
    neighbor_solution = current_solution.copy()
    index = random.randint(0, len(current_solution) - 1)
    
    # Randomly add or remove a mask
    neighbor_solution[index] = 1 - neighbor_solution[index]
    return neighbor_solution, index

def simulated_annealing(masks, Lc, initial_temperature, alpha, max_iterations):
    """
    Simulated Annealing Algorithm
    masks: all mask candidates
    Lc: target ground-truth mask
    initial_temperature: starting temperature
    alpha: temperature decay rate
    max_iterations: maximum number of iterations
    """
    # Initialize temperature and current solution
    T = initial_temperature
    current_solution = [random.randint(0, 1) for _ in range(len(masks))]  # Random initial solution
    merged_mask = np.zeros_like(Lc, dtype=bool)
    for i, choose in enumerate(current_solution):
        if choose:
            merged_mask |= masks[i]

    best_solution = current_solution.copy()
    best_masks = merged_mask.copy()
    best_iou = current_iou = compute_IoU(merged_mask, Lc)

    # Dynamic stopping parameters
    no_improve_steps = 0
    max_no_improve = 50  # Stop early if no improvement for 50 consecutive steps

    for _ in range(max_iterations):
        # Generate a neighboring solution
        neighbor_solution, idx = generate_neighbor(current_solution)
        new_mask = merged_mask.copy()
        if neighbor_solution[idx]:
            new_mask |= masks[idx]  # Add mask
        else:
            new_mask &= ~masks[idx] # Remove mask

        # Compute IoU of the new solution
        neighbor_IoU = compute_IoU(new_mask, Lc)

        # Calculate energy difference
        delta_E = neighbor_IoU - current_iou
        
        # Dynamic stopping check
        if neighbor_IoU > current_iou:
            no_improve_steps = 0
        else:
            no_improve_steps += 1
        
        if no_improve_steps >= max_no_improve:
            break  # Early termination

        # Determine whether to accept the new solution
        if delta_E > 0 or math.exp(delta_E / T) > random.random():
            current_solution = neighbor_solution
            merged_mask = new_mask
            current_iou = neighbor_IoU
            
            if current_iou > best_iou:  # Update global best
                best_solution = current_solution.copy()
                best_masks = merged_mask.copy()
                best_iou = current_iou
        # Cooling
        T *= alpha
    
    return best_solution, best_masks

# Class level mapping
level_mapping = {
    0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 6: 0,        # level 0
    5: 1, 12: 1, 13: 1, 21:1,                  # level 1
    7: 2, 18: 2, 19: 2,                        # level 2
    11: 3, 9: 3, 10: 3, 8: 3,                  # level 3
    15: 4, 16: 4, 17: 4, 14: 4,                # level 4
    20: 5, 22: 5                               # level 5
}

def make_pseudo(sam, logits, label):
    new_mask = np.full((600,600), 255, dtype=np.uint8)
    cls_list = [c for c in np.unique(label) if c != 255]
    cls_list = sorted(cls_list, key=lambda x: level_mapping[x])

    for cls_ in cls_list:
        Lc = logits == cls_
        filtered_masks = greedy_filter(sam, Lc, 0.001)
        if len(filtered_masks) == 0:
            continue
        elif len(filtered_masks) == 1:
            new_mask[filtered_masks[0]] = cls_
            continue
    
        # Simulated annealing hyperparameters
        initial_temperature = 10
        alpha = 0.95
        max_iterations = 1000

        best_sol, best_merged = simulated_annealing(filtered_masks, Lc, initial_temperature, alpha, max_iterations)
        new_mask[best_merged] = cls_  # Apply the optimized merged mask
    new_mask = np.where(new_mask == 255, logits, new_mask)
    
    return new_mask

def refine_mask_with_point_labels(mask, pred, point_label):
    """
    Refine the semantic segmentation mask based on point-level supervision.
    
    Parameters:
    - mask: numpy array of shape (H, W), predicted segmentation map.
    - pred: numpy array of shape (H, W), segmentation from another method.
    - point_label: numpy array of shape (H, W), point labels (255 means unlabeled).
    
    Returns:
    - refined_mask: numpy array of shape (H, W), refined segmentation map.
    """
    
    # Create a copy to avoid modifying the original mask
    refined_mask = mask.copy()
    
    # Find locations with valid point labels (values not equal to 255)
    valid_points = np.where(point_label != 255)
    
    for y, x in zip(*valid_points):
        # Get the label class from point_label
        label_class = point_label[y, x]
        
        # If the predicted class in mask does not match the point label
        if mask[y, x] != label_class and pred[y, x] == label_class:
            
            # Get connected components in pred
            labeled_pred, num_features = ndimage.label(pred == label_class)
            
            # Get the connected region label containing the point
            region_label = labeled_pred[y, x]
            
            # Extract the connected region
            connected_region = (labeled_pred == region_label)
            
            # Overwrite this region in refined_mask
            refined_mask[connected_region] = label_class
    
    return refined_mask

def process_file(file):
    label = cv2.imread(LABEL_PATH + file + ".png")[:,:,0]
    label[label == 0] = 255
    label = label - 1
    label[label == 254] = 255

    sam = np.load(SAM_PATH + file + ".npy")
    logits = np.load(LOGITS_PATH + file + ".npy")
    logits = cv2.resize(logits, (600, 600), interpolation=cv2.INTER_NEAREST)

    sam_mask = make_pseudo(sam, logits, label)

    sam_mask = refine_mask_with_point_labels(sam_mask, logits, label)

    np.save(f"{PSEDUO_PATH}/{file}.npy", sam_mask)

def parallel_processing(file_list):
    # Use multiprocessing Pool to process files in parallel
    with Pool(processes=8) as pool:  # Adjust `processes` based on your machine's CPU
        # Display progress bar with tqdm
        for _ in tqdm(pool.imap(process_file, file_list), total=len(file_list)):
            pass

# Path to point label files
LABEL_PATH = "/home/caq/DBFNet/dataset/ICGDrone/process/train/point_label/"
# Path to generated SAM masks (in .npy format)
SAM_PATH = "/home/caq/DBFNet/dataset/ICGDrone/process/train/sam/masks/"
# First-round coarse predictions from point-supervised model
LOGITS_PATH = "/home/caq/DBFNet/dataset/ICGDrone/process/train/logits/"
# Refined pseudo-labels generated by SAM
PSEDUO_PATH = "/home/caq/DBFNet/dataset/ICGDrone/process/train/pseduo/"

f = open("datafiles/ICGDrone/pseudo_list.txt", 'r')
file_list = [x[:-1] for x in f.readlines()]

import datetime

# Program start time
starttime = datetime.datetime.now()

# Start parallel processing
parallel_processing(file_list)

# Program end time
endtime = datetime.datetime.now()

# Print runtime
print((endtime - starttime).seconds)
