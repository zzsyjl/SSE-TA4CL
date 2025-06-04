import os
import shutil
from pathlib import Path

def create_split_folders(base_path, split_files, output_base_path):
    """
    Split NWPU-RESISC45 dataset into train/val/test folders based on split files.
    
    Args:
        base_path: Path to the original NWPU-RESISC45 folder
        split_files: Dictionary containing paths to split text files
        output_base_path: Path where split folders will be created
    """
    # Create output directories for each split
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(output_base_path, split)
        if not os.path.exists(split_path):
            os.makedirs(split_path)
            
        # Read the corresponding split file
        with open(split_files[split], 'r') as f:
            image_list = f.read().splitlines()
            
        # Process each image in the split
        for img_name in image_list:
            # Extract class name from image name (assuming format: class_name_number.jpg)
            class_name = '_'.join(img_name.split('_')[:-1])
            
            # Create class directory in split folder if it doesn't exist
            class_path = os.path.join(split_path, class_name)
            if not os.path.exists(class_path):
                os.makedirs(class_path)
                
            # Source and destination paths
            src_path = os.path.join(base_path, class_name, img_name)
            dst_path = os.path.join(class_path, img_name)
            
            # Copy the image
            try:
                shutil.copy2(src_path, dst_path)
                print(f"Copied {img_name} to {split}/{class_name}/")
            except FileNotFoundError:
                print(f"Warning: Could not find {src_path}")

def main():
    # Define paths
    base_path = "/home/jinglong/data/resisc45/NWPU-RESISC45"  # Replace with your dataset path
    output_base_path = "/home/jinglong/data/resisc45"   # Replace with your desired output path
    
    # Define split files
    split_files = {
        'train': '/home/jinglong/data/resisc45/resisc45-train.txt',
        'val': '/home/jinglong/data/resisc45/resisc45-val.txt',
        'test': '/home/jinglong/data/resisc45/resisc45-test.txt'
    }
    
    # Create split folders
    create_split_folders(base_path, split_files, output_base_path)
    print("Dataset splitting completed!")

if __name__ == "__main__":
    main()