import os
import shutil

def validate_and_clean_folders(base_path):
    """
    Validates that each folder in the base path contains the required structure.
    Deletes folders that do not match the required structure.

    Args:
        base_path (str): The path containing the hash folders to validate.
    """
    required_structure = ["pose", "rgba", "intrinsics.npy"]

    # Iterate through all items in the base path
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)

        # Skip if it's not a directory
        if not os.path.isdir(folder_path):
            continue

        # Check if the folder contains the required structure
        has_required_structure = True
        for item in required_structure:
            item_path = os.path.join(folder_path, item)
            if item == "pose" or item == "rgba":
                if not os.path.isdir(item_path):  # Check if it's a directory
                    has_required_structure = False
                    break
            elif item == "intrinsics.npy":
                if not os.path.isfile(item_path):  # Check if it's a file
                    has_required_structure = False
                    break

        # If the folder does not match the required structure, delete it
        if not has_required_structure:
            print(f"Deleting folder: {folder_path}")
            shutil.rmtree(folder_path)

if __name__ == "__main__":
    # Replace with the path to your base directory
    base_directory = "dataset/Objaverse/rendered_copy"
    validate_and_clean_folders(base_directory)