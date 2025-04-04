import os
import json

def pack_foldernames_to_json(directory, output_file):
    """
    Collects all folder names in the specified directory and saves them as a JSON list.

    Args:
        directory (str): The path to the directory containing the folders.
        output_file (str): The path to the output JSON file.
    """
    # Ensure the directory exists
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return

    # Collect all folder names in the directory
    folder_names = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]

    # Save the folder names to a JSON file
    with open(output_file, 'w') as json_file:
        json.dump(folder_names, json_file, indent=4)

    print(f"Folder names have been saved to '{output_file}'.")

# Example usage
if __name__ == "__main__":
    directory_path = "dataset/Objaverse/rendered_copy"  # Replace with your directory path
    output_json = "dataset/Objaverse/rendered_copy/train_uids.json"  # Replace with your desired output file name
    pack_foldernames_to_json(directory_path, output_json)