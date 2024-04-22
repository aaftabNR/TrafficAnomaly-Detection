import h5py

def merge_hdf5_files(file1_path, file2_path, merged_file_path):
    # Open the first HDF5 file for reading
    with h5py.File(file1_path, 'r') as file1:
        # Open the second HDF5 file for reading
        with h5py.File(file2_path, 'r') as file2:
            # Create a new HDF5 file for writing (or overwrite if it already exists)
            with h5py.File(merged_file_path, 'w') as merged_file:
                # Copy contents from the first file to the merged file
                for name, item in file1.items():
                    file1.copy(name, merged_file)
                # Iterate over the second file and copy its contents to the merged file
                for name, item in file2.items():
                    # Check if the item already exists in the merged file
                    if name in merged_file:
                        # If the item already exists, create a unique name for the copied item
                        new_name = name + '_from_file2'
                        file2.copy(name, merged_file, new_name)
                    else:
                        # Copy the item from the second file to the merged file
                        file2.copy(name, merged_file)

# Paths to the input HDF5 files
file1_path = r'D:\PBL2_project\Code\my_model.h5'
file2_path = r'D:\PBL2_project\Code\AccidentDetectionModel.h5'

# Path to the merged HDF5 file
merged_file_path = r'D:\PBL2_project\Code\merged_file.h5'

# Merge the HDF5 files
merge_hdf5_files(file1_path, file2_path, merged_file_path)

print("Merge operation completed successfully.")
