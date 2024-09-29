import os
import zipfile
import shutil

def unzip_directory(zip_file_path, extract_to_dir):
    # Handle long paths on Windows by using \\?\ prefix
    if os.name == 'nt':
        zip_file_path = '\\\\?\\' + os.path.abspath(zip_file_path)
        extract_to_dir = '\\\\?\\' + os.path.abspath(extract_to_dir)

    # Check if the directory exists, if not, create it
    if not os.path.exists(extract_to_dir):
        os.makedirs(extract_to_dir)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # Extract all files
        for member in zip_ref.namelist():
            filename = os.path.basename(member)
            # Skip directories (they will be created automatically)
            if not filename:
                continue

            # Remove the folder prefix (assuming there's only one folder in the zip)
            extracted_path = os.path.relpath(member, start=zip_ref.namelist()[0].split('/')[0])
            target_file = os.path.join(extract_to_dir, extracted_path)

            # Ensure the target directory exists
            target_dir = os.path.dirname(target_file)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            with zip_ref.open(member) as source, open(target_file, 'wb') as output_file:
                shutil.copyfileobj(source, output_file)

        print(f"Successfully extracted {zip_file_path} to {extract_to_dir}")

# Example usage
zip_file_path = r'path/to/zip'
extract_to_dir = r'output/directory'
unzip_directory(zip_file_path, extract_to_dir)
