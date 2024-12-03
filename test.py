import os

def count_python_lines(directory, exclude_dirs=None):
    total_lines = 0
    exclude_dirs = exclude_dirs or []

    # Traverse the directory
    for root, dirs, files in os.walk(directory):
        # Skip directories that are in the exclude list
        dirs[:] = [d for d in dirs if os.path.join(root, d) not in exclude_dirs]

        # For each file in the directory
        for file in files:
            # Check if it is a Python file
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    total_lines += len([line for line in lines if line.strip()])  # Count non-empty lines

    return total_lines

# Example usage
project_directory = r"C:\Users\SuperDoctorCat\PycharmProjects\Dynamic_network_capacity_oriented_short_term_prediction"
excluded_directories = [
    os.path.join(project_directory, 'archieve'),
    os.path.join(project_directory, 'SOTA'),
]

total_lines = count_python_lines(project_directory, excluded_directories)
print(f'Total lines of Python code: {total_lines}')
