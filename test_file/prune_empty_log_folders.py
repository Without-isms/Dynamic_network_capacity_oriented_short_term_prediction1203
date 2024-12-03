import os
import shutil
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.join(current_dir, f'..{os.path.sep}data{os.path.sep}checkpoint{os.path.sep}sz')

three_days_in_seconds = 0 * 24 * 60 * 60
current_time = time.time()

for root, dirs, files in os.walk(base_path):
    for dir_name in dirs:
        dir_path = os.path.join(root, dir_name)
        files_in_dir = os.listdir(dir_path)
        if len(files_in_dir) == 1 and files_in_dir[0] == "info.log":
            folder_mtime = os.path.getmtime(dir_path)
            if current_time - folder_mtime > three_days_in_seconds:
                shutil.rmtree(dir_path)
                print(f"Deleted folder: {dir_path}")

print("Folders containing only 'info.log' and older than three days have been deleted.")
