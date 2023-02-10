import os
import shutil

def remove_colmap(basedir):
    remove_list = ['database.db', 'colmap_output.txt', 'poses_bounds.npy', 'sparse']

    for idx in remove_list:
        file_path = os.path.join(basedir, idx)
        if os.path.isdir(file_path) and not os.path.islink(file_path):
            print(f'removing {file_path}')
            shutil.rmtree(file_path)
        elif os.path.exists(file_path):
            print(f'removing {file_path}')
            os.remove(file_path)


if __name__ == "__main__":
    print('REMOVE COLMAP Files')
    basedir = '/home/brozserver3/brozdisk/data/nerf/colmap_test_data/llff/fern'
    remove_colmap(basedir)
    

