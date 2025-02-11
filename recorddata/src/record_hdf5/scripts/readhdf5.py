# #读取一个hdf5文件
# #查看hdf5文件的文件结构
import h5py


def read_hdf5_file(file_path):
    # 打开HDF5文件
    with h5py.File(file_path, 'r') as hdf:
        # 打印文件中的所有组和数据集
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name}, Shape: {obj.shape}, Data: {obj[:]}")#
            elif isinstance(obj, h5py.Group):
                print(f"Group: {name}")

        hdf.visititems(print_structure)


# 示例用法
file_path = '/home/moning/dataset/HDFView-3.3.2App-win64/episode_3.hdf5'
file_path = '/home/robot/imitate_ws/src/record_hdf5/scripts/utils/data/collection_data/hdf5/humanplus_hdf5/episode_6.hdf5'
read_hdf5_file(file_path)

# import h5py
#
#
# def print_structure(name, obj):
#     """回调函数，用于打印对象的路径和类型"""
#     if isinstance(obj, h5py.Dataset):
#         obj_type = "Dataset"
#     elif isinstance(obj, h5py.Group):
#         obj_type = "Group"
#     else:
#         obj_type = "Unknown"
#
#     print(f"Path: {name}, Type: {obj_type}")
#
#
# # 打开一个现有的HDF5文件
# file_path = '/home/moning/dataset/HDFView-3.3.2App-win64/episode_1.hdf5'
# file_path = '/home/moning/dataset/zhuawawa_241224_hdf5/demonstration_1.hdf5'
#
# with h5py.File(file_path, 'r') as hdf:
#     # 使用visititems方法遍历文件中的所有对象
#     hdf.visititems(print_structure)