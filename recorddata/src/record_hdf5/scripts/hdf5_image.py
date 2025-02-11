# #读取一个hdf5文件
# #查看hdf5文件的文件结构
import h5py
import cv2
import matplotlib.pyplot as plt


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
file_path = '/home/robot/imitate_ws/src/record_hdf5/scripts/utils/data/collection_data/hdf5/humanplus_hdf5/episode_6.hdf5'
# file_path = '/home/moning/dataset/zhuawawa_241224_hdf5/demonstration_22.hdf5'
# read_hdf5_file(file_path)            
camera_names = ['cam_front']
image_dict = dict()
with h5py.File(file_path, 'r') as root:
    for cam_name in camera_names:
        image_dict[cam_name] = root[f'/observations/images/{cam_name}'][5]


    for cam_name in image_dict.keys():
        decompressed_image = cv2.imdecode(image_dict[cam_name], 1)
        print(image_dict[cam_name])#[255 255 255 ...   0   0   0]
        print(decompressed_image)#None
        plt.imshow(decompressed_image)
        plt.show()

                    # if self.width is not None and self.height is not None:
                    #     # print(image_dict[cam_name])#这里为什么全0
                    #     # print(decompressed_image)
                    #     decompressed_image = cv2.resize(decompressed_image, (self.width, self.height), interpolation=cv2.INTER_AREA)
                    # image_dict[cam_name] = np.array(decompressed_image)
                # if self.feature_loss:
                #     for cam_name in image_dict_future.keys():
                #         decompressed_image = cv2.imdecode(image_dict_future[cam_name], 1)
                #         if self.width is not None and self.height is not None: 
                #             decompressed_image = cv2.resize(decompressed_image, (self.width, self.height), interpolation=cv2.INTER_AREA)
                #         image_dict_future[cam_name] = np.array(decompressed_image)
            
            
