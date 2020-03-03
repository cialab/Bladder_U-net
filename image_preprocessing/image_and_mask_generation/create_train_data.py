import os
from shutil import copyfile
import random

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
	


def main():
	path_mask = "mask_aug/"
	path_img = "img_aug/"
	
	trainset_path = "Bladder_equal/"
	tr = "train/"
	trn = "trainannot/"
	ts = "test/"
	tsn = "testannot/"
	v = "val/"
	vn = "valannot/"
	
	"""
	num_tiles = {}
	
	for file in os.listdir(path_mask):
		file_name, file_extension = os.path.splitext(file)
		if file_extension != ".db":
			name_data = file_name.split('-')
			main_pic_name_list = name_data[:-5]
			main_pic_name = '-'.join(main_pic_name_list)
	
			num_tiles.setdefault(main_pic_name, 0)
			num_tiles[main_pic_name] += 1
	
	total_tiles = 0
	for key in num_tiles:
		print("Image: " + key + " generated: " + str(num_tiles[key]) + " tiles")
		total_tiles += num_tiles[key]
		
	print("total_tiles: " + str(total_tiles))
	"""
	
	
	val = ['Case-070 B', 'd000112-02', 'Case-212 B', '13']
	test = ['d000112-07', 'Case-164 B', '12', 'Case-111 B']
	
	masks = []
	
	for file in os.listdir(path_img):
		file_name, file_extension = os.path.splitext(file)
		if file_extension != ".db":
			masks.append(file)
		
	mask_parts = list(split(masks, 3))
	use_part = 2
		
	print("using part: " + str(use_part))
		
	for file in mask_parts[use_part]:
		file_name, file_extension = os.path.splitext(file)
		if file_extension != ".db":
			name_data = file_name.split('-')
			main_pic_name_list = name_data[:-5]
			main_pic_name = '-'.join(main_pic_name_list)
			
			if main_pic_name in test:
				copyfile(path_img+file, trainset_path+ts+file)
				copyfile(path_mask+file, trainset_path+tsn+file)
			elif main_pic_name in val:
				copyfile(path_img+file, trainset_path+v+file)
				copyfile(path_mask+file, trainset_path+vn+file)
			else:
				copyfile(path_img+file, trainset_path+tr+file)
				copyfile(path_mask+file, trainset_path+trn+file)
	
if __name__ == '__main__':
	main()	