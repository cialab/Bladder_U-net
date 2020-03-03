import os
from shutil import copyfile


def main():
	path_mask = "mask_tiles/"
	output_dir = "val_mask_tiles/"

	val = ['Case-070 B', 'd000112-02', 'Case-212 B', '13']

	for file in os.listdir(path_mask):
		file_name, file_extension = os.path.splitext(file)
		if file_extension != ".db":
			name_data = file_name.split('-')
			main_pic_name_list = name_data[:-2]
			main_pic_name = '-'.join(main_pic_name_list)
			
			if main_pic_name in val:
				copyfile(path_mask+file, output_dir+file)
				
if __name__ == '__main__':
	main()	