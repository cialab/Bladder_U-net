import os
from shutil import copyfile


def main():
	
	images = "image_tiles/"
	images_labelled = "image_tiles_labelsonly/"
	
	out_dir = "04/"
	
	good_images = []
	
	"""
	for file in os.listdir(images_labelled):
		file_name, file_extension = os.path.splitext(file)
		if file_extension != ".db":
			
			name_data = file_name.split('-')
			main_pic_name_list = name_data[:-2]
			main_pic_name = '-'.join(main_pic_name_list)
			
			if main_pic_name == "04":
				good_images.append(file_name)
	"""		
	
	for file in os.listdir(images):
		file_name, file_extension = os.path.splitext(file)
		if file_extension != ".db":
			name_data = file_name.split('-')
			main_pic_name_list = name_data[:-2]
			main_pic_name = '-'.join(main_pic_name_list)
			
			if main_pic_name == "04":
				copyfile(images+file, out_dir+file)
				
if __name__ == '__main__':
	main()	