import os
from shutil import copyfile

def save_copy(img_p, img, dest):
	img_i = Image.open(img_p+img)
	pic_i = np.array(img_i)
	
	if not mostly_blank(pic_i):
		print("saving: " + img)
		img_i.save(dest+img)
	
	img_i.close()

def main():
	path_mask = "mask/"
	output_dir = "mask_folders/"

	subdir_name = "mask_"
	dir_num = 0
	make_new_dir = 0
	curr_dir_name = output_dir+subdir_name+str(dir_num)+"/"
	
	for file in os.listdir(path_mask):
		file_name, file_extension = os.path.splitext(file)
		if file_extension != ".db":
			if make_new_dir == 0:
				curr_dir_name=output_dir+subdir_name+str(dir_num)+"/"
				os.mkdir(curr_dir_name)
				dir_num+=1
				make_new_dir = 50000
			make_new_dir -= 1
			#save_copy(path_mask, file, curr_dir_name)
			copyfile(path_mask+file, curr_dir_name+file)
				
if __name__ == '__main__':
	main()	