import os

path_img = "images_tif/"
path_mask = "mask/"#"mtlb_images_tiled/"

seen_names = []
seen_times = {}
todo_names = []


for file in os.listdir(path_img):
	file_name, file_extension = os.path.splitext(file)
	if file_extension == ".tiff":
		if file_name not in todo_names:
			todo_names.append(file_name)
			print("Have to tile: " + file_name)

for file in os.listdir(path_mask):
	file_name, file_extension = os.path.splitext(file)
	if file_extension == ".png":
		name_data = file_name.split('-')

		main_pic_name_list = name_data[:-2]
		main_pic_name = '-'.join(main_pic_name_list)
		
		if main_pic_name not in seen_names:
			seen_names.append(main_pic_name)
			print("Finished tiling: " + main_pic_name)
		else:
			seen_times.setdefault(main_pic_name, 0)
			seen_times[main_pic_name] += 1

for name in todo_names:
	if name not in seen_names:
		print("Still need to tile: " + name)
		
for name in todo_names:
	if name in seen_names:
		print(name + " generated: " +str( seen_times[name]) + " mask tiles")