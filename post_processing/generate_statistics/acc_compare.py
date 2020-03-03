import numpy as np
from PIL import Image
import os


def similar_color(color_a, color_b, range):
	diff_1 = (abs(int(color_a[0])-int(color_b[0])))
	diff_2 = (abs(int(color_a[1])-int(color_b[1])))
	diff_3 = (abs(int(color_a[2])-int(color_b[2])))
	
	return diff_1,diff_2,diff_3

def similar_seen(color, list):
	closest_color = None
	lowest_diff = None

	for c in list:
		d1,d2,d3 = similar_color(c, color, 20)
		total_diff = d1+d2+d3
		if lowest_diff == None or total_diff < lowest_diff:
			closest_color = c
			lowest_diff = total_diff
			
	return closest_color
	
def compare_accuracy(path_l, path_p, label, pred):
	img_l = Image.open(path_l+label)
	pic_l = np.array(img_l)
	
	img_p = Image.open(path_p+pred)
	pic_p = np.array(img_p)
	
	row_l, col_l, channel_l = pic_l.shape
	row_p, col_p, channel_p = pic_p.shape
	
	colors_p = []
	colors_l = []
	
	for j in range(0, col_p):
		for i in range(0, row_p):
			color_l = (pic_l[i][j][0], pic_l[i][j][1], pic_l[i][j][2])

			if color_l not in colors_l:
				colors_l.append(color_l)
	
	
	Color_accuracies = {}
	
	true_pos = 0
	false_neg = 0
	
	for j in range(0, col_p):
		for i in range(0, row_p):
			color_p = (pic_p[i][j][0], pic_p[i][j][1], pic_p[i][j][2])
			color_s = similar_seen(color_p, colors_l)
			
			color_l = (pic_l[i][j][0], pic_l[i][j][1], pic_l[i][j][2])

			key_l = str(color_l)
			
			Color_accuracies.setdefault(key_l, (0,0))
			
			true_pos, false_neg = Color_accuracies[key_l]
			
			if color_s != color_l:
				false_neg += 1
			else:
				true_pos += 1
				
			Color_accuracies[key_l] = (true_pos, false_neg)
	return Color_accuracies

def main():

	path_label = "img_label/colored/"
	path_compare = "results/test/8/he_n_eq/"
	dest = "label_pred/"
	
	label_files = []
	pred_files = []

	for file in os.listdir(path_label):
		file_name, file_extension = os.path.splitext(file)
		if file_extension == ".png":
			label_files.append(file)
			pred_files.append(file)

	pair_list = list(zip(label_files, pred_files))
	
	
	print("Comparing accuracies from: " + path_compare)
	
	Color_accuracies = {}
	
	num = 0
	for pair in pair_list:
		print(str(num) + "/" + str(len(pair_list)))
		try:
			num += 1
			acc = compare_accuracy(path_label, path_compare, pair[0],pair[1])
			
			for key in acc:
				Color_accuracies.setdefault(key, (0,0))
				true_pos, false_neg = Color_accuracies[key]
				tp, fn = acc[key]
				Color_accuracies[key] = (true_pos + tp, false_neg + fn)
		except:
			print("minor error, skipping")
			
	for key in Color_accuracies:
		true_pos, false_neg = Color_accuracies[key]
		print("Accuracy for color " + key + " was: " + str(true_pos / (true_pos + false_neg)) + " with TP " + str(true_pos) + " & FN " + str(false_neg))
	
	
		
if __name__ == '__main__':
	main()	