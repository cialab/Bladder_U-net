import os
from shutil import copyfile
import random

def main():
	main_path = "Bladder/"
	m_tr = "train/"
	m_trn = "trainannot/"
	m_ts = "test/"
	m_tsn = "testannot/"
	m_v = "val/"
	m_vn = "valannot/"
	
	trainset_path = "Bladder_test/"
	tr = "train/"
	trn = "trainannot/"
	ts = "test/"
	tsn = "testannot/"
	v = "val/"
	vn = "valannot/"
	
	"""
	for file in os.listdir(main_path+m_v):
		file_name, file_extension = os.path.splitext(file)
		if file_extension != ".db":
			rand = random.randint(1,101)
			if rand <= 15:
				copyfile(main_path+m_v+file, trainset_path+v+file)
				copyfile(main_path+m_vn+file, trainset_path+vn+file)
	
	"""
	#remove some:
	for file in os.listdir(trainset_path+m_v):
		file_name, file_extension = os.path.splitext(file)
		if file_extension != ".db":
			rand = random.randint(1,101)
			if rand <= 90:
				os.remove(trainset_path+v+file)
				os.remove(trainset_path+vn+file)
		
if __name__ == '__main__':
	main()	