This is a directory containing the code produced by Enes Yazgan
in the summer of 2019, for the purpose of identifying different tissue
layers in bladder biopsy slides using U-Nets

Directories:
NOTE: 	Each file is single_purpose, and written in a straightforward way. Reading through
		a .py file will tell you what I do and why. If you have any questions, please email:
		enes32y@gmail.com


- image_preprocessing
	This is where annotated bladder slide images get prepared
	
	NOTE: The bladder slides are found on Langley/original/cialab/image_database
	They are saved as .svs files, and their labels are .xml. For my work, I used
	the .svs files with a low amount of layers (each .svs file is actually a multi-layers
	.tiff file), so converting the file extension from .svs to .tiff retained image data
	and was workable. However, you may find some .svs files get corrupted when converted
	directly to .tiff. To remedy this issue, open the .svs file on Aperio ImageScope,
	and on the top bar, towards the right, there will be a region selector with a save icon
	on it. This will allow you to extract a region from the .svs file as an original
	resolution .tiff file. To extract an entire bladder slide image, just type in the 
	dimsions of the image (can be found on the bottom right of ImageScope) to the window that
	pops up when you select the extract region tool. Note that Python solutions exist to open
	multi-layer tiff files, however, these .svs files seem to have something wrong with the
	file descriptor, so the solutions online may not work as expected.
	
	- image_and_mask_generation/
		NOTE: Please look at the directories being used in the following files to get an
		understanding of which directories are used for what purpose. Note that you can 
		change them if you wish.
		
			- getMasks.m : 		Takes in a .tiff file from image/ and a .xml file from label/
								and returns grayscale mask tiles of size 512x512 into mask/
							
								NOTE: these directories are passed in when calling the method,
								so you can choose your own directories to use. I have included
								some .tiff files and .xml files in their proper directories
								for you to test it out
							
			- tileImages.m :	Uses the same logic in getMasks.m to take full .tiff images from
								image/ and separate them into 512x512 tiles and save them to
								image_tiles/
								
			- mask_overlay_ontiles_multiproc.py : 	This program takes tiles from image_tiles/ and
													mask/ and saves tiles to mask_tiles_overlay/
													This program essentially labels unlabelled tissue
													IMPORTANT NOTE: It is no longer necessary, because
													we now generate our dataset with unlabelled
													regions completely removed, as they confuse the
													U-Net
											   
			- remove_unlabelled_areas_multiproc.py: 	This program takes tiles from image_tiles/ and
														mask/ and saves tiles to image_tiles_labelsonly/
														and mask_tiles_labelsonly/ 
														It removes unlabelled regions from both the image
														and mask. It also filters out mostly empty tiles
													
			NOTE:	For files ending in _multiproc, I used multiprocesing to speed the code up
					However, after opening a certain amount of images, the program crashes.
					Waiting for completion when you hit this limit essentially slows the program
					back down to original speeds. To negate this issue, I split the images/masks
					into y pieces, and just run the program on part x. You can run this back to back
					on the same node, or run on different parts on different nodes to speed
					the process up even more. Note that right now you have to edit the use_part
					variable inside the .py code, but you can change it to a commandline argument
					if you want.
					
			- create_train_data.py : 	Saves tiles into train/val/test folders to create dataset
										Make sure you use the same file structure seen in Bladder/val
										in the U-Net_Model folder
					
			- misc_debug_code/ : 		Title says all. Contains some debugging code.
			- tissue_representation/ : 	Contains code that generates an augmented version of the dataset,
										where tiles containing underrepresented tissue are flipped/rotated
										and zoomed to increase the instances they are seen in
										
			- utility_code/:
				- quick_convert.py: 	Changes file extensions of files in a folder
				- quick_matlab.py:		Prints out the commands for getMasks.m, so you can just copy/paste
										them into your terminal to run them back to back. Nice to use
										for sorting into groups and running on separate nodes
				- quick_matlab_tiles.py:	Same thing but for tileImages.m
				- see_masks.py: 			See which images you've made into masks, 
											and which ones are left to do
				- sort_masks.py:			Sorts masks into folders based on what image they come from
				- tile_gen_img_only.py:	Old code that uses Python to tile an image. May be useful for smaller
										images, but unfortunately large images will break it (PIL limitation)
			
	
	- inputs/ : 	This folder contains various .tiff files of bladder slides, as well as their 
				corresponding .xml files. There are also some .png images of masks generated
				from some of these .tiff & .xml files. Note that these are just for practice
				uses; the masks were created in an outaded fashion. You can, however,
				run the .tiff &.xml files on the up to date mask generation code found 
				in image_and_mask_generation
				
				It also contains the following, useful files:
				- val/ : contains unlabelled slide .tiff
				- etc/ : same, but contains the super-large ones that aren't worth using
				- train/ : 	contains the .tiff files I used for my training dataset (39 slides). 
							This is also found in image_and_mask_generation/
				- tile_Images.m : same as in image_and_mask_generation/
				- remove_gigantic.py : 	saves tiles with tissue present to a new folder. These tiles
										are what out model will perform predictions on to send to
										a pathologist. We don't need the mostly empty slides because
										we know that the modle will just predict them as background anyway
										so we don't need to waste time on those
										NOTE: This program also divides list into sublists to enable running
										on multiple nodes, but if you want you can just use the whole list
				
- U-Net_model/
	This folder contains the meat of the U-Net code. This is where training and predictions occur

	NOTE: it's a bit confusing, but also too late to change, but the terminology used 
	in this folder uses 'test' to denote the validation set (gets held out and checked
	against during training) and 'val' denotes the test set that the model runs predictions
	on after training. Feel free to change this.

	- Bladder/ : 				Directory containing test/train/val tiles
	
	- Bladder_equal/ :			Augmented version of the dataset in Bladder,
								tiles with underrespresented tissue types have been
								augmented to be seen more in this dataset
								
	- good_Hdf5/ :				Contains some saved .hdf5 models
	
	- npydata/ :					Contains npy data produced by data_bladder and unet_bladder

	- data_bladder.py : 		takes images & masks from the Bladder directory
								and saves training/validation/testing samples
								into their own .npy files, stored in npydata
								Used to prepare data for training/validation/testing
								
	- data_bladder_equal : 		same as data_bladder, just uses Bladder_equal
	
	- *+'test'+*.py : 			anything with test in it was used to test smaller datasets
								before beginning a long training session
							
	- predict_bladder_*.py :	performs predictions on the tiles in Bladder_val/
								with the corresponding model. Results are saved to
								results_val/[corresponding folder]
								
	- unet_bladder_*.py	:		Trains corresponding U-Net and saves results of the
								predictions made on the tiles in Bladder/val to
								results/[corresponding folder]
								
	- test2mask2pic.py : 		Helper function used in data_bladder

- post_processing/
	This folder contains code that operates on the outputs of our model
	
	NOTE: 	The purpose of this folder is to get a set of prediction tiles from the U-Net, and
			then stitch it back to the same size as its original bladder slide image. So let'same
			say we predicted on some tiles from 04.tiff, we would use 04.tiff as well as the
			few hundred prediction tiles generated as inputs, and we would output an 04.png
			of the same dimensions as 04.tiff, that contained the predictions made by the U-Net
			in their correct positions. The result is a per-pixel labelled version of the original
			slide.
	
	- create_fullsize_prediction/
		-test_in/
			-image_pred/: 	contains the .tiff file you have the predictions pf
			-pred_all/: 	contains all the other .tiff files
		-test_out/
			-image_pred/: 	contains the predictions you want to stitch together
		-test_stitched/:
			-image_pred/:	contains full resolution stitched prediction
			-preview_pred/:	contains 4000 * (4000*aspectratio) downscaled version of stitched prediction
							(most images are too large to be viewed, so this is useful for quick observation)
			-thumb_pred/:	contains a tiny thumbnail version of the stitched prediction
			
		
	- generate_statistics/	