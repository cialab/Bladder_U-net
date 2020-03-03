import os
import time

directory = "val_tiles_predict/"
dirs = next(os.walk(directory))[1] #[x[0] for x in os.walk(directory)]
num_items_before = 0
for dir in dirs:
	for file in os.listdir(directory+dir):
		num_items_before += 1


start = time.time()
num_items_after = 0
for dir in dirs:
	for file in os.listdir(directory+dir):
		num_items_after += 1
end = time.time()

tile_diff = num_items_after - num_items_before
time_diff = end - start
print("completed: " + str(tile_diff) + " items in " +  str(time_diff) + " seconds")

tiles_per_min = tile_diff * (60/time_diff)

print("Rate of completion is: " + str(tiles_per_min) + " tiles/min")