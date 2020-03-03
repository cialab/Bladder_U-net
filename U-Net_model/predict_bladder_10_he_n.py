# -*- coding:utf-8 -*-

from unet_bladder_10_he_n import *
from data_bladder import *

myunet = myUnet()
model = myunet.get_unet()
model.load_weights('unet_bladder_10_he_n.hdf5')

# test2mask
imgs_test = myunet.load_val_data()
imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
np.save('./results_val/10/he_n/bladder_mask_test.npy', imgs_mask_test)

# mask2pic
myunet.save_val_img()
