-------------------------------
-- 540438 - Maurizio La Rosa --
-------------------------------

This repository hosts the project for the exam of the course Devices and circuits for artificial intelligence from the Data Analysis degree of the Universityof Messina. The project consists in building a machine learning model for image classification.

The dataset to be used is hosted at kaggle, at the link https://www.kaggle.com/datasets/gpiosenka/100-bird-species and currently contains images for 525 bird species to be classified by the model. I downloaded the dataset on April, 17th, 2023, and that version contains 515 bird species.

It is useful to note that images in the dataset should have all the exact same shape (224, 224, 3), while I found that all images of the 'PLUSH CRESTED JAY' species and one image from the 'DON'T REMEMBER THE SPECIES, FILL INFO WITH FUTURE COMMIT' species have variable shapes. Hence, in my code, I check for images' shapes and remove images that don't match the common shape. This is important because imported images have the shape of 3D Numpy (np, when imported) arrays and I need to transform the list of images into a 4D Numpy array. The function np.array() can do it automatically when fed a list of 3D Numpy arrays, but images must have all the same shape.
