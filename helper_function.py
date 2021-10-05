import tensorboard as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
import pandas as pd



def view_random_image(target_dir, classes): 
  '''
  Plots images from folder to get to know the sata better
  target_dir --> directory to the master folder such as 'train' which contains class folders / class images
  classes --> list of classes 
  '''
  rand_class = random.choice(classes)
  #Set up the random directory 
  target_folder = target_dir + "/" + rand_class
  #Pick out a random image 
  random_image = random.sample(os.listdir(target_folder),1)
  #View random image: 
  img = mpimg.imread(target_folder + "/" + random_image[0])
  plt.imshow(img)
  plt.axis(False)
  plt.title("Class: " +rand_class)  
  
  
