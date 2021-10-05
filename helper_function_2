import tensorboard as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
import pandas as pd

def view_random_image(target_dir, classes): 
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

  
def sample():
  print("hello")
