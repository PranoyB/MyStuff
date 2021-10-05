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
  
def create_tensorboard_callback(dir_name, experiment_name):
  '''
  dir_name --> where to save this
  experiment_name --> what to name it as
  '''
  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%n%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
  print(f"Saving TensorBoard log files to: {log_dir}")
  return tensorboard_callback

  
def sample():
  print("hello")
