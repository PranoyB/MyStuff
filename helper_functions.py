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

def plot_losses(model_history):
  ''' 
  Plots the loss curves and accuracy curves given a model hisotry object
  '''
  history = pd.DataFrame(model_history.history)
  plt.figure(figsize=(7,5))
  plt.plot(history['loss'],label='Training Loss')
  plt.plot(history['val_loss'],label='Validation Loss')
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.grid(True)
  plt.legend()
  plt.figure(figsize=(7,5))
  plt.plot(history['accuracy'],label='Training Accuracy')
  plt.plot(history['val_accuracy'], label = 'Validation Accuracy')
  plt.xlabel("Epochs")
  plt.ylabel("Accuracy")
  plt.legend()
  plt.grid(True)

  
def unzipper(link,folder_name):
  ''' 
  Given the link to download the ziped file from, it saves the unziped version with foldername in the working directory
  '''
  !wget link
  !unzip name+".zip"
  
  
def walk_through_dir(foldername):
  '''
  given the folder name constaining training and tresting data, we can get some more data
  '''
  for dirpath, dirnames, filenames in os.walk(foldername):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'")
