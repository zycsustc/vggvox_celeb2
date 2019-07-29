# vggvox_celeb2
An keras version of vggvox_celeb2 model for identification and sound character
## 1. Can only take in .wav files longer than 5 seconds !!!
## 2. You need to assign input_file_path, output_file_path, and the name of the model after you initialized the vggvox_celeb2 model using vggvox_init() function in vggvox_celeb2.py file, remember to import that .py file in your main file.
## 3. How to use:
### First: import vggvox_celeb2 as vg2
#### Notice: You have to establish and save the model first if you haven't save it before, just use vg2.vggvox_init(already_saved = False), the function will automatically create and save the model as vggvox_celeb2.h5.
### Second: model_name = vg2.vggvox_init(already_saved=True/False)
### Third: output_path_saved = vg2.speaker_character(input_file_path, output_file_path, model_name)
### If you want to use a single .wav file path as input and got the z vector as an array, do as following:
### First: import vggvox_celeb2 as vg2
### Second: model_name = vg2.vggvox_init(already_saved=True/False)
### Third: output_z_array = vg2.speaker_single(input_file_path)
### Now enjoy the .json file which contains all the sound characters！！！`
