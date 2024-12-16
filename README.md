This is a training model for identification of hand tools and IDs with OCRing for running in a Raspberry PI.

Use ./run.sh on the docker folder to download and use the docker image.

After downloaded you'll be brought to a docker file terminal.

You'll need to create a 'tools' folder in the '/python/training/classification/data' folder.

For that, use 'cd python/training/classification/data' and then 'mkdir tools'. Then, 'cd' into Tools by using 'cd tools' and use 'touch labels.txt' to create a 'labels.txt' file.

Use 'nano labels.txt' to edit the txt file, and write the classes you'll need in alphabetical order, one per line. It'll look like this, for example:

background
hammer
measuring_tape
pliers
screwdriver

Use 'Ctrl+O' to save the file and 'Ctrl+X' to exit.

Now, in the terminal, use 'camera-capture /dev/video0' to initialize the image capturing program. Set the Dataset Path to '/jetson-inference/python/training/classification/data/tools'
and the Class Labels to '/jetson-inference/python/training/classification/data/tools/labels.txt'. Your program should recognize all the classes and the three sets automatically.

I've used 200 train images for each class, and 30/30 images for val and test. The result was really good. When you're done, exit the program by closing the Data Capture Control.

On the '/jetson-inference/python/training/classification' folder, run 'python3 train.py --model-dir=models/tools --batch-size=8 --workers=1 --epochs=50 data/tools/'.

'model-dir' is the directory the model will be saved. 'batch-size' is the group of image that will be tested and trained together. Sometimes lower batch-sizes are better, sometimes higher batch-sizes are better.

You'll have to experiment, but for this case I've found 'batch-size=8' works really well. I used 'epochs=50' and the result was good, but you can mess with this parameter too. Higher 'epochs' mean your model will
train more on the same dataset, lower 'epochs' mean your model will train less. Finally, 'data/tools/' is the folder your dataset is in, including the 'labels.txt' file.

This will take some time. When it's over, you should run 'python3 onnx_export.py --model-dir=models/tools'. This will generate a .onnx file. .onnx models make the inference faster and more accurate, and it also
works with many deep learning frameworks, like Tensorflow and PyTorch.

To see your results, run 'imagenet --model=models/tools/resnet18.onnx --labels=data/tools/labels.txt --input_blob=input_0 --output_blob=output_0 /dev/video0'. '/dev/video0' is your mounted camera.

Now, the docker image will launch a camera visualization of the model.

Don't forget to save your files to your computer by running 'sudo docker ps' to visualize the docker image then run 'sudo docker cp <your_docker_image_number>:/jetson-inference /home/<your_username>/jetson_inference_backup '.

Docker image files are sometimes not persistent, so you should save your files to your computer so you don't lose your dataset!

With this, you created your dataset and made inference upon it. Thanks for reading.

EDIT: containers folder isn't being uploaded, probably because of git commands used by Dustin. For now, use 'git clone https://github.com/dusty-nv/jetson-containers.git -b bc8d0264ef25aa0d1d25a54e4658f491d2fa130f --single-branch'
to fetch the 'containers' folder from the original repository (nothing was changed here).

