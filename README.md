# NBodyCUDA
Source files for CUDA implementation and host-side verification of a 2-D N-Body Simulation, made for the final project for the class EE 524 P Au24 GPU Computing.

# Prerequisite
Install and configure ffmpeg following the instructions in this link: https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/#

# Instructions for running host-side verification code
Run the code
Open command prompt and cd into the output directly where the generated images are kept
Convert the images into a gif via the following command: ffmpeg -framerate 24 -i %05d.bmp cpuOut.gif