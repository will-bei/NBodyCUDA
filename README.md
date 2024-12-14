# NBodyCUDA
Source files for CUDA implementation and host-side verification of a 2-D N-Body Simulation, made for the final project for the class EE 524 P Au24 GPU Computing.

# Prerequisite
Install and configure ffmpeg following the instructions in this link: https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/#

# Instructions for running host-side verification code
Run the code
    - Use HostSideNBody3.cpp, an updated version that has more accurate modelling and uses normal distribution initialization
Open command prompt and cd into the output directly where the generated images are kept
Convert the images into a gif via the following command: ffmpeg -framerate 25 -i %07d.bmp -c:v libx264 -r 50 cpuOut1.mp4
    - Note: the resulting .mp4 file may not work on the default Windows media player, VLC player is recommended

# Example output

https://drive.google.com/file/d/1X9cBlXzyAuK_9LTDH3F2OeaEU-lOEU8_/view?usp=drive_link
