Face detection and recognition using OpenCV
=============
A python demo code for face detection and recognition in images using OpenCV3
by Shizuo KAJI

## Licence
MIT Licence

# Requirements
- python 3.6

Install [Anaconda](https://www.anaconda.com/download/)
 if you do not have python 3 on your system.

- OpenCV 3.3
-- macOS: use homebrew (e.g., `brew install opencv`)
-- Windows: download the binary package
(e.g. opencv_python‑3.3.0‑cp36‑cp36m‑win_amd64.whl)
from [here](http://www.lfd.uci.edu/%7Egohlke/pythonlibs/#opencv)
and install with 
`pip install opencv_python‑3.3.0‑cp36‑cp36m‑win_amd64.whl`

# Example
Let's try with the famous AT&T dataset.
Download the dataset from
http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
and extract.

`python face-recog.py -h`
gives a brief description of command line arguments

`python face-recog.py train.txt test.txt -a fisher -R att_faces`
reads training images from train.txt and outputs prediction for images in test.txt
using the Fisherface algorithm.
Each line of train.txt consists of two entries separated by a tab;
the relative path to an image file from the directory specified by -R (att_faces in this example),
and the label indicating who is in the picture.

`python face-recog.py train.txt test.txt -a eigen -R att_faces —cropped cropped`
outputs prediction using Eigenface and saves cropped and resized face images 
to the directory specified by --cropped (in this example, cropped). The output directory must exist in advance.



