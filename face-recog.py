# -*- coding: utf-8 -*-
#!/usr/bin/env python
#
# @brief face detector/recogniser
# @section Requirements:  python3,  opencv3
# @version 0.01
# @date Oct. 2017
# @author Shizuo KAJI (shizuo.kaji@gmail.com)
# @licence MIT

import cv2
import numpy as np
from PIL import Image
import argparse

# class for dataset
class Dataset():
    def __init__(self, path, DataDir, detector, cropdir, crop_size=200):
        self.images = []
        self.labels = []
        self.filenames = []
        print("loading dataset from: %s"%path)
        with open(path) as input:
            for line in input:
                filename, label = line.strip().split('\t')
                im = np.array(Image.open(DataDir+"/"+filename).convert('L'), dtype=np.uint8)
                faces = detector.detectMultiScale(im)
                if (len(faces)==0):
                    print("face not found in {}".format(filename))
                for (x, y, w, h) in faces:
                    cropped_img = cv2.resize(im[y: y + h, x: x + w], (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
                    if cropdir:
                        cv2.imwrite(cropdir+"/cropped_"+filename.replace('/',''),cropped_img)
                    self.images.append(cropped_img)
                    self.labels.append(label)
                    self.filenames.append(filename)
        print("loaded: size {} number {}".format(crop_size,len(self.images)))
        self.labels = np.array(self.labels, dtype=np.int32)

    def __len__(self):
        return len(self.images)

    def get_example(self, i):
        return self.images[i],self.labels[i],self.filenames[i]


# dummy callback
def nothing(val):
    pass

def slider_callback(val):
    global mean,eigvec,slider_res,slider_num
    w = np.array([cv2.getTrackbarPos("PCA_{}".format(i), "PCA") for i in range(slider_num)])
    w = (w-int(slider_res/2))/10.0
    img = mean + (w * eigvec).sum(axis=1).reshape((args.crop_size,args.crop_size))
    img = (img-np.min(img))/np.ptp(img)
    cv2.imshow("PCA", img)

# main
if __name__ == '__main__':

    archs = {
        'eigen': cv2.face.EigenFaceRecognizer_create,
        'fisher': cv2.face.FisherFaceRecognizer_create,
        'lbph': cv2.face.LBPHFaceRecognizer_create
    }

    parser = argparse.ArgumentParser(description='face detector/recogniser')
    parser.add_argument('train', help='path to training set', default='train')
    parser.add_argument('test', help='path to test set', default='test')
    parser.add_argument('--arch', '-a', choices=archs.keys(), default='fisher',
                        help='recogniser architecture')
    parser.add_argument('--cascade', '-c', default="haarcascade_frontalface_default.xml",
                        help='path to the cascade file')
    parser.add_argument('--crop_size', '-s', type=int, default=200,
                        help='size of cropped face images')
    parser.add_argument('--num', '-n', type=int, default=5,
                        help='number of sliders for PCA face')
    parser.add_argument('--root', '-R', default="./images",
                        help='directory containing images')
    parser.add_argument('--cropped', default="",
                        help='output directory for cropped images')
    parser.add_argument('--gui','-g', action='store_true',
                        help='GUI for face generation (valid only with Eigenface)')
    args = parser.parse_args()

    # set up cascade face detector
    detector = cv2.CascadeClassifier(args.cascade)

    # load dataset
    train_set = Dataset(args.train, args.root, detector, args.cropped, args.crop_size)
    test_set = Dataset(args.test, args.root, detector, args.cropped, args.crop_size)

    # train the model
    print("training model {}...".format(args.arch))
    model = archs[args.arch]()
    model.train(train_set.images, train_set.labels)
    if args.arch in ['eigen','fisher']:
        print("num_components {}".format(model.getNumComponents()))        

    if args.gui and args.arch in ['eigen','fisher']:
        # GUI for face generation
        print("Press 'q' to exit")
        global mean, eigvec, slider_res, slider_num
        mean = model.getMean().reshape((args.crop_size,args.crop_size))/255.0
        eigvec = model.getEigenVectors()[:,:args.num]
        slider_res = 500
        slider_num = args.num
        cv2.namedWindow("PCA", cv2.WINDOW_NORMAL)
        cv2.imshow("PCA", mean)
        for i in range(args.num):
            cv2.createTrackbar("PCA_{}".format(i), "PCA", 0, slider_res, slider_callback)
            cv2.setTrackbarPos("PCA_{}".format(i), "PCA", int(slider_res/2))
        while (True):
            if cv2.waitKey(50) & 0xFF == ord("q"):  # when q key is pressed
                break
        cv2.destroyAllWindows()
        exit()

    # prediction
    print("predicting... (higher confidence means higher possibility of error)")
    for i in range(len(test_set)):
        img, label, filename = test_set.get_example(i)
        prediction, confidence = model.predict(img)
        print("{}, Truth {}, Prediction {}, Confidence: {}".format(filename, label, prediction, confidence))
        #if(label != prediction):
        #    cv2.imshow("mis-classified", img)
        #    cv2.waitKey(3000)
