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
    parser.add_argument('--root', '-R', default="./images",
                        help='directory containing images')
    parser.add_argument('--cropped', default="",
                        help='output directory for cropped images')
    args = parser.parse_args()

    # set up cascade face detector
    detector = cv2.CascadeClassifier(args.cascade)

    # load dataset
    train_set = Dataset(args.train, args.root, detector, args.cropped, 200)
    test_set = Dataset(args.test, args.root, detector, args.cropped, 200)

    # train the model
    model = archs[args.arch]()
    #print("model {}, num_components {}, eigenvalues {}".format(args.arch, recogniser.num_components, recogniser.eigenvalues))
    model.train(train_set.images, train_set.labels)

    # prediction
    for i in range(len(test_set)):
        img, label, filename = test_set.get_example(i)
        prediction, confidence = model.predict(img)
        print("{}, Truth {}, Prediction {}, Confidence: {}".format(filename, label, prediction, confidence))
        #if(label != prediction):
        #    cv2.imshow("mis-classified", img)
        #    cv2.waitKey(3000)

    cv2.destroyAllWindows()