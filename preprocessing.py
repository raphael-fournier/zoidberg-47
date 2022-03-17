# Standard scientific Python imports
from matplotlib import pyplot
from pprint import pprint
import numpy, os
from PIL import Image
from glob import glob

# Import datasets, classifiers and performance metrics
# from sklearn.svm import SVC

class Dataset:
    def __init__(self, path):
        self.data = [];
        self.labels = []
        self.name = os.path.basename(path)
        for i in range(len(groups)):
            pprint(path + '/' +groups[i]+ '/*.jpeg')
            for filename in glob(path + '/' +groups[i]+ '/*.jpeg'):
                image = XRay(filename)
                self.data.append(image.image)
                self.labels.append(i)
        self.getStats()
        self.displayStats()

    def displayImage(self, i):
        pyplot.imshow(self.data[i], cmap='Greys')
        pyplot.show()

    def getStats(self):
        total = len(self.labels)
        counters = [0] * len(groups)
        distrib = []
        for label in self.labels:
            counters[label] += 1
        for counter in counters:
            distrib.append(round(counter*100/total, 3))
        self.stats = {"total": total, "nbr_images": counters, "distr_images:": distrib}

    def displayStats(self):
        pprint(self.stats)
        pyplot.bar(groups, self.stats["nbr_images"])
        pyplot.title(self.name)
        pyplot.show()

    def comparePrediction(self, result, classifier):
        errors = sum(result != self.labels)
        successRate = round((len(self.labels)-errors)*100/len(self.labels), 3)
        print(errors, " erreurs : ", str(successRate), "% de réussite")

        disp = metrics.ConfusionMatrixDisplay.from_predictions(self.labels, result)
        disp.figure_.suptitle("Confusion Matrix : " + classifier + " ; " + str(successRate) + "% de réussite")
        # print(f"Confusion matrix:\n{disp.confusion_matrix}")
        pyplot.show()

class XRay:
    def __init__(self, filename):
        image = Image.open(filename)
        # Normalize image
        self.normalize(image, (image.size), imageSize)

    def normalize(self, image, or_size, new_im_size):
        (or_width, or_height) = or_size
        (new_width, new_height) = new_im_size
        ratio = min(new_width/or_width, new_height/or_height)
        (xray_width, xray_height) = (int(or_width*ratio), int(or_height*ratio))

        rgbImage = (image if image.mode == "RGB" else image.convert("RGB"))
        residedXRay = rgbImage.resize((xray_width, xray_height))
        newImage = Image.new("RGB", new_im_size, (0, 0, 0))
        newImage.paste(residedXRay, (int((new_width-xray_width)/2), int((new_height-xray_height)/2)))
        self.image = numpy.asarray(newImage) / 255

# def scoring(Y, Y_pred):
#     return sum(Y == Y_pred)

groups = ["NORMAL", "PNEUMONIA"];
imageSize = (256, 256)
trainsample = Dataset("chest_Xray/train")