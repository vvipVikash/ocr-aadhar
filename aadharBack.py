#Working for all type of aadhar card 
import easyocr
import cv2


class AadharIdBackReader(object):

    def __init__(self) -> None:
        self.reader = easyocr.Reader(['en'], gpu=False)
        self.output = {}
        self.img = None
        self.leftImage = None
        self.rightImage = None
        self.pattern = 0

    def begin(self, img):
        self.output = {}
        self.img = img
        self.img = cv2.fastNlMeansDenoisingColored(self.img, None, 10, 10, 7, 15)
        self.cropImage()
        self.extractInformation()
        return self.output

    def cropImage(self):
        height, width, _ = self.img.shape
        width_cutoff = (width // 2) + 5
        self.leftImage= self.img[:, :width_cutoff]
        self.rightImage = self.img[:, width_cutoff:]

    def extractInformation(self):
        start = 0
        sentence = ""
        flag = 0
        for i in range(2):
            if i == 0:
                results = self.reader.readtext(self.leftImage, detail=1, paragraph=False) 
            elif i == 1:
                results = self.reader.readtext(self.rightImage, detail=1, paragraph=False) 
            for (bbox, text, prob) in results:
                result = text.lower().find('address')
                if result == 0:
                    start = 1
                else:
                    continue
                if start == 1:
                    for (bbox, text, prob) in results:
                        if "address" in text.lower():
                            flag = 1
                            continue
                        if flag == 1:
                            sentence += " " + text
                            if len(text) == 6 and text.isalnum():
                                flag = 0
                        a = sentence.split(",", 1)
                    try:
                        self.output["gname"] = ((a[0].split(":"))[1]).strip()
                    except:
                        self.output["gname"] = ((a[0].split(";"))[1]).strip()

                    self.output["address"] = a[1].strip()
        
assigning = AadharIdBackReader()
image = cv2.imread("path/image_name.jpg")
result = assigning.begin(image)
print(result)
