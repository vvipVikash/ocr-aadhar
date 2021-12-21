import easyocr
import cv2


class AadharIdBackReader(object):

    def __init__(self) -> None:
        self.reader = easyocr.Reader(['en'], gpu=False)
        self.output = {}
        self.img = None
        self.pattern = 0

    def begin(self, img, pattern):
        self.output = {}
        self.img = img
        self.img = cv2.fastNlMeansDenoisingColored(self.img, None, 10, 10, 7, 15)
        self.pattern = pattern
        self.cropImage()
        self.extractInformation()
        return self.output

    def cropImage(self):
        print("Inside crop")
        height, width, _ = self.img.shape
        width_cutoff = (width // 2) + 5
        if self.pattern == 1: 
            self.img= self.img[:, :width_cutoff]
        elif self.pattern ==2:
            self.img = self.img[:, width_cutoff:]

    def extractInformation(self):
        sentence = ""
        flag = 0
        results = self.reader.readtext(self.img, detail=1, paragraph=False)  
        e11 = cv2.getTickCount()
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
            self.output["Father's Name"] = ((a[0].split(":"))[1]).strip()
        except:
            self.output["Father's Name"] = ((a[0].split(";"))[1]).strip()

        self.output["Address"] = a[1].strip()
        

x = AadharIdBackReader()
im = cv2.imread("images/AdharBackD.jpg")
z = x.begin(im, 2)
print(z)
