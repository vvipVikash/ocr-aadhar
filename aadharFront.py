#Working fine---Aadhar front side---with detection of barcode if not found go with manual

import cv2
import re
import easyocr
from pyzbar.pyzbar import decode


class AadharIdFrontReader(object):

    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=False)
        self.img = None
        self.gray = None
        self.closed = None
        self.points = []
        self.cropped_qr = None
        self.data = None
        self.x = None
        self.y = None
        self.w = None
        self.h = None
        self.barcodeOutput = {}
        self.personalised = {}
        self.output = {}

    def start(self, img):
        self.img = img
        self.imagePreprocessing()
        self.qrDetection()
        res = self.cropQr()
        if res == 'Detected':
            resp = self.qrDecoder()
            if resp == 'Decoded':
                self.dataBeautify()
                self.personalisedSetup()
                return self.personalised
            else:
                self.manualAadharFront()
                return self.output
        else:
            self.manualAadharFront()
            return self.output

    def showImage(self, imgg):
        cv2.imshow("Show", imgg)
        cv2.waitKey(0)

    def imagePreprocessing(self):

        """Converting the image to grayscale"""
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        dst = cv2.fastNlMeansDenoisingColored(self.img, None, 10, 10, 7, 15)                

        """x,y gradient calculation"""
        x_grad = cv2.Sobel(self.gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        y_grad = cv2.Sobel(self.gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

        """subtraction of y-gradient from the x-gradient"""
        gradient = cv2.subtract(x_grad, y_grad)
        gradient = cv2.convertScaleAbs(gradient)

        """Blurring Image"""
        blurred = cv2.blur(gradient, (3, 3))
        
        """Image Thresholding"""
        (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
        
        """constructing closing kernel and apply it to the thresholded image"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        """Erosions and Dilations"""
        closed = cv2.erode(closed, None, iterations=4)
        self.closed = cv2.dilate(closed, None, iterations=4)
        
    def qrDetection(self):

        """Finding contours in thresholded image ->sort the contours by their area, keeping the largest one"""
        cnts, hierarchy = cv2.findContours(self.closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

        """Compute the rotated bounding box of the largest contour and set the co-ordinates"""
        rect = cv2.minAreaRect(c)
        x, y, w, h = cv2.boxPoints(rect)
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def cropQr(self):

        try:
            """Cropping the qr code detected from the image"""
            self.cropped_qr = self.img[int(round(self.x[1]))-5:int(round(self.w[1]))+5, int(round(self.h[0]))-5:int(round(self.y[0]))+5]
            return "Detected"
        except:
            return "Not detected"

    def qrDecoder(self):

        """Trying to fetch details from the qr code by resizing the qr code"""
        attempt = 0
        ratio = [0.8, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0]
        while attempt < 7:
            try:
                img = cv2.resize(self.cropped_qr, None, fx=ratio[attempt], fy=ratio[attempt], interpolation=cv2.INTER_CUBIC)
                data = decode(img)
                if data:
                    self.data = data
                    return "Decoded"
                else:
                    pass
            except:
                pass
            finally:
                attempt += 1
        return "Not Decoded"

    def dataBeautify(self):
        for items in self.data:
            dataDecoded = items.data.decode('utf-8')
            new = ""
            flag = 0
            for i in range(len(dataDecoded)):
                new += dataDecoded[i]
                if dataDecoded[i] == '"':
                    flag += 1
                    if flag == 2:
                        new += "*"
                        flag = 0

        new2 = new.split('*')
        new2 = [items.strip() for items in new2]
        for items in new2:
            val = items.split('=')
            try:
                self.barcodeOutput[val[0]] = val[1]
            except:
                self.barcodeOutput[val[0]] = None
        for key,val in self.barcodeOutput.items():
            if "uid" in key:
                uid = val.replace('"', '')              
            try:
                self.barcodeOutput[key] = val.replace('"', '')
            except:
                pass
        self.barcodeOutput["uid"] = uid

    def personalisedSetup(self):
        self.personalised["uid"] = self.barcodeOutput["uid"]
        self.personalised["name"] = self.barcodeOutput["name"]
        self.personalised["gender"] = self.barcodeOutput["gender"]
        try:
            self.personalised["gname"] = self.barcodeOutput["co"]
        except:
            self.personalised["gname"] = self.barcodeOutput["gname"]
        self.personalised["dob"] = self.barcodeOutput["dob"]
        self.personalised["state"] = self.barcodeOutput["state"]
        try:
            self.personalised["address"] = self.barcodeOutput["house"]+", "+self.barcodeOutput["street"]+", "+self.barcodeOutput["vtc"]+", "+self.barcodeOutput["dist"]+", "+self.barcodeOutput["subdist"]+", "+self.barcodeOutput['state']+" - "+self.barcodeOutput['pc']
        except:
            self.personalised["address"] = self.barcodeOutput["vtc"]+", "+self.barcodeOutput["dist"]+", "+self.barcodeOutput['state']+" - "+self.barcodeOutput['pc']

    def manualAadharFront(self):
        results = self.reader.readtext(self.img, detail=1, paragraph=False)
        loop_no = 0
        for (bbox, text, prob) in results:

            # =====================================================================================================

            # Below inside if condition 2 data assigned with the help of "DOB" regex.First 1 is itself "DOB"
            # Second 1 is Father Name in two conditions(Same Line and Seperate Line).

            # =====================================================================================================
            
            dob = re.search(r'(\d+/\d+/\d+)', text)
            if dob:
                self.output["dob"] = dob.group(0)
                if "Father" in results[loop_no - 1][1]:
                    print("Father's Name is in Front Side in single line.")
                    data = results[loop_no - 1][1].replace('Father', '')
                    self.output["gname"] = data
                elif "Father" in results[loop_no - 2][1]:
                    print("Father's Name is in front side in seperate line.")
                    self.output["gname"] = results[loop_no - 1][1]
                else:
                    print("Father's Name not found in front side")

            # =====================================================================================================
            
            if "government" in text.lower() or "india" in text.lower() or "government of india" in text.lower():
                self.output["name"] = results[loop_no + 2][1]

            # =====================================================================================================

            if "female" in text.lower():
                self.output["gender"] = "F"
            elif "male" in text.lower():
                self.output["gender"] = "M"

            # =====================================================================================================

            aadhar_regex = ("^[0-9]{1}[0-9]{3}\\" + "s[0-9]{4}\\s[0-9]{4}$")
            aadhar = re.compile(aadhar_regex)
            aadhar_num = re.search(aadhar, text)
            if aadhar_num:
                self.output["uid"] = aadhar_num.group(0)

            # =====================================================================================================

            loop_no += 1

            # =============END OF CODE===================================================================        

assigning = QRDecoder()
image = cv2.imread("path/image_name.jpg")
testing = assigning.start(image)
print(testing)
