import cv2 as cv
import copy
import numpy as np

class ImageDetection():
	ImageGray = None
	OriginalImage = None
	OutlineColor = (255, 0, 0)

	Width, Height = 0, 0

	BoxList = []
	ImageList = []

	def __init__(self) -> None:
		pass

	def SetImage(self, Image):
		self.OriginalImage = copy.deepcopy(Image)
		self.Height = Image.shape[0]
		self.Width = Image.shape[1]
		#회색으로 변환합니다.
		self.ImageGray = cv.cvtColor(copy.deepcopy(Image), cv.COLOR_BGR2GRAY)
		self.BoxList=[]
		self.ImageList=[]

	def MinAndMax(self, Value, Min : int, Max : int):
		if Value < Min:
			return Min
		elif Value > Max:
			return Max
		return Value

	def GetImages(self):
		return self.ImageList

	def Rotate(self, ResultImage, angle):
		image_center = tuple(np.array(ResultImage.shape[1::-1]) / 2)
		rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
		ResultImage = cv.warpAffine(ResultImage, rot_mat, ResultImage.shape[1::-1], flags=cv.INTER_LINEAR)
		return ResultImage

	def Convert(self):
		# BLUR 로 변환한 후, OTSU 를 사용해 이미지를 변환합니다
		BlurImage = cv.GaussianBlur(self.ImageGray, (5, 5), 0)
		_, OneZeroImage = cv.threshold(BlurImage, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
		ConvertedImage = copy.deepcopy(self.OriginalImage)
		# 이러면 비교적 깔끔히 이미지를 인식할 수 있습니다.

		# 코너를 찾습니다
		Contours, _ = cv.findContours(OneZeroImage, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
		for Index, Contour in enumerate(Contours):
			# 해당 코너의 범위를 확인하고 너무 크거나 작으면 무시합니다
			Area = cv.contourArea(Contour)
			if Area < 2000 or 100000 < Area:
				continue

			# cv.drawContours(ConvertedImage, Contours, Index, (0, 0, 255), 2)
			
			# 기울어진 박스를 찾습니다.
			Rect = cv.minAreaRect(Contour)
			AngledBox = cv.boxPoints(Rect)
			AngledBox = np.int0(AngledBox)
			cv.drawContours(ConvertedImage, [AngledBox], 0, self.OutlineColor, 3)

			# 이미지를 자르기 위해 x,y,w,h 를 구합니다
			x, y, w, h = cv.boundingRect(Contour)
			# 여기서 약간의 오차 보정을 위해 +- 70 을 추가 범위로 제공합니다
			AdditionValue = 30
			x-=int(AdditionValue / 2)
			y-=int(AdditionValue / 2)
			w+=AdditionValue
			h+=AdditionValue
			# 최대와 최소를 설정합니다 오류날 수 있기 때문이죠!
			x,w,y,h = self.MinAndMax(x, 0, self.Width), self.MinAndMax(w, 0, self.Width), self.MinAndMax(y, 0, self.Height), self.MinAndMax(h, 0, self.Height)
			cv.rectangle(ConvertedImage, (x,y), (x+w, y+h), (0, 0, 255), 3)

			# Angle = Rect[2]
			# if w < h:
			# 	Angle = 90 - Angle
			# else:
			# 	Angle = -Angle
			
			# CroppedImage = copy.deepcopy(self.OriginalImage[y:y+h, x:x+w])
			# RotatedImage = self.Rotate(CroppedImage, Angle)
			# 이미지를 회전시켜줍니다
			
			self.ImageList.append(copy.deepcopy(self.OriginalImage[y:y+h, x:x+w]))
			self.BoxList.append(((x,y,w,h), AngledBox))


		return ConvertedImage