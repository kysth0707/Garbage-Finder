from Module.ImageSlicer import ImageDetection
import cv2 as cv
import os
import shutil

ImageCount = 0

TargetPos = r'E:\GithubProjects\Garbage-Finder\RawDatasets\Garbage classification\Garbage classification'
for _folder in os.listdir(TargetPos):
	FolderPath = f"{TargetPos}\\{_folder}"
	ImageCount += len(os.listdir(FolderPath))


	shutil.rmtree(f"./Datasets/{_folder}")
	os.makedirs(f"./Datasets/{_folder}")
print(ImageCount)

ImageChanger = ImageDetection()

Index=0
for _folder in os.listdir(TargetPos):
	FolderPath = f"{TargetPos}\\{_folder}"
	for _file in os.listdir(FolderPath):
		ImagePath = f"{FolderPath}\\{_file}"

		img = cv.imread(ImagePath)
		ImageChanger.SetImage(img)
		ImageChanger.Convert()
		if len(ImageChanger.GetImages()) == 1:
			# cv.imshow("asdf", ImageChanger.GetImages()[0])
			cv.imwrite(f"./Datasets/{_folder}/{_file}", ImageChanger.GetImages()[0])
		else:
			# cv.imshow("asdf", img)
			cv.imwrite(f"./Datasets/{_folder}/{_file}", img)

		# cv.waitKey(0)
		# cv.destroyAllWindows()



		print(f"\rFile Converting... {Index + 1}/{ImageCount}  ", end="")
		Index+=1