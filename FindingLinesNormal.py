import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
import argparse
import time 

#==================================JIKA INGIN MENGGUNAKAN ARGUMENT PARSER=====================================



#ap.add_argument("--thr", default=0.2, type=float, help="confidence threshold to filter out weak detections")
    #Labels of network.
# parser = argparse.ArgumentParser(
#         description='Script to run MobileNet-SSD object detection network ')
# parser.add_argument("--video", help="path to video file. If empty, camera's stream will be used")
# parser.add_argument("--prototxt", default="MobileNetSSD_deploy.prototxt",
#                     help='Path to text network file: '             
#                      'MobileNetSSD_deploy.prototxt for Caffe model or '
#                     )
# parser.add_argument("--weights", default="MobileNetSSD_deploy.caffemodel",
#                     help='Path to weights: '
#                    'MobileNetSSD_deploy.caffemodel for Caffe model or '
#                     )

# args = parser.parse_args()



#================================================================================================================


def canny(image): #Function untuk tekstur canny
    gray=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) #Ubah gambar menjadi abu2 agar ringan di proses
    blur=cv2.GaussianBlur(gray, (5,5),8) #Untuk memblurkan gambar agar lebih smooth
    canny=cv2.Canny(blur,50,150) #Lalu baru di jadikan canny
    return canny
    
# def make_points(image, line):
#     slope, intercept = line
#     y1 = int(image.shape[0])# bottom of the image
#     y2 = int(y1*3/5)         # slightly lower than the middle
#     x1 = int((y1 - intercept)/slope)
#     x2 = int((y2 - intercept)/slope)
#     return [[x1, y1, x2, y2]]
 
# def average_slope_intercept(image, lines):
#     left_fit    = []
#     right_fit   = []
#     if lines is None:
#         return None
#     for line in lines:
#         for x1, y1, x2, y2 in line:
#             fit = np.polyfit((x1,x2), (y1,y2), 1)
#             slope = fit[0]
#             intercept = fit[1]
#             if slope < 0: # y is reversed in image
#                 left_fit.append((slope, intercept))
#             else:
#                 right_fit.append((slope, intercept))
#     # add more weight to longer lines
#     left_fit_average  = np.average(left_fit, axis=0)
#     right_fit_average = np.average(right_fit, axis=0)
#     left_line  = make_points(image, left_fit_average)
#     right_line = make_points(image, right_fit_average)
#     averaged_lines = [left_line, right_line]
#     return averaged_lines

def display_lines(image, lines): #Function tampilkan jgaris
    line_image=np.zeros_like(image) #Membuat array , yang mana di mulai dari 0 untuk di merge dengan image
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2=line.reshape(4) #Bikin area kiri kanan atas bawah
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),30) #Cv2. Line untuk menggambar garis pada jalan 
    return line_image

def region_of_interest(image): #Mencari area jalan
    height=image.shape[0] #Mengambbil informasi tinggi dari gambar 
    polygons=np.array([
        [(150,height),(1650,height),(1000,250)]  #Untuk membuat area plot, yg mana untuk di ukur untuk mendapatkan informasi garis
    ])
    mask=np.zeros_like(image) #Kita buat mask , dengan array yg berjumlah 0 pada dimensi image
    cv2.fillPoly(mask,polygons,255) #Kita gambarkan polygons yang sudah dibuat lalu di merge
    masked_image=cv2.bitwise_and(image,mask) #Mengambil informasi bitwise dari array 
    return masked_image



cap = cv2.VideoCapture("MyVideo.mp4") #Tampilkan Video
#net=cv2.dnn.readNetFromCaffe(args["prototxt"],args["model"])

print("load model....")
while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame) #Process pertama 
    cropped_canny = region_of_interest(canny_image) #Setelah dapat informasi lalu di cari area nya yg sudah di ukur sebelumnya
    lines = cv2.HoughLinesP(cropped_canny, 2, np.pi/180, 100, np.array([]), minLineLength=40,maxLineGap=10) #Kita cari lines nya dengan metode Hough Transform
    #averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, lines) #Tampilkan garis nya
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1) #Kita tambah weight agar terlihat jelas garis nya
    frame_resized=cv2.resize(combo_image,(1200,720)) #Kita resize video agar tidak melebihi ukuran layar

    cv2.imshow("result", combo_image) #Tampilkan hasilnya
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break #Jika menekan tombol q artinya keluar dari program
cap.release() #Release video
cv2.destroyAllWindows() 


