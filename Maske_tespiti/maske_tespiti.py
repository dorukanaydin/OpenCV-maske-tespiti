import cv2
import numpy as np
#Webcam görüntüsünde maske takılı değilse yüzümüze korona resmi koyup maske tak diyen,maske takılıysa teşekkür eden program

face_cascade = cv2.CascadeClassifier(r"opencv\cascade_xml\haarcascade_frontalface_default.xml")
mouth_cascade = cv2.CascadeClassifier(r"opencv\cascade_xml\haarcascade_mcs_mouth.xml")

korona = cv2.imread(r"opencv\video_ve_resimler\korona.png")
yazi_maskeli = cv2.imread(r"opencv\video_ve_resimler\yazi.png")
yazi_maskesiz = cv2.imread(r"opencv\video_ve_resimler\yazi2.png")

korona_gray = cv2.cvtColor(korona,cv2.COLOR_BGR2GRAY)
ret,mask = cv2.threshold(korona_gray, 5, 255, cv2.THRESH_BINARY) #5-255 değerleri arası beyaz yapıldı.

cv2.namedWindow("Maske Tespiti")

cam = cv2.VideoCapture(0)

cam_x, cam_y = int(cam.get(3)), int(cam.get(4))    #Kameranın boyutlarını değişkenlere atadık.

cerceve = np.zeros((cam_y+200, cam_x, 3), np.uint8)     #Çerçevenin alt kısmına yazı yazmak için 200'lük yer ekledik.

yazi_maskeli = cv2.resize(yazi_maskeli, (cam_x, 200))
yazi_maskesiz = cv2.resize(yazi_maskesiz, (cam_x, 200))

while cam.isOpened():
    ret, frame = cam.read()
    video = frame.copy()
    
    if not ret:
        print("Hata...")
    
    cerceve[-200:,:] = yazi_maskeli  #Başlangıçta yazi_maskeli'nin yazması için (-200 demek aşağıdan başlamak)
    cerceve[:-200,:] = video         #Çerçevenin üst tarafına anlık görüntü eklendi.

    gray_image = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray_image, 1.1, 7)
    
    if(len(faces) == 0):
        print("Yüz tespit edilemedi.")
    
    else:
        for x,y,w,h in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
            roi_gray = gray_image[y:y+h, x:x+w]     #Belirlenen yüzü ağız arama işlemi için kırptık.
            roi = frame[y:y+h, x:x+w]
            
            mouth = mouth_cascade.detectMultiScale(roi_gray, 1.2, 5)
            
            
            if(len(mouth) == 0):
                print("Maske Takili")
                
            else:
                print("Maske tak")
                cerceve[-200:,:] = yazi_maskesiz
                dsize = (w,h)
                
                korona_resize = cv2.resize(korona, dsize)
                mask_resize = cv2.resize(mask, dsize)
                mask_inv = cv2.bitwise_not(mask_resize)   #Tersleme işlemi yapıldı.(0'lar 1,1'ler 0)
                
                
                img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
                img2_fg = cv2.bitwise_and(korona_resize, korona_resize, mask=mask_resize)
                toplam = cv2.add(img1_bg,img2_fg)
                
                video[y:y+h, x:x+w] = toplam         #Korona resmi yüzümüze eklendi.
                cerceve[:-200,:] = video
        
    cv2.imshow("Maske Tespiti",cerceve)
    
    if cv2.waitKey(33) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
    