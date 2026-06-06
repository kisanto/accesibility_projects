import cv2
import os
import time
#στο παρων προγραμα αποθηκευουμε μονο κεφαλεα λατινακα γραμαμτα και αριθμους απο το 0-9
BASE_PATH = "dataset"
if not os.path.exists(BASE_PATH):
    os.makedirs(BASE_PATH)
#εδω σεταρουμε τι ςρυθμισεις της καμερας

cap = cv2.VideoCapture(0)# το μηδεν ειναι για την καμερα του υπολογηστη και το 1 για τοπυ κινητου
cap.set(cv2.CAP_PROP_FPS, 10) 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
current_letter = "" 


print("1. γραψε το γραμμα που δειχνεις (λατινικοι χαρακτηρες)")
print("2. με το κενο αποθηκευσε το γραμμα και την εικονα του")
print("3. με το esc τερματιζει το προγραμμα")

while True: # παρακατω ειναι ο λα 4πλα για να μπορεις να προχωριει η διαδικασια εισαγωγης δεδομενων πιο γρηγορα
    ret, frame = cap.read()
    if not ret: break
    #εδω σεταρουμε το region of interest
  
    h, w, _ = frame.shape
  
    x1, y1, x2, y2 = (w//2-80)+90, (h//2-80)-90, (w//2+80)+90, (h//2+80)-90
    roi = frame[y1:y2, x1:x2]
    
    x12, y12, x22, y22 = (w//2-80)-90, (h//2-80)-90, (w//2+80)-90, (h//2+80)-90
    roi2 = frame[y12:y22, x12:x22]

    x13, y13, x23, y23 = (w//2-80)+90, (h//2-80)+90, (w//2+80)+90, (h//2+80)+90
    roi3 = frame[y13:y23, x13:x23]
    
    x14, y14, x24, y24 = (w//2-80)-90, (h//2-80)+90, (w//2+80)-90, (h//2+80)+90
    roi4 = frame[y14:y24, x14:x24]


    #kanoyme to roi grayscale για να τονισουμε της διαφορες και περετερο επξεργασια της εικονας
  
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    gray2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
    blur2 = cv2.GaussianBlur(gray2, (7, 7), 0)
    thresh2 = cv2.adaptiveThreshold(blur2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    gray3 = cv2.cvtColor(roi3, cv2.COLOR_BGR2GRAY)
    blur3 = cv2.GaussianBlur(gray3, (7, 7), 0)
    thresh3 = cv2.adaptiveThreshold(blur3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    gray4 = cv2.cvtColor(roi4, cv2.COLOR_BGR2GRAY)
    blur4 = cv2.GaussianBlur(gray4, (7, 7), 0)
    thresh4 = cv2.adaptiveThreshold(blur4, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
  
    status_text = f"Active Letter: {current_letter.upper()}" if current_letter else "TYPE A LETTER TO START"
  
    cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.rectangle(frame, (x12, y12), (x22, y22), (255, 255, 255), 2)
    
    cv2.imshow("τι βλεπω", frame)
    cv2.imshow(" αυτο αποθηκευεται", thresh)
    cv2.imshow(" αυτο αποθηκευεται2", thresh2)
    cv2.imshow(" αυτο αποθηκευεται3", thresh3)
    cv2.imshow(" αυτο αποθηκευεται4", thresh4)

    key = cv2.waitKey(1) & 0xFF
    
    
    if (key >= ord('a') and key <= ord('z')) or (key >= ord('0') and key <= ord('9')):
        current_letter = chr(key)
        print(f"στοχος το γραμμα: {current_letter.upper()}")
        
     
        letter_folder = os.path.join(BASE_PATH, current_letter.upper())
        if not os.path.exists(letter_folder):
            os.makedirs(letter_folder)

    elif key == 32: # εδω με το space αποθηκευοντε οι φωτογραφιες
        if current_letter:
            letter_folder = os.path.join(BASE_PATH, current_letter.upper())
            filename = f"{letter_folder}/{int(time.time())}.png"
            filename2 = f"{letter_folder}/{int(time.time())}2.png"
            filename3 = f"{letter_folder}/{int(time.time())}3.png"
            filename4 = f"{letter_folder}/{int(time.time())}4.png"
            cv2.imwrite(filename, thresh)
            cv2.imwrite(filename2, thresh2)
            cv2.imwrite(filename3, thresh3)
            cv2.imwrite(filename4, thresh4)
            print(f"αποθηκευθηκε: {filename}")
            print(f"αποθηκευθηκε: {filename2}")
            print(f"αποθηκευθηκε: {filename3}")
            print(f"αποθηκευθηκε: {filename4}")
                
        else:
            print(" σφαλμα πατησε καινο για αποθηκευση")

    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
