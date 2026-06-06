import sys
print(sys.executable)
import cv2
import cv2.aruco as aruco 
import numpy as np
import os

#εδω  χρησιμοποιουμε τον σποτερ απο πριν
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

def get_wraped_board(frame, corners, ids):
    pts_src = np.zeros((4,2), dtype="float32")

    for i in range(len(ids)):
        marker_id = ids[i][0]
        if marker_id < 4:
            pts_src[marker_id] = np.mean(corners[i][0], axis=0)
    
    width = 800 #αυτα πρεπει να προσαρμοστουν αναλογα με τον πιανακ
    height = 600
    pts_dts = np.array([[0,0], [width,0], [width,height], [0,height]], dtype="float32")

    matrix = cv2.getPerspectiveTransform(pts_src, pts_dts)
    wraped = cv2.warpPerspective(frame, matrix, (width, height))
    return wraped

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5 * 20 * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (20, 20), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

# οπως και στο αλλο προγραμμα πρεπει οι μεταβλητες του hog  να παρεμενουν ιδιοιες
hog = cv2.HOGDescriptor((20,20), (10,10), (5,5), (5,5), 9)
#φορονουμε τα δεδομενα μας
with np.load("processed_data_hog_final220samples.npz") as data:
    train_data = data["train_data"]
    train_labels = data["train_labels"]
#training
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_RBF)
svm.setC(12.5)
svm.setGamma(0.5)
svm.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)
# η καμερα
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 10) 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

letters = sorted(os.listdir("dataset")) 
num_classes = len(letters)

while True:
    ret, frame = cap.read()
    if not ret: break

    corners, ids, rejected = detector.detectMarkers(frame)
    if ids is not None:
        
        if len(ids) == 4:
            board_view = get_wraped_board(frame, corners, ids)
            
            # εδω κανουμε εν ακλολπακι να καληψουμε τα μαρκες στο feed  που δεχετε το hog 
            m = 40  
            h, w = board_view.shape[:2]
            bg_color = (255, 255, 255) 
            
            cv2.rectangle(board_view, (0, 0), (m, m), bg_color, -1)              
            cv2.rectangle(board_view, (w-m, 0), (w, m), bg_color, -1)            
            cv2.rectangle(board_view, (w-m, h-m), (w, h), bg_color, -1)          
            cv2.rectangle(board_view, (0, h-m), (m, h), bg_color, -1)            
            # απαρετητες επεξεργασιες για το τι βλεεπει σοτν πιανκα
            gray_board = cv2.cvtColor(board_view, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray_board, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)
            
            kernel = np.ones((5,5), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            #εδω χρησιμοποιουμε μια μεθοδο η οποια βλεπει μελανι μαζεμενο στον πινακα βρισζεκι το κεντορο του μελανιου και το κανει το roi για την αναγνωριση του απο τον hog
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            raw_rects = []
            for cnt in contours:
                x, y, w_box, h_box = cv2.boundingRect(cnt)
                if 15 < w_box < 200 and 15 < h_box < 200:
                    raw_rects.append([x, y, w_box, h_box])
                    raw_rects.append([x, y, w_box, h_box])

            final_rects, _ = cv2.groupRectangles(raw_rects, groupThreshold=1, eps=0.2)
            
            padding = 8  
            for rect in final_rects:
                x, y, w_box, h_box = rect
                
                x_pad = max(0, x - padding)
                y_pad = max(0, y - padding)
                w_pad = min(w - x_pad, w_box + (padding * 2))
                h_pad = min(h - y_pad, h_box + (padding * 2))
                
                roi = thresh[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]

                mask = np.zeros((max(w_pad, h_pad), max(w_pad, h_pad)), np.uint8)
                mask[(max(w_pad, h_pad) - h_pad) // 2 : (max(w_pad, h_pad) - h_pad) // 2 + h_pad, 
                     (max(w_pad, h_pad) - w_pad) // 2 : (max(w_pad, h_pad) - w_pad) // 2 + w_pad] = roi

                test_img = cv2.resize(mask, (20, 20)) 
                test_img = deskew(test_img)
                
                test_piggy = hog.compute(test_img).reshape(1, -1)

               
                test_img = cv2.resize(mask, (20, 20)) 
                test_img = deskew(test_img)
                
                test_piggy = hog.compute(test_img).reshape(1, -1)

                # η πρωτη μαντεψια για το τι μπορει να ειναι αυτο το γραμμα
                _, result = svm.predict(test_piggy)
                idx_1st = int(result[0][0])  

                
                # εδω υπολογιζουμε την γωμετρικη αποσταση απο ολες τις εικονες γι ανα δουμε την δευτερη καλητερη μοπαντεψια
                distances = np.linalg.norm(train_data - test_piggy, axis=1)
                
                flat_labels = train_labels.flatten()
                unique_classes = np.unique(flat_labels)
                
                class_distances = {}
                for cls in unique_classes:
                    
                    class_indices = np.where(flat_labels == cls)[0]
                    if len(class_indices) > 0:
                        
                        class_distances[int(cls)] = np.min(distances[class_indices])
                
                
                sorted_by_similarity = sorted(class_distances.keys(), key=lambda k: class_distances[k])
                
                # Scan our sorted closest classes and pick the runner-up that isn't the 1st guess winner
                idx_2nd = idx_1st
                for candidate in sorted_by_similarity:
                    if candidate != idx_1st:
                        idx_2nd = candidate
                        break
              

                if 0 <= idx_1st < len(letters):
                    label_1st = letters[idx_1st]
                    label_2nd = letters[idx_2nd] if 0 <= idx_2nd < len(letters) else "?"
                    
                    εδω κυκλονουμε τα γραματα
                    cv2.rectangle(board_view, (x, y), (x+w_box, y+h_box), (0, 255, 0), 2)
                    
                    # και γραφουμε της μαντεψιες μας
                    display_text = f"{label_1st} ({label_2nd})"
                    cv2.putText(board_view, display_text, (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("τι μεντευει", board_view)

    aruco.drawDetectedMarkers(frame, corners, ids)
    cv2.imshow("τι βλεπει η καμερα", frame)
    
    if cv2.waitKey(1) & 0xFF == 27: 
        break

cap.release()
cv2.destroyAllWindows()
