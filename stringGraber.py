import sys
print(sys.executable)
import cv2
import cv2.aruco as aruco 
import numpy as np
import os
import time
from collections import Counter
# εδω πλεον προσπαθουμε ν αρπαξουμε τι βλεπουμε σοτν πινακα σε μορφη string
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

def get_wraped_board(frame, corners, ids):
    pts_src = np.zeros((4,2), dtype="float32")

    for i in range(len(ids)):
        marker_id = ids[i][0]
        if marker_id < 4:
            pts_src[marker_id] = np.mean(corners[i][0], axis=0)
    
    width = 800
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


hog = cv2.HOGDescriptor((20,20), (10,10), (5,5), (5,5), 9)

with np.load("processed_data_hog_final220samples.npz") as data:
    train_data = data["train_data"]
    train_labels = data["train_labels"]

svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_RBF)
svm.setC(12.5)
svm.setGamma(0.5)
svm.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 15) 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

letters = sorted(os.listdir("dataset")) 

#εδω επηδειθ κατα την προσπαθεια μποορει να αμφιταλεντεβετε τη τελικητ ιμη που μας δινει ο υπολογιστης προσπαθουμε να βουμε κατι σταθερο 
tracked_blobs = []
last_update_time = time.time()
final_string_1st = ""
final_string_2nd = ""

while True:
    ret, frame = cap.read()
    if not ret: break

    now = time.time()

    corners, ids, rejected = detector.detectMarkers(frame)
    if ids is not None:
        
        if len(ids) == 4:
            board_view = get_wraped_board(frame, corners, ids)
            
            
            m = 70  
            h, w = board_view.shape[:2]
            bg_color = (255, 255, 255) 
            
            cv2.rectangle(board_view, (0, 0), (m, m), bg_color, -1)              
            cv2.rectangle(board_view, (w-m, 0), (w, m), bg_color, -1)            
            cv2.rectangle(board_view, (w-m, h-m), (w, h), bg_color, -1)          
            cv2.rectangle(board_view, (0, h-m), (m, h), bg_color, -1)            
            
            gray_board = cv2.cvtColor(board_view, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray_board, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)
            
            kernel = np.ones((5,5), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

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

              
                _, result = svm.predict(test_piggy)
                idx_1st = int(result[0][0])  

                distances = np.linalg.norm(train_data - test_piggy, axis=1)
                flat_labels = train_labels.flatten()
                unique_classes = np.unique(flat_labels)
                
                class_distances = {}
                for cls in unique_classes:
                    class_indices = np.where(flat_labels == cls)[0]
                    if len(class_indices) > 0:
                        class_distances[int(cls)] = np.min(distances[class_indices])
                
                sorted_by_similarity = sorted(class_distances.keys(), key=lambda k: class_distances[k])
                
                idx_2nd = idx_1st
                for candidate in sorted_by_similarity:
                    if candidate != idx_1st:
                        idx_2nd = candidate
                        break

                if 0 <= idx_1st < len(letters):
                    label_1st = letters[idx_1st]
                    label_2nd = letters[idx_2nd] if 0 <= idx_2nd < len(letters) else "?"
                    
                    # Draw immediate live frame predictions on screen
                    cv2.rectangle(board_view, (x, y), (x+w_box, y+h_box), (0, 255, 0), 2)
                    display_text = f"{label_1st} ({label_2nd})"
                    cv2.putText(board_view, display_text, (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    
                    cx = x + w_box // 2
                    cy = y + h_box // 2
                    matched = False
                    εδω προσπαθουμε να αρπαξουμε την τημη που εινια πιο επιμονη 
                    for blob in tracked_blobs:
                        bcx = blob['x'] + blob['w'] // 2
                        bcy = blob['y'] + blob['h'] // 2
                        spatial_distance = np.sqrt((cx - bcx)**2 + (cy - bcy)**2)
                        
                        if spatial_distance < 45:
                            
                            blob['history'].append((now, label_1st, label_2nd))
                            blob['x'], blob['y'] = x, y
                            blob['w'], blob['h'] = w_box, h_box
                            matched = True
                            break
                            
                    if not matched:
                        tracked_blobs.append({
                            'x': x, 'y': y, 'w': w_box, 'h': h_box,
                            'history': [(now, label_1st, label_2nd)]
                        })

           #αυτο το κασνουμε αν α 5 δευτερα για να μην καταπινουμε τους πορους μας σαν νερο
            for blob in tracked_blobs:
              
                blob['history'] = [item for item in blob['history'] if now - item[0] <= 5.0]
                
            # Drop blobs completely if they haven't been seen anywhere for 5 seconds
            tracked_blobs = [b for b in tracked_blobs if len(b['history']) > 0]

            
            time_elapsed = now - last_update_time
            countdown = max(0, int(5 - time_elapsed))
            
            if time_elapsed >= 5.0:
                # Filter out accidental camera noise flashes
                valid_blobs = [b for b in tracked_blobs if len(b['history']) >= 5]
                
                # περνουμε τους χαρακτηρες απο δεξια προς τα αριστερα
                valid_blobs.sort(key=lambda b: b['x'])
                
                str_1st_list = []
                str_2nd_list = []
                
                for b in valid_blobs:
                    # Isolate accumulated historical records
                    g1_list = [item[1] for item in b['history']]
                    g2_list = [item[2] for item in b['history']]
                    
                    # το γραμμα που εμανιστκε πιο πολυ 
                    most_common_1st = Counter(g1_list).most_common(1)[0][0]
                    most_common_2nd = Counter(g2_list).most_common(1)[0][0]
                    
                    str_1st_list.append(most_common_1st)
                    str_2nd_list.append(most_common_2nd)
                    
                final_string_1st = "".join(str_1st_list)
                final_string_2nd = "".join(str_2nd_list)
                
                
                print(f"παραθυρο 5 δευτερολεπτων")
                print(f"πρωτη μαντεψια: '{final_string_1st}'")
                print(f"δευτερη: '{final_string_2nd}'")
                print(f"=====================================\n")
                
                last_update_time = now

            cv2.imshow("εικονικος πινακας", board_view)

    aruco.drawDetectedMarkers(frame, corners, ids)
    cv2.imshow("καμερα", frame)
    
    if cv2.waitKey(1) & 0xFF == 27: 
        break

cap.release()
cv2.destroyAllWindows()
