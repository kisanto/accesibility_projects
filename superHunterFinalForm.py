import sys
print(sys.executable)
import cv2
import cv2.aruco as aruco 
import numpy as np
import os
import time
import itertools 
from collections import Counter
from spellchecker import SpellChecker

#εδω πλεον θα αποθηκευουμε και σε ενα αρχειο τι γραφουμε με ενα τριγκερ
spell = SpellChecker()
spell.word_frequency.add("kisando")


aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

def get_wraped_board(frame, corners, ids):
    pts_src = np.zeros((4,2), dtype="float32")
    for i in range(len(ids)):
        marker_id = ids[i][0]
        if marker_id < 4:  # Core corners are strictly 0, 1, 2, 3
            pts_src[marker_id] = np.mean(corners[i][0], axis=0)
    
    width = 800
    height = 600
    pts_dts = np.array([[0,0], [width,0], [width,height], [0,height]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(pts_src, pts_dts)
    return cv2.warpPerspective(frame, matrix, (width, height))

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5 * 20 * skew], [0, 1, 0]])
    return cv2.warpAffine(img, M, (20, 20), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)


hog = cv2.HOGDescriptor((20,20), (10,10), (5,5), (5,5), 9)

with np.load("processed_data_hog_final220samples.npz") as data:
    train_data, train_labels = data["train_data"], data["train_labels"]

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

tracked_blobs = []
identified_lines = ["Awaiting text..."]
raw_lines = [""]


last_update_time = time.time()
trigger_held = False  
save_feedback_expiry = 0

while True:
    ret, frame = cap.read()
    if not ret: break

    now = time.time()
    corners, ids, rejected = detector.detectMarkers(frame)
    
    if ids is not None:
        flat_ids = ids.flatten()
        
     
        if all(cid in flat_ids for cid in [0, 1, 2, 3]):
            board_view = get_wraped_board(frame, corners, ids)
            
            # ArUco protection masks (90px)
            m = 90  
            h, w = board_view.shape[:2]
            cv2.rectangle(board_view, (0, 0), (m, m), (255, 255, 255), -1)              
            cv2.rectangle(board_view, (w-m, 0), (w, m), (255, 255, 255), -1)            
            cv2.rectangle(board_view, (w-m, h-m), (w, h), (255, 255, 255), -1)          
            cv2.rectangle(board_view, (0, h-m), (m, h), (255, 255, 255), -1)            
            
            gray_board = cv2.cvtColor(board_view, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray_board, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

            #εδω εχουμε την σκανδαλη μας 
            if 4 in flat_ids:
                if not trigger_held:
                    filename = "αρχειο.txt"
                    with open(filename, "w", encoding="utf-8") as f:
                        
                        for idx, line in enumerate(identified_lines):
                            f.write(f"Line {idx+1}: {line}\n")
                    print(f"\αποθηκευθηκε'{filename}'")
                    save_feedback_expiry = now + 2.0
                    trigger_held = True  
            else:
                trigger_held = False  

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            raw_rects = []
            for cnt in contours:
                x, y, w_box, h_box = cv2.boundingRect(cnt)
                if 15 < w_box < 200 and 15 < h_box < 200:
                    raw_rects.append([x, y, w_box, h_box])
                    raw_rects.append([x, y, w_box, h_box])

            final_rects, _ = cv2.groupRectangles(raw_rects, groupThreshold=1, eps=0.2)
            
            filtered_rects = []
            for rect in final_rects:
                rx, ry, rw, rh = rect
                rcx, rcy = rx + rw // 2, ry + rh // 2
                is_inner_contour = False
                for existing in filtered_rects:
                    ex, ey, ew, eh = existing
                    ecx, ecy = ex + ew // 2, ey + eh // 2
                    if np.sqrt((rcx - ecx)**2 + (rcy - ecy)**2) < 10:
                        is_inner_contour = True
                        break
                if not is_inner_contour:
                    filtered_rects.append(rect)

            padding = 8  
            for rect in filtered_rects:
                x, y, w_box, h_box = rect
                x_pad, y_pad = max(0, x - padding), max(0, y - padding)
                w_pad, h_pad = min(w - x_pad, w_box + (padding * 2)), min(h - y_pad, h_box + (padding * 2))
                
                roi = thresh[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
                mask = np.zeros((max(w_pad, h_pad), max(w_pad, h_pad)), np.uint8)
                mask[(mask.shape[0]-h_pad)//2:(mask.shape[0]-h_pad)//2+h_pad, (mask.shape[1]-w_pad)//2:(mask.shape[1]-w_pad)//2+w_pad] = roi

                test_img = deskew(cv2.resize(mask, (20, 20)))
                test_piggy = hog.compute(test_img).reshape(1, -1)

                _, result = svm.predict(test_piggy)
                idx_1st = int(result[0][0])  

                distances = np.linalg.norm(train_data - test_piggy, axis=1)
                flat_labels = train_labels.flatten()
                class_distances = {int(cls): np.min(distances[flat_labels == cls]) for cls in np.unique(flat_labels) if len(np.where(flat_labels == cls)[0]) > 0}
                sorted_by_similarity = sorted(class_distances.keys(), key=lambda k: class_distances[k])
                
                idx_2nd = idx_1st
                for candidate in sorted_by_similarity:
                    if candidate != idx_1st:
                        idx_2nd = candidate
                        break

                if 0 <= idx_1st < len(letters):
                    label_1st = letters[idx_1st]
                    label_2nd = letters[idx_2nd] if 0 <= idx_2nd < len(letters) else "?"
                    cx, cy = x + w_box // 2, y + h_box // 2

                    cv2.rectangle(board_view, (x, y), (x+w_box, y+h_box), (0, 255, 0), 2)
                    cv2.putText(board_view, f"{label_1st}({label_2nd})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    matched = False
                    for blob in tracked_blobs:
                        if np.sqrt((cx - blob['cx'])**2 + (cy - blob['cy'])**2) < 15:
                            blob['history'].append((now, label_1st, label_2nd))
                            blob['x'], blob['y'], blob['w'], blob['h'], blob['cx'], blob['cy'] = x, y, w_box, h_box, cx, cy 
                            matched = True
                            break
                    if not matched:
                        tracked_blobs.append({'x': x, 'y': y, 'w': w_box, 'h': h_box, 'cx': cx, 'cy': cy, 'history': [(now, label_1st, label_2nd)]})

            for blob in tracked_blobs:
                blob['history'] = [item for item in blob['history'] if now - item[0] <= 5.0]
            tracked_blobs = [b for b in tracked_blobs if len(b['history']) > 0]

        
            time_elapsed = now - last_update_time
            if time_elapsed >= 5.0:
                valid_blobs = [b for b in tracked_blobs if len(b['history']) >= 2]
                if len(valid_blobs) > 0:
                    rows = []
                    for b in valid_blobs:
                        placed = False
                        for row in rows:
                            if abs(b['cy'] - np.mean([member['cy'] for member in row])) <= 65:
                                row.append(b)
                                placed = True
                                break
                        if not placed: rows.append([b])
                    
                    rows.sort(key=lambda r: np.mean([member['cy'] for member in r]))
                    raw_lines, identified_lines = [], []
                    
                    for row in rows:
                        row.sort(key=lambda b: b['cx'])
                        row_letter_nodes = []
                        for i in range(len(row)):
                            b = row[i]
                            if i > 0:
                                if (b['x'] - (row[i-1]['x'] + row[i-1]['w'])) > (row[i-1]['w'] + b['w']) * 0.45:
                                    row_letter_nodes.append((" ", " "))
                            g1 = [item[1] for item in b['history']]
                            g2 = [item[2] for item in b['history']]
                            if g1 and g2:
                                row_letter_nodes.append((Counter(g1).most_common(1)[0][0], Counter(g2).most_common(1)[0][0]))
                            
                        words_nodes, current_word = [], []
                        for node in row_letter_nodes:
                            if node == (" ", " "):
                                if current_word: words_nodes.append(current_word); current_word = []
                                words_nodes.append([(" ", " ")])
                            else: current_word.append(node)
                        if current_word: words_nodes.append(current_word)
                            
                        raw_row_chunks, clean_row_chunks = [], []
                        for word_node in words_nodes:
                            if word_node == [(" ", " ")]:
                                raw_row_chunks.append(" "); clean_row_chunks.append(" "); continue
                            word_1st_raw = "".join([n[0] for n in word_node])
                            raw_row_chunks.append(word_1st_raw)
                            
                            if word_1st_raw.isalpha():
                                permutations = ["".join(p).lower() for p in itertools.product(*word_node)]
                                valid_permutations = spell.known(permutations)
                                if valid_permutations:
                                    clean_row_chunks.append(max(valid_permutations, key=lambda w: spell.word_usage_frequency(w)).upper())
                                else:
                                    corrected = spell.correction(word_1st_raw.lower())
                                    clean_row_chunks.append(corrected.upper() if corrected else word_1st_raw.upper())
                            else:
                                clean_row_chunks.append(word_1st_raw.upper())
                                
                        raw_lines.append("".join(raw_row_chunks))
                        identified_lines.append("".join(clean_row_chunks))
                
                last_update_time = now

            cv2.imshow("εικονικος πινακας", board_view)

    aruco.drawDetectedMarkers(frame, corners, ids)
    cv2.imshow("καμερα", frame)
    
    if cv2.waitKey(1) & 0xFF == 27: 
        break

cap.release()
cv2.destroyAllWindows()
