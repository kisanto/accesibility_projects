import cv2
import numpy as np
import os

#ξανα η deskew
def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*20*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (20, 20), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img
#εδω περνουμε το αρχειο με τις παραμετρους και της πληροφοριες για τα χαρακτιριστικα των γραματων και το φορτονουμε 
with np.load("processed_data_hog_final220samples.npz") as data:
    train_data = data["train_data"]
    train_labels = data["train_labels"]

hog = cv2.HOGDescriptor((20,20), (10,10), (5,5), (5,5), 9)

#ςδω κανουμε μερικα μαγικαι και φτιαχνουμε εναν support vector machine vlassifier του οποιου η δουλεια ειναι να αποφασιζει σε πια κλαση ανοικουν τα χαρακτριστικα (απο το αρχειο )
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_RBF)
svm.setC(12.5)# η αυτηστιροτιτα του μοντελου εξαρτατε απο αυτο
svm.setGamma(0.5)#τοπηκες επιροες
svm.train(train_data, cv2.ml.ROW_SAMPLE, train_labels) #και ορα για μελετη για το svm
#εδω αρχεικοποιυμε την καμερα
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 10) 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
letters =sorted(os.listdir("dataset" ))# καλο ειναι να ειπαρχει ενας φακελος με με φακελους απο ολους οτυς χαρακτηρες που θελουμε να αναγνωριζει 

while True:
    ret, frame = cap.read()
    if not ret: break
    h, w, _ = frame.shape
    x1, y1, x2, y2 = w//2-80, h//2-80, w//2+80, h//2+80
    roi = frame[y1:y2, x1:x2] #στο συγκεριμενο προγραμα προσπαθουμε να αναγνωρισει το γραμα σε ενα συγκεκριμενο σειμειο αρα το βοηθαμε να βρει τα γραμματα

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 2)#γινοντε οι καταλληλες επεξεργασιεςε για να μοιαζουν με τα δεδομενα που εχουν επεξεργαστη προηγουμενος
    #παιζουμε μπαλα με adaptive threshhold γιατι μπορει να εχουμ εκαπιον χαλαρο μαρκαδορο αλλα θελουμε το ρπγραμμα μας να λειτουργει ακομα
    kernel =np.ones((3,3),np.uint8)

    test_img = cv2.resize(thresh, (22, 22))
    test_img = deskew(test_img)
    test_piggy = hog.compute(test_img).reshape(1,-1)
    #εδω συγκρινουμε τα χαρακτιριστικα απο το dataset με αυτα που "βλεπει "ο hog αυτην την στημη
    res = svm.predict(test_piggy, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
    raw_score = res[1][0][0]
    _, result =svm.predict(test_piggy)
    _, result = svm.predict(test_piggy)
    idx = int(result[0][0])  #περνουμε το καλητερο σκλρο και λεμε οτι εινια η μαντεψια του υπολογηστη
    
    if 0 <= idx < len(letters):
        predicted_name = letters[idx]
    else:
        predicted_name = "δεν το ξερω :("
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"βλεπω ...: {predicted_name}", (x1, y1-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("αναγνωριση", frame)
    cv2.imshow("επεξεργασμενο feed", thresh)

    if cv2.waitKey(1) & 0xFF == 27: #με το esc βγενουμε απο το προγραμμα
        break

cap.release()
cv2.destroyAllWindows()
