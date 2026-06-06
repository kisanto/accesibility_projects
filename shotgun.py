import cv2
import os
import numpy as np

# εδω επιλεγουμε καπιο ετοιμο dataset το οποιο θελουμε να καθαρισουμε απο το θορυβο και το προεπεξεργαζομαστε 
BASE_PATH = "dataset2.2"
# ξεχωριζουμε βεβεα τους φακελους ωστε να μπορουμε να εχουμε το προτυπο ξεχωριστοα πο το επεξεργασμενο
OUTPUT_PATH = "dataset6.2" 

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# εδω φτιαχνουμε μια λιστα με τα γραματα που εχει το dataset μας τα οποια διαβαζει απο τα ονοματα των φακελων
letters = [f for f in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, f))]

for letter in letters:
    input_folder = os.path.join(BASE_PATH, letter)
    output_folder = os.path.join(OUTPUT_PATH, letter)
    #εδω δημιουργουμε φακελους γραματων στην περιπτωησ που δεν υπαρχει 
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    print(f"Cleaning folder: {letter}...")#και ξεκιναμε την επεξεργασια
    
    for img_name in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_name)
        
        # φωρτονουμε την εικονα
        img = cv2.imread(img_path)
        if img is None: continue
        
        # μετατρεπουμε την εικονα σε κλιμακα γκρι
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # μειονουμε τον θορυβο με γκαουσιανο φιλτρο
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        
        
        _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV)
        # η κατω παυλα απλα βεβεωνη οτι το thresh θα παραμινει εικονα και δεν θα μαζεψει σκουπιδια
        
        # περναμε και ενα κερνελ για ενα τελικο σχεδον σκουπισμα απο οτι σκουπιδακι μπορει να εχει η εικονα
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.erode(thresh, kernel, iterations=1)
        # και εδω τα αποθηκευουμε πλεον στον αλλο φακελο 
        save_path = os.path.join(output_folder, img_name)
        cv2.imwrite(save_path, thresh)

print("\n--- DONE! ---")
print(f"All cleaned images are in the '{OUTPUT_PATH}' folder.")
