import cv2
import numpy as np
import os

#εδω γινετε πρελον η τελικη επεξεργασια ωστε ο αλγοριθμος να μπορει να παρει τις πληροφοριες που χρειαζετε και να φτασει ενα βημα πιο κοντα στην αναγνωριση

#η deskew θα μας βωειθησει να επεξεργαστουμε γραματα τα οπια ειναι στραβα θα κανει μια παλικαρια να τα ισωσει
def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*20*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (20, 20), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img
# αυτο ειναι το ασπορμαυρο και προεπεξεργασμενο dataset μας 
BASE_PATH ="dataset6.2"

size = 22 #εδω επιλεγουμε το μεθεγθοε σε πιξελ για καθε εικονα οσο μεγαλητερο τοσο καλητερη αναλυση αλλα τοσο πιο μεγαλο φορτιο

hog = cv2.HOGDescriptor(_winSize =(20, 20),_blockSize =(10,10),_blockStride=(5,5), _cellSize=(5,5), _nbins=9) #εδω αποφασισα να συνεχισω με τον hog descriptor το οποιο κοιτα ακμες σχηματα και της κατευθνσεις των γραμμων που βγασζει απολυτο νοημα για τα γραμματα
#πειραματιζοντας με  της παραπανω τημες μπορουμε να κανουμε τον αλγοριθμος καλητερο και χειροτερο
data=[]
labels = []

letters = sorted(os.listdir(BASE_PATH))
label_map ={letter:i for i ,letter in enumerate(letters)}
print(f"βρεθηκε{len(letters)} γραμμα: {letters}")

for letter in letters :
    letter_path = os.path.join(BASE_PATH,letter)
    if not os.path.isdir(letter_path): continue
    print(f"επεξεργασια γραμματος :{letter}")
    for img_name in os.listdir(letter_path):
        img_path = os.path.join(letter_path,img_name)
        img =cv2.imread(img_path ,cv2.IMREAD_GRAYSCALE)
        if img is None : continue 
        img_resized = cv2.resize(img,(size,size))
        img_deskewed = deskew(img_resized)
        hog_features = hog.compute(img_deskewed)# εδω πλεον περνουμε ολες της απαρετιτες οππληροφοριες απο καθε εικονα για τα γραμματα γωνιες γραμμες κατευθνσεις και τους φτιαχνουμε ενα προφιλ
        flattend = hog_features.flatten() #αυτο πλεον γινετε ενας μονοδιαστατος πινακας
        data.append(flattend)
        labels.append(label_map[letter])#αποθηκευση 

data = np.array(data,dtype=np.float32)
labels = np.array(labels,dtype=np.int32)


np.savez("processed_data_hog_final220samples.npz", train_data=data,train_labels= labels)

print("\n---etoimos ---")
print(f"oikones poy epeksergastikan: {len(data)}")
print(f"sxhma : {data.shape}") 
print("αποθηκευμενα στο 'processed_data_hog_final220samples.npz'") #και τελος η επεξεργασια
