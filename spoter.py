import cv2
import cv2.aruco as aruco 
import numpy as np
#εδω με την βωηθεια καπιων σηματων θια βρησκουμε τον πιανκα και αναξαρτητα μετην κληση της καμερα προς τον πινακα εμιες θα τα βλεπουμε ολα καλα
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
#σε αυητ την μεθοδο προσπαθουμε να τενδοσουμε τον πιανκα σε ενα ορθογωνιο
def get_wraped_board(frame, corners, ids):
    pts_src =np.zeros((4,2),dtype="float32")

    for i in range(len(ids)): #οι γνωιες μας ειναι τα σηματα
        marker_id = ids[i][0]
        if marker_id <4 :
            pts_src[marker_id]= np.mean(corners[i][0], axis=0)
    
    width = 800 #οριζουμε ποσο θελουμε να ειναι ο πινκαας μας συνιθος με το μεγεθος η τον λογο του κανονικου πιανακα
    height =600
    pts_dts =np.array([[0,0],[width,0],[width,height],[0,height]],dtype="float32")

    matrix =cv2.getPerspectiveTransform(pts_src,pts_dts)

    wraped =cv2.warpPerspective(frame,matrix,(width,height))
    return wraped

#ξεκιναμε την καμερα
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 10) 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

while True:
    ret ,frame = cap.read()
    if not ret : break

    corners, ids ,_ =aruco.detectMarkers(frame,aruco_dict,parameters=parameters)

    
    if ids is not None:
        # ζωγραφιζουμε οτι σημα βλεπουμε
        print(f"I see {len(ids)} markers: {ids.flatten()}") # This prints to your terminal

        if len(ids) == 4:
            board_view = get_wraped_board(frame, corners, ids)
            cv2.imshow("εικονικος πινκαας", board_view)
            #μονο οταν δει ολες της γωνιες μα δειχνει τον εικονικο πινκαα
            
    aruco.drawDetectedMarkers(frame, corners, ids)
    cv2.imshow("τι βλεπει η καμερα", frame)
    if cv2.waitKey(1) & 0xFF == 27: # esc για εξοδο
        break
cap.release()
cv2.destroyAllWindows()
