import cv2 # Importing the library
#===== Standard meaning values =====
MODEL_MEAN_VALUES = (78.4463377603,
                     87.7689143744,
                     114.895847746)
#===== Creating a list of age groups =====
age_list = ['(0, 2)', '(4, 6)', '(8, 12)',
            '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)',
            '(60, 100)'
            ]
#===== Gender classification list =====
gender_list = ['Male', 'Female']
#===== Loading files that recognize age and gender =====
def filesGet():
    age_net = cv2.dnn.readNetFromCaffe(
        'data/deploy_age.prototxt',
        'data/age_net.caffemodel'
    )
    gender_net = cv2.dnn.readNetFromCaffe(
        'data/deploy_gender.prototxt',
        'data/gender_net.caffemodel'
    )
    return (age_net, gender_net)

def read_from_camera(age_net, gender_net):
    font = cv2.FONT_HERSHEY_SIMPLEX # Font type
    image = cv2.imread('images/girl2.jpg') # Loading the image
    #===== Face detection file =====
    face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')
    #===== Grayscale conversion =====
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    #===== Detect multiple faces in a single image =====
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    if (len(faces) > 0): # Counting the number of faces
        print("Found {} Faces".format(str(len(faces))))
    
    for (x, y, w, h) in faces:
        # Drawing a rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
        # Extracting and copying the face for algorithm input
        face_img = image[y:y + h, h:h + w].copy()
        blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        #=== Predicting gender =====
        gender_net.setInput(blob)
        gender_p = gender_net.forward() # Output
        gender = gender_list[gender_p[0].argmax()]
        print("Gender : " + gender)
        #=== Predicting age =====
        age_net.setInput(blob)
        age_p = age_net.forward() # Output
        age = age_list[age_p[0].argmax()]
        print("Age : " + age)
        G_A = "%s %s" % (gender, age)
        cv2.putText(image, G_A, (x, y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('program', image)
    cv2.waitKey(0)
if __name__ == "__main__":
    age_net, gender_net = filesGet()
    read_from_camera(age_net, gender_net)
