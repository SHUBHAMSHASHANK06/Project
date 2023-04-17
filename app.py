import streamlit as st
import face_recognition
import numpy as np
import cv2 as cv

def recognize(search_for, search_in):
  # Load a sample picture and learn how to recognize it.
  known_image = face_recognition.load_image_file(search_for)
  encoding = face_recognition.face_encodings(known_image)[0]

  # Load an image with unknown faces
  unknown_image = face_recognition.load_image_file(search_in)

  # Find all the faces and face encodings in the unknown image
  face_locations = face_recognition.face_locations(unknown_image)
  face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

  cv_image = cv.imread(unknown_image)


  # Loop through each face found in the unknown image
  for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

      # See if the face is a match for the known face(s)
      matches = face_recognition.compare_faces([encoding], face_encoding)

      # Use the known face with the smallest distance to the new face
      face_distances = face_recognition.face_distance([encoding], face_encoding)
      best_match_index = np.argmin(face_distances)
      if matches[best_match_index]:

          # Draw a box around the face 
	  cv.rectangle(cv_image, (left - 20, top - 20), (right + 20, bottom + 20), (255, 0, 0), 20)

  # Display the resulting image
  cv.imshow("Output", cv_image)  

def main():
	"""Face Search App"""

	st.title("Face Search App")
	st.text("Build with Streamlit")

	activities = ["Search","About"]
	choice = st.sidebar.selectbox("Select Activty",activities)

	if choice == 'Search':
		st.subheader("Face Search")

		search_for = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])
    search_in = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])

		if search_for is not None and search_in is not None:
      search_for, search_in = st.image(search_for), st.image(search_in)
      recognize(search_for, search_in)

  elif choice == 'About':
    st.subheader("Face Search App")
		st.markdown("Built by SHUBHAM SHASHANK")

if __name__ == '__main__':
		main()	
