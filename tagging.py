# Python program to tag all the images in tagged_images
    
# importing cv2
import cv2
import json


# font
font = cv2.FONT_HERSHEY_SIMPLEX
# org
question_org = (20, 20)
answer_org = (20, 40)
# fontScale
fontScale = 0.5
# Blue color in BGR
black = (0, 0, 0)
white = (255, 255, 255)
# Line thickness of 2 px
inner_thickness = 1
outer_thickness = 3


# path = r"/home/lawrence92/TRICD/" + "000000000139 copy.jpg"
# print(path)
# image = cv2.imread(path)
# image = cv2.putText(image, 'OpenCV', org, font, fontScale, color, thickness, cv2.LINE_AA)
# # window_name = 'Image'
# # cv2.imshow(window_name, image)
# cv2.imwrite("tagged_test2.jpg", image)


with open('tags.json', 'r') as openfile:
 
    # Reading from json file
    tags_dict = json.load(openfile)
    

answers = tags_dict["annotations"]
questions = tags_dict["questions"]
fileNames = []
for tag in questions:
    #print(tag)
    image_id =  tag["image_id"]
    # print(image_id)
    question = tag["question"]
    question_id = tag["question_id"]
    file_name = tag["file_name"]
    
    int_id = int(image_id) - 1
    answer_dict = answers[int_id]
    answer = answer_dict['answer']
    coco_type = answer_dict['coco_type']
    printed_answer = "answer: " + str(answer) + " | type: " + coco_type
    # path
    path = r"/home/lawrence92/TRICD/val2017/" + file_name
    
    # Reading an image in default mode
    image = cv2.imread(path)
    # Window name in which image is displayed
    # window_name = 'Image'
    
    # Using cv2.putText() method
    image = cv2.putText(image, question, question_org, font, 
                       fontScale, white, outer_thickness, cv2.LINE_AA)
    image = cv2.putText(image, question, question_org, font, 
                       fontScale, black, inner_thickness, cv2.LINE_AA)
    
    image = cv2.putText(image, printed_answer, answer_org, font, 
                       fontScale, white, outer_thickness, cv2.LINE_AA)
    image = cv2.putText(image, printed_answer, answer_org, font, 
                       fontScale, black, inner_thickness, cv2.LINE_AA)
    
    # Write new image into taggged_images folder
    if file_name in fileNames:
        file_name = file_name[:-4] + "F" + file_name[-4:]
    fileNames.append(file_name)
    destination_path = r"/home/lawrence92/TRICD/tagged_images/" + file_name
    # destination_path = r"/home/lawrence92/TRICD/train_images/" + file_name
    
    cv2.imwrite(destination_path, image)
    
    # Displaying the image
    # cv2.imshow(window_name, image) 
    # break



    

