import os
import re
import xml.etree.ElementTree as ET
from pascal_voc_writer import Writer
import cv2
import shutil

class annotationConverter:
    '''Class to handle conversion between ML/CNN annotation formats'''
    
    def __init__(self,num2class_label_mapping=None, class2num_label_mapping=None):
        self.num2class_label_mapping = num2class_label_mapping
        self.class2num_label_mapping = class2num_label_mapping

    def process_YOLOdirectory(self,in_dir,out_dir,img_dir,copy_imgs=False):
        '''
        Convert all annotations in an input directory containing yolo annotations

        Iterates through all txt files in directory (ignoring 'classes.txt') and
        attempts to convert them to xml files. Copies images to output dir if
        desired.
        
        Keyword Arguments:
        in_dir: String of path to directory containing yolo annotations
        out_dir: String of path to directory to place converted (VOC) annotations
        img_dir: Directory containing annotated images
        copy_imgs: Boolean indicator on whether to copy the iamges to the out_dir

        Returns:
        None
        '''

        # Validate inputs
        in_dir=in_dir.replace('\\','/')
        img_dir=img_dir.replace('\\','/')
        out_dir=out_dir.replace('\\','/')
        os.makedirs(out_dir,exist_ok=True)
        assert(os.path.isdir(in_dir)), 'Input annotation directory not found. (%r)' %in_dir
        assert(os.path.isdir(img_dir)), 'Image directory not found. (%r)' %img_dir
        assert(os.path.isdir(out_dir)), 'Output annoation directory not found. (%r)' %out_dir

        # Find all text files in directory (should contain only annotation files)
        txt_files = [a for a in os.listdir(in_dir) if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(?i)(.txt)$', a)]
        #Iterate through all text files
        for t in txt_files:
            # Ignore classes file
            if t != 'classes.txt':
                txt_filepath = os.path.join(in_dir,t)
                self.txt_to_xml(txt_filepath=txt_filepath,img_dir=img_dir,out_dir=out_dir,copy_imgs=copy_imgs)

    def txt_to_xml(self,txt_filepath, img_dir, out_dir,copy_imgs=False):
        '''
        Convert YOLO txt annotation to PascalVOC xml annotation

        Parses the annotation filepath for filenames, reads image to get shape,
        and parses yolo annotation file for class and bounding boxes. Then writes
        data to xml annotation. Copys images to output dir if desired.
        
        Keyword Arguments:
        filepath: String of path to yolo txt annotation file
        img_dir: String of directory containing annotated images
        out_dir: String of path to directory to place converted (VOC) annotations
        copy_imgs: Boolean indicator on whether to copy the iamges to the out_dir

        Returns:
        None
        '''

        # Validate inputs
        txt_filepath=txt_filepath.replace('\\','/')
        img_dir=img_dir.replace('\\','/')
        out_dir=out_dir.replace('\\','/')
        assert(os.path.exists(txt_filepath)), 'Input annotation filepath not found. (%r)' %txt_filepath
        assert(os.path.isdir(img_dir)), 'Image directory not found. (%r)' %img_dir
        assert(os.path.isdir(out_dir)), 'Output annoation directory not found. (%r)' %out_dir

        # Parse the txt filepath to get the image name and image path
        filename_base = txt_filepath[txt_filepath.rfind('/')+1:txt_filepath.rfind('.txt')]
        txt_filename = filename_base+'.txt'
        img_filename = filename_base+'.jpg'
        # Remove the hit/missXXXX numbers at the end of the image to find its folder
        img_sub_dir = filename_base[:-4]
        # Add extra folder into image directory path if needed
        if os.path.isdir(os.path.join(img_dir,img_sub_dir)):
            img_dir = os.path.join(img_dir,img_sub_dir).replace('\\','/')
        img_filepath = os.path.join(img_dir,img_filename).replace('\\','/')
        xml_dst_filepath = os.path.join(out_dir,filename_base+'.xml')

        # Read in image to get height and width
        img = cv2.imread(img_filepath,cv2.IMREAD_ANYDEPTH|cv2.IMREAD_UNCHANGED)
        img_shape = img.shape
        width = img_shape[1]
        height = img_shape[0]
        if len(img_shape) == 2:
            depth = 1
        else:
            depth = img_shape[2]

        # Instantiate xml writer
        writer = Writer(img_filepath,width,height,depth=depth)

        # Read yolo text file annotation
        with open(txt_filepath,'r') as f:
            txt_lines = f.readlines()
        
        if txt_lines != []:
            # Process each annotation in the file
            for line in txt_lines:
                class_num, x_cent, y_cent, box_width, box_height = self.parse_txt_line(line)
                # # Parse text file to extract informaiton from file
                # class_num = line[:line.find(" ")]
                # x_cent = line[line.find(class_num) + 1 + len(class_num):
                #                     line.find(" ", line.find(class_num) + 1 + len(class_num))]
                # y_cent = line[line.find(x_cent) + 1 + len(x_cent):
                #                     line.find(" ", line.find(x_cent) + 1 + len(x_cent))]
                # box_width = line[line.find(y_cent) + 1 + len(y_cent):
                #                         line.find(" ", line.find(y_cent) + 1 + len(y_cent))]
                # box_height = line[line.find(box_width) + 1 + len(box_width):
                #                         line.find(" ", line.find(box_width) + 1 + len(box_width))]
                # Get class name using mapping dictionary
                if self.num2class_label_mapping is not None:
                    class_name = self.num2class_label_mapping[int(class_num)]
                # Compute PASCAL VOC style bounding box parametersparameters
                x1 = int(round(float(x_cent)*width -
                                    (float(box_width)*width)/2))
                x2 = int(round(float(x_cent)*width +
                                    (float(box_width)*width)/2))
                y1 = int(round(float(y_cent)*height -
                                    (float(box_height)*height)/2))
                y2 = int(round(float(y_cent)*height +
                                    (float(box_height)*height)/2))
                # Add annotation entry to xml file
                writer.addObject(name=class_name,xmin=x1,ymin=y1,xmax=x2,ymax=y2)

            # Save xml file
            if copy_imgs is True:
                img_src_path = img_filepath
                if os.path.exists(img_filepath):
                    img_dst_path = os.path.join(out_dir,img_filename)
                    shutil.copyfile(img_src_path, img_dst_path)
                    writer.save(xml_dst_filepath)
                    print('Saved xml annotation '+xml_dst_filepath)
                else:
                    print('No annotation created because couldnt find and copy image',img_filepath)

            else:
                writer.save(xml_dst_filepath)
                print('Saved xml annotation '+xml_dst_filepath,' (No copy image)')
        else:
            print('No xml file created for',txt_filename,'because no data found')

    def parse_txt_line(self,line):
        '''
        Parses line of yolo text file to extract bbox data
        
        Keyword Arguments:
        line (str): String of text file corresponding to one item in the yolo annotation
        
        Returns:
        class_num: number of class
        x_cent: X coord of center of bounding box
        y_cent: Y coord of center of bounding box
        box_width: Widht of bounidng box
        box_height: Height of bounding box
        ''' 
        class_num = line[:line.find(" ")]
        x_cent = line[line.find(class_num) + 1 + len(class_num):
                            line.find(" ", line.find(class_num) + 1 + len(class_num))]
        y_cent = line[line.find(x_cent) + 1 + len(x_cent):
                            line.find(" ", line.find(x_cent) + 1 + len(x_cent))]
        box_width = line[line.find(y_cent) + 1 + len(y_cent):
                                line.find(" ", line.find(y_cent) + 1 + len(y_cent))]
        box_height = line[line.find(box_width) + 1 + len(box_width):
                                line.find(" ", line.find(box_width) + 1 + len(box_width))]
        
        return class_num, x_cent, y_cent, box_width, box_height

    def process_XMLdirectory(self,in_dir,out_dir,img_dir,copy_imgs=False):
        '''
        Convert all annotations in an input directory containing xml annotations

        Iterates through all xml files in directory and attempts to convert tehm
        to yolo .txt files. Copies images if desired
        
        Keyword Arguments:
        in_dir: String of path to directory containing yolo annotations
        out_dir: String of path to directory to place converted (YOLO) annotations
        img_dir: Directory containing annotated images
        copy_imgs: Boolean indicator on whether to copy the iamges to the out_dir

        Returns:
        None
        '''

        # Validate inputs
        in_dir=in_dir.replace('\\','/')
        img_dir=img_dir.replace('\\','/')
        out_dir=out_dir.replace('\\','/')
        assert(os.path.isdir(in_dir)), 'Input annotation directory not found. (%r)' %in_dir
        assert(os.path.isdir(img_dir)), 'Image directory not found. (%r)' %img_dir
        assert(os.path.isdir(out_dir)), 'Output annoation directory not found. (%r)' %out_dir

        # Find all text files in directory (should contain only annotation files)
        xml_files = [a for a in os.listdir(in_dir) if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(?i)(.xml)$', a)]
        #Iterate through all xml files
        for x in xml_files:
            xml_filepath = os.path.join(in_dir, x)
            self.xml_to_txt(xml_filepath=xml_filepath,img_dir=img_dir,out_dir=out_dir,copy_imgs=copy_imgs)

    def xml_to_txt(self,xml_filepath,img_dir,out_dir,copy_imgs=False):
        """
        Convert all PASCAL VOC format XML located at the 'in_dir' directory to YOLO
        text files to be placed at 'out_dir' directory. Files will have the same names
        as the VOC files.

        Keyword Arguments:
        filepath: String of path to yolo txt annotation file
        img_dir: String of directory containing annotated images
        out_dir: String of path to directory to place converted (VOC) annotations
        copy_imgs: Boolean indicator on whether to copy the iamges to the out_dir
        """

                    
        # Check that in_dir exists
        xml_filepath=xml_filepath.replace('\\','/')
        img_dir=img_dir.replace('\\','/')
        out_dir=out_dir.replace('\\','/')
        assert(os.path.exists(xml_filepath)), 'Input annotation filepath not found. (%r)' %xml_filepath
        assert(os.path.isdir(img_dir)), 'Image directory not found. (%r)' %img_dir
        assert(os.path.isdir(out_dir)), 'Output annoation directory not found. (%r)' %out_dir

        # Get basename of file
        filename_base = xml_filepath[xml_filepath.rfind('/')+1:-4]
        txt_filename = filename_base+'.txt'
        img_filename = filename_base+'.jpg'
        img_filepath = os.path.join(img_dir,img_filename).replace('\\','/')

        # Initialize list for storing xml data
        xml_list = []
        # Class/name dictionary if YOLO txt file cant have strings
        class_dict = self.class2num_label_mapping

        # Import data from XML file
        tree = ET.parse(xml_filepath)
        root = tree.getroot()
        img_width = int(root.find('size').find('width').text)
        img_height = int(root.find('size').find('height').text)
        # Iterate thorugh each object root in xml to extract "object" info
        for member in root.findall('object'):
            # Yolo format: [object # (int), x center (float), y center (float), box width (float), box height (float)]
            # All values in the annotation are normalized by image size.
            obj_params = (class_dict[member.find('name').text],
                            round(float(((int(member.find('bndbox').find('xmax').text)+int(member.find('bndbox').find('xmin').text))/2)/img_width),6),
                            round(float(((int(member.find('bndbox').find('ymax').text)+int(member.find('bndbox').find('ymin').text))/2)/img_height),6),
                            round(float((int(member.find('bndbox').find('xmax').text)-int(member.find('bndbox').find('xmin').text))/img_width),6),
                            round(float((int(member.find('bndbox').find('ymax').text)-int(member.find('bndbox').find('ymin').text))/img_height),6))
            # Append object data to a list
            xml_list.append(obj_params)
        
        # Construct output path from output directory and xml filename
        output_path = os.path.join(out_dir,txt_filename)

        # Save xml file
        if copy_imgs is True:
            img_src_path = img_filepath
            if os.path.exists(img_filepath):
                img_dst_path = os.path.join(out_dir,img_filename)
                if img_dst_path != img_filepath:
                    shutil.copyfile(img_src_path, img_dst_path)
                self.write_yolo_txt(output_path=output_path, xml_list=xml_list)
                print('Saved txt annotation '+output_path)
            else:
                print('No annotation created because couldnt find and copy image',img_filepath)

        else:
            print(xml_list)
            self.write_yolo_txt(output_path=output_path, xml_list=xml_list)
            print('Saved txt annotation '+output_path,' (No copy image)')
    
    def write_yolo_txt(self, output_path, xml_list):
        '''
        Writes xml data compiled to list to at text file
        
        Keyword Arguments:
        output_path: Whole .txt filepath string to write yolo data
        xml_list List with each element of form [object # (int), x center (float), y center (float), box width (float), box height (float)]

        Returns:
        None
        '''

        # Make txt file in form with space delimite: [object # (int), x center (float), y center (float), box width (float), box height (float)]
        with open(output_path, 'w') as f:
            # For each tuple in the xml_list
            for obj in xml_list:
                # For each object in the tuple
                for i in range(0,len(obj)):
                    ele = obj[i]
                    # If the object is last in the tuple write a newline
                    if i == 5:
                        f.write(str(ele)+'\n')
                    # If the object isn't last write a space after
                    else:
                        f.write(str(ele)+' ')

    def test_txt_annot(self,txt_dir,img_dir):
        txts = [f for f in os.listdir(txt_dir) if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(?i)(.txt)$', f)]
        txts = [txts[0], txts[1], txts[2]]
        images = [ele[:-4]+".jpg" for ele in txts]

        # Iterate through images and show bounding box
        for img_name in images:
            img_index = images.index(img_name)
            img_path = os.path.join(img_dir,img_name)
            img = cv2.imread(img_path)
            isize = img.shape
            width = isize[1]
            height = isize[0]

            # Read in annotation
            txt_filepath = os.path.join(txt_dir,txts[img_index])
            with open(txt_filepath,'r') as f:
                txt_lines = f.readlines()
            if txt_lines != []:
                # Process each annotation in the file
                for line in txt_lines:
                    class_num, x_cent, y_cent, box_width, box_height = self.parse_txt_line(line)
                    x1 = int(round(float(x_cent)*width -
                                    (float(box_width)*width)/2))
                    x2 = int(round(float(x_cent)*width +
                                        (float(box_width)*width)/2))
                    y1 = int(round(float(y_cent)*height -
                                        (float(box_height)*height)/2))
                    y2 = int(round(float(y_cent)*height +
                                        (float(box_height)*height)/2))
                    print(x1,x2,y1,y2)
                    if class_num == '0':
                        c=(0,255,0)
                    else:
                        c=(0,0,255)
                    cv2.rectangle(img,(x1,y1),(x2,y2),c,4)
                    cv2.putText(img,str(class_num),(x1,y1-4),cv2.FONT_HERSHEY_SIMPLEX,1,c,3)
            img_resize = cv2.resize(img,dsize=(int(isize[1]/3),int(isize[0]/3)))
            cv2.imshow('IMAGE',img_resize)
            cv2.waitKey(0)

class InPlaceAnnotationModifier:
    ''' Modify existing annotations in current location '''

    def __init__(self,dir_to_modify,walk_dir = False):
        '''
        Initialize annotaion modifier with directory containing annotations
        
        Keyword Arguments:
        dir_to_modify (str): Path to directory with annotations
        walk_dir (bool): Indicator whether to walk through subdirectories
        '''
        self.dir = dir_to_modify
        self.walk_dir = walk_dir

    def modify_all_yolo(self,modify_dict):
        '''
        Modify variable in all yolo annotations to the same value
        
        Keywords Arguments:
        modify_dict (dict): Dictionary containing keyword value pair
        '''
        print("Modfying all labels in "+self.dir)
        # Assertions for correct arguemtns
        assert isinstance(modify_dict, dict), "modify_dict argument must be a dictionary instance"
        for key_name in modify_dict.keys():
            assert key_name.lower() in ["class","x1","y1","w","h"], 'modify_dict keys must include one or more of the following ([)"class","x1","y1","w","h")'

        # Analyze trajectory and modify
        if self.walk_dir is True:
            all_dirs = [dir_contents[0] for dir_contents in os.walk(self.dir)]
            for ea_dir in all_dirs:
                dir_files = os.listdir(ea_dir)
                txt_files = [filename for filename in dir_files if filename[-4:] == '.txt']
                for ea_file in txt_files:
                    # Open each file and modify data
                    with open(os.path.join(ea_dir,ea_file),'r') as f:
                        data = [float(x) for x in f.readlines()[0].strip().split()]
                        for key_name in modify_dict.keys():
                            if key_name.lower() == "class":
                                data[0] = modify_dict[key_name]
                            elif key_name.lower() == "x1":
                                data[1] = modify_dict[key_name]
                            elif key_name.lower() == "y1":
                                data[2] = modify_dict[key_name]
                            elif key_name.lower() == "w":
                                data[3] = modify_dict[key_name]
                            elif key_name.lower() == "h":
                                data[4] = modify_dict[key_name]
                        # Error handeling if box exceeds image dimensions
                        _,x1,y1,w,h=data
                        if x1 - w/2 <=0:
                            data[1] = w/2 + 1/3072
                        if x1 + w/2 >= 1:
                            data[1] = 1 - w/2 - 1/3072
                        if y1 - h/2 <= 0:
                            data[2] = h/2 + 1/2048
                        if y1 + h/2 >= 1:
                            data[2] = 1 - h/2 - 1/2048
                        
                    with open(os.path.join(ea_dir,ea_file),'w') as f:
                        self.write_list_to_yolo(data=data,f=f)
        print("... all labels modified complete.")

    
    def write_list_to_yolo(self,data,f):
        '''
        Create a yolo annotation string from the list of data
        
        Keyword Arguments:
        data (list): List containing [[class, x1, y1, w, h],...] to write to yolo file
        f (file): File object to write data to
        '''
        # For each object in the tuple
        for i in range(0,len(data)):
            ele = data[i]
            if i == 0:
                f.write(str(int(ele))+' ')
            # If the object is last in the tuple write a newline
            elif i == 4:
                f.write(str(round(ele,6))+'\n')
            # If the object isn't last write a space after
            else:
                f.write(str(round(ele,6))+' ')

# # Define mapping between yolo and txt labe
# YOLO2VOC_mapping = {0:'Impalement',1:'Impalement',2:'Fill',3:'Other',4:'Other',5:'Other',6:'Other'}
# # Instantiate annotation converter
# A = annotationConverter(num2class_label_mapping=YOLO2VOC_mapping)
# # Convert all annotations in directory
# A.process_YOLOdirectory(in_dir = "D:/Dumitriu_Detection/NN Classification/Labelling/Labels",
#                         out_dir = "D:/Documents/TensorFlow2/workspace/ImpDetectDani/images",
#                         img_dir = "D:/Dumitriu_Detection/Injection Videos/Image_Sequences_8bitJPEG",
#                         copy_imgs = True)


# Define mapping between yolo and txt label
VOC2YOLO_mapping = {'Tip':0}
# Instantiate annotation converter
A = annotationConverter(class2num_label_mapping=VOC2YOLO_mapping)
# Convert all annotations in directory
A.process_XMLdirectory(in_dir = "D:/Documents/TensorFlow2/workspace/TipDetectYolo/images/train",
                        out_dir = "D:/Documents/TensorFlow2/workspace/TipDetectYolo/labels/train",
                        img_dir = "D:/Documents/TensorFlow2/workspace/TipDetectYolo/images/train",
                        copy_imgs = False)
A.process_XMLdirectory(in_dir = "D:/Documents/TensorFlow2/workspace/TipDetectYolo/images/val",
                        out_dir = "D:/Documents/TensorFlow2/workspace/TipDetectYolo/labels/val",
                        img_dir = "D:/Documents/TensorFlow2/workspace/TipDetectYolo/images/val",
                        copy_imgs = False)
A.process_XMLdirectory(in_dir = "D:/Documents/TensorFlow2/workspace/TipDetectYolo/images/test",
                        out_dir = "D:/Documents/TensorFlow2/workspace/TipDetectYolo/labels/test",
                        img_dir = "D:/Documents/TensorFlow2/workspace/TipDetectYolo/images/test",
                        copy_imgs = False)
# # A.test_txt_annot(txt_dir = "D:/Documents/TensorFlow2/workspace/ImpDetectYolo/annot_convert_test",
# #                  img_dir = "D:/Documents/TensorFlow2/workspace/ImpDetectYolo/annot_convert_test")


C = InPlaceAnnotationModifier(dir_to_modify="D:/Documents/TensorFlow2/workspace/TipDetectYolo/labels",walk_dir=True)
box_size = 20
mod_dict = {"class":0,"w":box_size/3072,"h":box_size/2048}
C.modify_all_yolo(mod_dict)


