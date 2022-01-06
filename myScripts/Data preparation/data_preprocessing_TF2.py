import os
import re
import shutil
import math
import random
import glob
import pandas as pd
import io
import xml.etree.ElementTree as ET
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
# import tensorflow.compat.v1 as tf
# from PIL import Image
# from object_detection.utils import dataset_util, label_map_util
# from collections import namedtuple
# from tensorflow._api.v2 import train

class DataPrepper:
    """
    Class to perform various preprocessing steps prior to training Tensor Flow data
    """

    def __init__(self):
        pass

    def get_annotation_format(self,source):
        '''
        Determines annotation format by looking at file extensions in source.

        Keyword Arguments:
        source: String path of directory containing images and annotations

        Returns:
        annot_format: String declaring the annotation format ("yolo" | "pascalvoc" | "inconclusive")
        '''

        # Specifiy directory paths
        source = source.replace('\\', '/')
        assert(os.path.isdir(source)), "Source directory not found. %r" %source

        # Evaluate if source contains txt (Yolo) or xml (PascalVOC) annotations
        txt_boolean = True if len([f for f in os.listdir(source) if ".txt" in f])>0 else False
        xml_boolean = True if len([f for f in os.listdir(source) if ".xml" in f])>0 else False

        # Determine annotation format
        if txt_boolean is True and xml_boolean is False:
            annot_format = "yolo"
        elif xml_boolean is True and txt_boolean is False:
            annot_format = "pascalvoc"
        else:
            annot_format = "inconclusive"

        return annot_format


    
    def partition_data(self, source, dest, test_ratio=0.1, val_ratio=0, copy_data=True):
        '''
        Splits data into test/train data. Can retain original data by
        copying data into new directories. Creates test and train
        directories if they don't exist

        Keyword Arguments:
        source: String path of directory containing images and annotations
        dest: String path of directory where test/train directories are (to be created)
        test_ratio: Float [0 1.0] corresponding ratio of data used for testing
        copy_data: Boolean indicator for whether data is copied or moved

        Returns:
        None
        '''

        # Specifiy directory paths
        source = source.replace('\\', '/')
        assert(os.path.isdir(source)), "Source directory not found. %r" %source
        dest = dest.replace('\\', '/')
        assert(os.path.isdir(dest)), "Destination directory not found. %r" %dest
        train_dir = os.path.join(dest, 'train')
        test_dir = os.path.join(dest, 'test')
        val_dir = os.path.join(dest,'val')

        # Create directories for train/test data if nonexistant
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        if not os.path.exists(val_dir):
            os.makedirs(val_dir)

        # Get list of images and annotations
        images = [f for f in os.listdir(source) if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(?i)(.jpg|.jpeg|.png)$', f)]
        annot_format = self.get_annotation_format(source)
        if annot_format == 'yolo':
            annots = [a for a in os.listdir(source) if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(?i)(.txt)$', a)]
            annot_ext = ".txt"
        elif annot_format == 'pascalvoc':
            annots = [a for a in os.listdir(source) if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(?i)(.xml)$', a)]
            annot_ext = ".xml"
        else:
            annots = []
            annot_ext = None
        
        # Intersect images and annotations to find common elements (only images that have been annotated)
        image_names = [f[:f.find(".")] for f in images]
        annot_names = [a[:a.find(".")] for a in annots]
        image_annot_intersect = list(set(image_names).intersection(annot_names))

        # Modify images and annotations list to only include images that have been annotated
        if not len(image_annot_intersect) == len(image_names) == len(annot_names):
            images = [f for f in images if f[:f.find(".")] in image_annot_intersect]
        
        # Compute number of test images based on split test_ratio
        num_images = len(image_annot_intersect)
        num_test_images = math.ceil(test_ratio*num_images)
        num_val_images = math.ceil(val_ratio*num_images)

        # Copy (or move) images/annotations to test and train folders if the test/train dirs are empty
        if len(image_annot_intersect) != 0:
            if len(os.listdir(test_dir)) == 0 & len(os.listdir(train_dir)) == 0:
                # Copy (or move) images/annotations to test folder
                for i in range(num_test_images):
                    idx = random.randint(0, len(images)-1)
                    filename = images[idx]
                    if copy_data is True:
                        shutil.copyfile(os.path.join(source, filename),
                                os.path.join(test_dir, filename))
                        annot_filename = os.path.splitext(filename)[0]+annot_ext
                        shutil.copyfile(os.path.join(source, annot_filename),
                                os.path.join(test_dir,annot_filename))
                    else:
                        shutil.move(os.path.join(source, filename),
                                os.path.join(test_dir, filename))
                        annot_filename = os.path.splitext(filename)[0]+annot_ext
                        shutil.move(os.path.join(source, annot_filename),
                                os.path.join(test_dir,annot_filename))
                    images.remove(images[idx])
                
                # Copy (or move) images/annotations to val folder
                for i in range(num_val_images):
                    idx = random.randint(0, len(images)-1)
                    filename = images[idx]
                    if copy_data is True:
                        shutil.copyfile(os.path.join(source, filename),
                                os.path.join(val_dir, filename))
                        annot_filename = os.path.splitext(filename)[0]+annot_ext
                        shutil.copyfile(os.path.join(source, annot_filename),
                                os.path.join(val_dir,annot_filename))
                    else:
                        shutil.move(os.path.join(source, filename),
                                os.path.join(val_dir, filename))
                        annot_filename = os.path.splitext(filename)[0]+annot_ext
                        shutil.move(os.path.join(source, annot_filename),
                                os.path.join(val_dir,annot_filename))
                    images.remove(images[idx])

                # Copy (or move) images/annotations to train folder
                for filename in images:
                    if copy_data is True:
                        shutil.copyfile(os.path.join(source, filename),
                                os.path.join(train_dir, filename))
                        annot_filename = os.path.splitext(filename)[0]+annot_ext
                        shutil.copyfile(os.path.join(source, annot_filename),
                                os.path.join(train_dir,annot_filename))
                    else:
                        shutil.move(os.path.join(source, filename),
                                os.path.join(train_dir, filename))
                        annot_filename = os.path.splitext(filename)[0]+annot_ext
                        shutil.move(os.path.join(source, annot_filename),
                                os.path.join(train_dir,annot_filename))
            else:
                print(
                    "Images and annotations not copied because training and/or testing directory weren't empty")
        else:
            print("Images and annotations not copied because either couldn't determine annotation format, couldn't find matching image and annotation file names, or no images and/or annotations in source directory.")
    
    def compile_xml_data(self, image_source, annot_dest, save_to_csv=True):
        """
        Process xml files in test/train directories and return dataframes of data.

        Keyword Arguments:
        image_source: String path of directory containing images and annotations
        annot_dest: String path of directory for annotations (label map and TFrecords)
        save_to_csv: Boolean indicator for whether a copy of xml data is saved to csv

        Returns:
        train_data: Pandas dataframe of train data
        test_data: Pandas dataframe of test data
        """

        # Specifiy directory paths
        image_source = image_source.replace('\\', '/')
        assert(os.path.isdir(image_source)), "Source directory not found. %r" %image_source
        train_dir = os.path.join(image_source, 'train')
        test_dir = os.path.join(image_source, 'test')
        annot_dest = annot_dest.replace('\\', '/')

        # Verify (or create) annotation directory
        if not os.path.exists(annot_dest):
            os.makedirs(annot_dest)

        # Initialize xml_data
        test_data = None
        train_data = None

        # Iterate through data in test and train to compile
        for paths in [test_dir, train_dir]:
            paths = paths.replace('\\', '/')
            # Get indicator of type of data (test vs train)
            type_of_data = paths[paths.rfind('/')+1:]
            # If testing or training folders missing, don't make CSV
            if not (os.path.exists(train_dir) | os.path.exists(test_dir)):
                print(
                    "XML data not compiled because because missing test or train image source folders.")
            else:
                # Initialize list for storing xml data
                xml_list = []
                # Iterate through each xml file in provided path
                for xml_file in glob.glob(paths + '/*.xml'):
                    # Import data from XML file
                    tree = ET.parse(xml_file)
                    root = tree.getroot()
                    # Get file name and image size
                    filename = root.find('filename').text
                    width = int(root.find('size').find('width').text)
                    height = int(root.find('size').find('height').text)
                    # Iterate thorugh each object root in xml to create a csv line
                    for member in root.findall('object'):
                        bndbox = member.find('bndbox')
                        value = (filename,
                                width,
                                height,
                                member.find('name').text,
                                int(bndbox.find('xmin').text),
                                int(bndbox.find('ymin').text),
                                int(bndbox.find('xmax').text),
                                int(bndbox.find('ymax').text),
                                )
                        xml_list.append(value)
                # Create header for csv fole
                column_name = ['filename', 'width', 'height',
                               'class', 'xmin', 'ymin', 'xmax', 'ymax']
                # Create csv type data structure
                xml_df = pd.DataFrame(xml_list, columns=column_name)
                if type_of_data == 'test':
                    test_data = xml_df
                elif type_of_data == 'train':
                    train_data = xml_df
                
                # Write to csv file
                if save_to_csv is True:
                    xml_df.to_csv(
                        annot_dest+'/{}_labels.csv'.format(type_of_data), index=None)
                    print('Successfully converted {} xml to csv.'.format(type_of_data))
        
        return train_data, test_data

    def create_tf_example(self,group, path, label_map_dict):
        '''
        Create data for a TFrecord

        Keyword:
        group: idk (https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#)
        path: idk (https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#)
        label_map_dict: Dictionary of label map class name assoicated to id

        Returns:
        tf_example: tf.train.Example for TFrecord
        '''

        # Read each jpg in data_path
        with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
            encoded_jpg = fid.read()
        # Convert image to bytes
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        # Open bytes jpg
        image = Image.open(encoded_jpg_io)
        # extract image size
        width, height = image.size
        # encode each file as a utf8 image
        filename = group.filename.encode('utf8')
        # format each image to jpg
        image_format = b'jpg'
        # Initialize TFRecord data placeholders
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []
        
        # Iterate through all the entries in the group (CSV?) assign TFRecord data
        for _, row in group.object.iterrows():
            xmins.append(row['xmin'] / width)
            xmaxs.append(row['xmax'] / width)
            ymins.append(row['ymin'] / height)
            ymaxs.append(row['ymax'] / height)
            classes_text.append(row['class'].encode('utf8'))
            class_label_str = row['class']
            classes.append(self.class_text_to_int(row_label=class_label_str,label_map_dict=label_map_dict))

        # Create TFRecord Entry
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
        return tf_example


    def gen_class_labels(self,train_data,test_data):
        """
        Analyzes data for unique class labels.

        Keyword Arguments:
        train_data: Pandas dataframe of train data
        test_data: Pandas dataframe of test data

        Returns:
        class_labesl: List of unique class names
        """

        # If training labels csv doesn't exist, dont make class labels
        if train_data is None or test_data is None:
            print("Class labels not created because train or test data is None:")
        # Compile unique class names
        else:
            # Concatenate train and test data dataframes
            data_concat = pd.concat([train_data,test_data])
            # Extract list of all class names from the test/train data
            class_labels = list(data_concat.apply(set)['class'])

        return class_labels
    
    def gen_labelmap(self, annot_dest, class_labels):
        """
        Generate/save a label map in annotation directory.
        
        Keyword Arguments:
        annot_dest: String path of directory for annotations (label map and TFrecords)
        class_labes: List of unique class name
        
        Returns:
        label_map_dict: Dictionary associating class names and their unique indices
        """

        # Check if label map exists in training directory already
        if 'label_map.pbtxt' in os.listdir(annot_dest):
            label_map_path = annot_dest + '/label_map.pbtxt'
            print('New label map not created. Label map loaded from',label_map_path)
            # Create label map dictionary from saved labelmap file
            label_map = label_map_util.load_labelmap(label_map_path)
            label_map_dict = label_map_util.get_label_map_dict(label_map)
        else:
            # Check that class labels exist
            if len(class_labels) > 0:
                # Write the data to the label_map file and ssociate unique index with each class 
                label_map_path = annot_dest + '/label_map.pbtxt'
                with open(label_map_path, 'w') as f:
                    for classes in class_labels:
                        f.write('item {\n')
                        f.write(
                            '   id: ' + str(class_labels.index(classes) + 1)+'\n')
                        # If writing last class, don't write new lines to end of file
                        if classes == class_labels[-1]:
                            f.write("   name: '" + str(classes) + "'\n}")
                        else:
                            f.write("   name: '" + str(classes) + "'\n}\n\n")
                print('Data map successfully create in ' + label_map_path)
                # Create label map dictionary from saved labelmap file
                label_map = label_map_util.load_labelmap(label_map_path)
                label_map_dict = label_map_util.get_label_map_dict(label_map)
            else:
                print("Label map not created because empty class labels")
                label_map_dict = None

        return label_map_dict
    
    def create_TFRecords(self,image_source, annot_dest,record_type, label_map_dict, data=None,from_CSV=False,csv_path=None):
        '''
        Create TFrecord from all the available data and saves in annotation directory

        Keyword Arguments:
        image_source: String path of directory containing images and annotations
        annot_dest: String path of directory for annotations (label map and TFrecords)
        record_type: String indicator ('train' or 'test') indicating type of data
        label_map_dict: Dictionary associating class names and their unique indices
        data: Pandas data frame of data for TFrecord
        from_CSV: Boolean indicator for wheter data is supposed to come from CSV. Must set data to None
        csv_path: String path of csv file for TF record data

        Returns:
        None (creates TFrecord in annotation directory)

        '''

        # Confirm existance of annotation destination
        image_source = image_source.replace('\\', '/')
        assert(os.path.isdir(image_source)), "Source directory not found. %r" %image_source
        annot_dest = annot_dest.replace('\\', '/')
        assert(os.path.isdir(annot_dest)), "Destination directory not found. %r" %annot_dest

        # Confirm record type validity
        assert(record_type in ["test","train"]), "record_type must be string of 'train' or 'test'. Not %r" %record_type
        record_full_path = annot_dest+"/"+record_type+'.record'

        # Initialize  TFrecord writer
        writer = tf.python_io.TFRecordWriter(record_full_path)
        path = os.path.join(image_source)

        # Compile data for TF record
        if data is not None:
            examples = data
        elif from_CSV is True and csv_path is not None:
            assert(os.path.exists(csv_path)), "CSV path not found. %r" %csv_path
            examples = pd.read_csv(csv_path)

        # Create tfrecord from data
        grouped = self.split_df_data(examples, 'filename')
        for group in grouped:
            tf_example = self.create_tf_example(group=group, path=path,label_map_dict=label_map_dict)
            writer.write(tf_example.SerializeToString())
        writer.close()
        print('Created '+record_type+' TFrecord. '+record_full_path)

    

    def class_text_to_int(self, row_label, label_map_dict):
        '''
        Helper fcn to get unique class index from label map

        Keyword Arguments:
        row_label: String of class name
        label_map_dict: Dictionary associating class names and their unique indices

        Returns:
        class_index: Unique index (int) of class string
        '''
        assert(label_map_dict is not None), "Label map dictionary is none."
        class_index = label_map_dict[row_label]

        return class_index
    
    def split_df_data(self,df, group):
        '''
        Group pandas dataframe by specific group
        https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html

        df: Pandas dataframe to group
        group: How to group

        Returns:
        idk
        '''
        data = namedtuple('data', ['filename', 'object'])
        gb = df.groupby(group)
        return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

    
    


# P = DataPrepper()
# # Split the data into test/train
# P.partition_data(source="D:/Documents/TensorFlow2/workspace/ImpDetectDani/images",dest="D:/Documents/TensorFlow2/workspace/ImpDetectDani/images",test_ratio=0.1,copy_data=True)
# # Compile all data
# train_data, test_data = P.compile_xml_data(image_source="D:/Documents/TensorFlow2/workspace/ImpDetectDani/images",annot_dest="D:/Documents/TensorFlow2/workspace/ImpDetectDani/annotations",save_to_csv=True)
# # Generate class labels from test/train data
# class_labels = P.gen_class_labels(train_data=train_data, test_data=test_data)
# # Create labelmap file from class labels
# label_map_dict = P.gen_labelmap(annot_dest="D:/Documents/TensorFlow2/workspace/ImpDetectDani/annotations", class_labels=class_labels)
# # Create train TFR record
# P.create_TFRecords(image_source="D:/Documents/TensorFlow2/workspace/ImpDetectDani/images",annot_dest="D:/Documents/TensorFlow2/workspace/ImpDetectDani/annotations",record_type='train',label_map_dict=label_map_dict,data=train_data)
# P.create_TFRecords(image_source="D:/Documents/TensorFlow2/workspace/ImpDetectDani/images",annot_dest="D:/Documents/TensorFlow2/workspace/ImpDetectDani/annotations",record_type='test',label_map_dict=label_map_dict,data=test_data)

P = DataPrepper()
# Split the data into test/train
P.partition_data(source="D:/Documents/TensorFlow2/workspace/ImpDetectYolo/images",dest="D:/Documents/TensorFlow2/workspace/ImpDetectYolo/images",test_ratio=0.1,val_ratio=0.1,copy_data=False)