import os
import re
import pandas as pd
import time
from pathlib import Path
import numpy as np
import cv2
from datetime import datetime
from tkinter import filedialog

class ImpDetCompiler:
    '''
    Class to associate between different saved data types and prepare data for
    impalement training in tensorflow
    '''

    def __init__(self,img_dir,img_stack_dir,inj_data_dir,ml_image_dir):
        '''
        Initialize class. Set class attributes for various directories

        Keyword Arguments:
        img_dir: String for directory containing all folders with images corresponding to injection process (EPI Cells)
        img_stack_dir: String for directory containing all image stacks for cell detectin (and annotation info of detected cell)
        inj_data_dir: String for directory containing text file data of success inject cells (impalement success data)
        ml_image_dir: String for directory containing all images for the machine learning model to be trained
        '''

        # Verify input directories and set attributes
        self.img_dir = img_dir.replace('\\', '/')
        assert(os.path.isdir(self.img_dir)), "Image directory not found. (%r)" %self.img_dir
        self.img_stack_dir = img_stack_dir.replace('\\', '/')
        assert(os.path.isdir(self.img_stack_dir)), "Image stack directory not found. (%r)" %self.img_stack_dir
        self.inj_data_dir = inj_data_dir.replace('\\', '/')
        assert(os.path.isdir(self.inj_data_dir)), "Injection data directory not found. (%r)" %self.inj_data_dir
        self.ml_image_dir = ml_image_dir.replace('\\', '/')
        assert(os.path.isdir(self.ml_image_dir)), "Machine learning workspace image directory not found. (%r)" %self.ml_image_dir

    def read_injection_data(self):
        '''
        Read in all files corresponding to impalement/injection success.
        
        Iterates through all text files in directory and appends data to
        a pandas dataframe.
        
        Keyword Arguments:
        None

        Returns:
        df_cell: Pandas data frame containing all txt file data about individual cell.
        '''

        # Init data frame to return
        df_cell = pd.DataFrame()

        # Compile all text files in directory (should be only files w/ impalement data)
        txt_files = [a for a in os.listdir(self.inj_data_dir) if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(?i)(.txt)$', a)]

        # Iterate through all text files and append data to dataframe
        for t in txt_files:
            # Get date and time from filename
            date_str = t[t.find("Impalement_Successes_")+21:t.find("_Time")]
            time_str = t[t.find("_Time")+5:t.find(".txt")].replace("-",":")
            # Read in data, remove extraneous columns, and add file date and time info
            columns_to_remove = ['R2', 'Amp_var', 'Mean_var', 'Std_var', 'Amp_fit', 'Mean_fit', 'Std_fit', 'Max_ten_var', 'Max_ten_frame', 'Cell_inten_avg', 'Cell_inten_std', 'Bg_inten_avg', 'Bg_inten_std', 'Cell_size', 'Norm_var', 'Vol_auto']
            data = pd.read_csv(os.path.join(self.inj_data_dir,t),sep="\t")
            data = data.drop(columns=columns_to_remove)
            data['Date'] = [date_str]*len(data.index)
            data['Time'] = [time_str]*len(data.index)
            # Append text file data to dataframe to return
            df_cell = pd.concat([df_cell,data])
        print('Read in text files. Number of files:',len(txt_files))

        # return df_cell

        df_session = pd.DataFrame()
        sess_dir = "E:/My Drive/Graduate/BSBRL Research/Robotics for Spat Trans/Results/Injection Trials/Individual Trials"
        sess_files = [a for a in os.listdir(sess_dir) if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(?i)(.csv)$', a)]
        print('Reading in '+str(len(sess_files))+' session files.')
        for s in sess_files:
            datestr = s[:s.rfind("_")]
            datetime_date = datetime.strptime(datestr,"%m-%d-%Y")
            session_filepath = os.path.join(sess_dir,s)
            if datetime_date >= datetime.strptime("08-10-2021","%m-%d-%Y"):
                session_col_remove = ["Unnamed: 0","Date","Vid Name","Speed","Speed Travel","Speed Retract","Speed Poke","Back Pressure","Pressure","Duration","D offset","Maybe","Success Cells","Maybe Cells","Subs. Pipette","ML","Poke","Cut","Purge","Hyst. Comp.","Buzz Dur","Clogged","Include Data", "Note"]
                sess = pd.read_csv(session_filepath)
                sess = sess.drop(columns=session_col_remove)
                sess = sess.rename(columns = {"Success":"Success N"})
                print(sess)
                sess['Date'] = datetime.strftime(datetime_date,"%Y-%m-%d")
                df_session=pd.concat([df_session,sess])
        df_test = pd.merge(df_cell,df_session,how="inner",on=["Date","Time"])
        print('Session files done')
        
        return df_test

    def associate_cell_to_image(self,df_imp,save_data_to_csv=True):
        '''
        Associate between different data types finding the cell number and its 
        coresponding injection status and associated images. Save csv file if desired.

        for dates in unique_injection_dates
            compile all impalment data for that date
            read in cell number/lcoation txt files for that date
            compile into list of cell number and position
            for inj trial in unique injection trials
                read in injection trial detected cell data
                match bounding boxes between impalement data in injection trial data
                for found matches
                    match gui centroid location between prev match and cell number/location
                    find cell number or number options based off xgui coords

        Keyword Arguments:
        df_imp: Pandas dataframe containing all impalement data from text files.
        save_data_to_csv: Boolean indicator for wheter to save associated data to csv file.

        Returns:
        df_associate: Pandas dataframe containing input data frame w/ associated data columns (like cell num)
        '''

        # Init dataframe to return
        df_associate = pd.DataFrame()

        # Dates of injections
        unique_dates = df_imp['Date'].unique()

        # Iterate through dates in impalement detection data
        for dates in unique_dates:
            print('Associating data from:',dates,'...',end=' ')
            # Extract sub-dataframe for impalement data on specific date
            t0 = time.time()
            t1 = time.time()
            df_imp_date = df_imp.loc[df_imp['Date'] == dates]
            # Directory to text file in injection image directory that contains cell number and its xy location
            cell_text_file_dir = os.path.join(self.img_dir,dates,'Annotations')
            # Compile all text files for cell numbers and their locations for specific date
            if os.path.isdir(cell_text_file_dir):
                cell_txt_files = [a for a in os.listdir(cell_text_file_dir) if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(?i)(.txt)$', a)]
                # Init data frame for cell number and locatoin
                df_cell = pd.DataFrame(columns=['X_GUI','Y_GUI','Cell'])
                # Iterate through all cell number/location txt files
                t1 = time.time()
                for t in cell_txt_files:
                    if not 'ImpDet' in t:
                        # Get date and time of text file modification/creation
                        datetime_str = datetime.fromtimestamp(os.path.getmtime(os.path.join(cell_text_file_dir,t))).strftime('%Y-%m-%d %H:%M:%S')
                        date_str = datetime_str[:datetime_str.find(" ")]
                        time_str = datetime_str[datetime_str.find(" ")+1:]
                        # Get cell number from filename
                        cell_num = int(t[t.find("Cell")+4:t.find(".txt")])
                        # Read location data in text file and add cell num,time, and date to dataframe
                        data = pd.read_csv(os.path.join(cell_text_file_dir,t),sep=' ',names=['X_GUI','Y_GUI'])
                        data['Cell'] = [cell_num]*len(data.index)
                        data['Date_Cell'] = [date_str]*len(data.index)
                        data['Time_Cell'] = [time_str]*len(data.index)
                        # Concatenate individual cell data to all cell data
                        df_cell = pd.concat([df_cell,data])
                t1 =time.time()
                # Unique injection trials on specific date
                unique_slices = df_imp_date['Stack_id'].unique()
                # Iterate through all injection trials on specific date
                for slices in unique_slices:
                    # Extract dub data frame corresponding to impalement data for specfic date and injection trial
                    df_imp_date_slice = df_imp_date.loc[df_imp_date['Stack_id'] == slices]
                    # Rename some columns to match impalement data to detected cell stack data
                    df_imp_date_slice =df_imp_date_slice.rename(columns={'BB_x1':'x1','BB_x2':'x2','BB_y1':'y1','BB_y2':'y2'})
                    # Read in detected cell data in image stack files
                    mip_file_dir = os.path.join(self.img_stack_dir,date_str,'MIP')
                    df_stack = pd.read_csv(os.path.join(mip_file_dir,slices+'_values.txt'),sep='\t')
                    # Find the intersection of impalement data and image stack data based on bounding box location (matching rows)
                    # if dates == "2021-11-12":
                    #     print(df_imp_date_slice)
                    #     print(df_stack)
                    df_merge = pd.merge(df_imp_date_slice,df_stack,on=['x1','x2','y1','y2'],how='inner')
                    # Find intersection of impalemnt/image stack data and cell data based on cell centroid location in GUI (matching rows)
                    # These intersections/matchings assoicate impalement data with detected cell data with targeted cell data
                    df_merge2 = pd.merge(df_merge,df_cell,on=['X_GUI','Y_GUI'],how='inner')
                    # Add intersection data to data frame to return
                    df_associate = pd.concat([df_associate,df_merge2])
            print('finished\t'+str(round(time.time() - t0,4))+'s')

        # Save data to csv file if desired
        print(df_associate)
        
        return df_associate

class DataProcessor:
    def __init__(self, img_dir=None,img_stack_dir=None,inj_data_dir=None,ml_image_dir=None):
        self.IDC = ImpDetCompiler(img_dir=img_dir,
                                  img_stack_dir=img_stack_dir,
                                  inj_data_dir=inj_data_dir,
                                  ml_image_dir = ml_image_dir)
        self.read_data()
        self.associate_data()

    def read_data(self):
        ''' Use impale detect compiler to read in data '''
        print('Using IDC to read injection data...')
        self.df_read = self.IDC.read_injection_data()
        print('... read data done!')

        # Ask to save the file
        inj_data_path = Path(self.IDC.inj_data_dir)
        pre_fdir = inj_data_path.parent.absolute()
        fpath =self.save_to_file(pre_fdir=pre_fdir,pre_fname="read_data",ext='.csv')
        print(fpath)

        if fpath is not None and fpath != [] and fpath != '':
            _ = self.df_read.to_csv(fpath)
            print('Saved new csv of associated data',fpath)

    def associate_data(self):
        ''' Associate across different data types using impale detect compiler '''
        print('Using IDC to associate data...')
        self.df_associate = self.IDC.associate_cell_to_image(df_imp=self.df_read)
        print('... associate data done!')

        # Ask to save the file
        inj_data_path = Path(self.IDC.inj_data_dir)
        pre_fdir = inj_data_path.parent.absolute()
        fpath =self.save_to_file(pre_fdir=pre_fdir,pre_fname="associated_data",ext='.csv')

        if fpath is not None and fpath != [] and fpath != '':
            _ = self.df_associate.to_csv(fpath)
            print('Saved new csv of associated data',fpath)

    def save_to_file(self,pre_fdir=None,pre_fname=None,pre_fpath=None,time_str=None,date_str=None,ext='*.*'):
        ''' helper fcn to save to file 
        
        Returns:
        filepath (str): Path to file save location
        '''

        # Make sure at least pass directory or path
        assert(pre_fdir is not None or pre_fpath is not None), "No file directory or path passed to save file"

        # Create file suffix
        if time_str is not None and date_str is not None:
            fname_ending = "_"+date_str+"_Time"+time_str+ext
        else:
            self.time_str = datetime.now().strftime("%H-%M-%S")
            self.date_str = time.strftime("%Y-%m-%d")
            fname_ending = "_"+self.date_str+"_Time"+self.time_str+ext
        # Initial save locatoin
        if pre_fdir is not None and pre_fname is not None:
            initialdir = pre_fdir
            initialfile = pre_fname+fname_ending
            # initialfile = os.path.join(pre_fdir,pre_fname+fname_ending)
        elif pre_fdir is not None:
            pre_fname = "RENAME_FILE"
            initialfile = os.path.join(pre_fdir,pre_fname+fname_ending)
        else:
            initialfile=pre_fpath

        filetypes = [("CSV Files","*.csv"),("All Files","*.*"),("Text Documents","*.txt")]
        filepath = filedialog.asksaveasfilename(title='Save file?',initialdir=initialdir,initialfile=initialfile,defaultextension=ext,filetypes=filetypes)
        
        return filepath

    # def evaluate_data(self, filepath=None, df_associate=None, impale = True,mosaic_img_x =100, mosaic_img_y=100):
    #     '''
    #     Compile all data and segment/remove data if needed.
        
    #     Keyword Arguments:
    #     filepath: String to csv file containing associated data from associate_cell_to_images
    #     df_associate: Pandas dataframe containing input data frame w/ associated data columns (like cell num)
    #     impale: Boolean to indicate whether compile impaled or missed cells
    #     mosaic_img_x: Pixel width of each tile in mosaic image
    #     mosaic_img_y: Pixel height of each tile in mosaic image 
    #     '''

    #     # Check that both inputs are not passed
    #     assert(not(filepath is None and df_associate is None)), "Cannot pass input arguments simealtaneously for filename and df_associate"

    #     # Set dataframe
    #     if filepath is not None:
    #         fpath = Path(filepath)
    #         base_dir = fpath.parent.absolute()
    #         df = pd.read_csv(filepath)
    #     else:
    #         df = df_associate

    #     # Read in only data that has been successfully injected and reset indices
    #     if impale == True:
    #         success_indicator = 1
    #     else:
    #         success_indicator = 0
    #     df = df.loc[df['Success'] == success_indicator]
    #     df = df.reset_index()

    #     # Init numpy array for mosaic of images
    #     ROI_imgs = np.zeros([mosaic_img_y,mosaic_img_x,len(df.index)],dtype=np.uint8)
        
    #     # Init list to add to dataframe for xml
    #     df_xml_column_names = ['Date','ImgName','Width','Height','Depth','Class','X1','X2','Y1','Y2']
    #     df_xml_list = []
    #     print('Processing '+str(len(df.index))+' images.\n')     
    #     # Iterate through all the rows (cells) in dataframe and extract ROI of impalement
    #     for ind,row in df.iterrows():
    #         # Get date, cell number and bounding box data
    #         date_str = row['Date']
    #         cell_num = row['Cell']
    #         x1 = row['x1']
    #         x2 = row['x2']
    #         y1 = row['y1']
    #         y2 = row['y2']
    #         # Path to directory containing injection images for this row/cell's date
    #         imp_img_dir = os.path.join(self.img_dir, date_str)
    #         # Compile all image names that correspond to the third image of the specific injection
    #         cell_imp_imgs = [f for f in os.listdir(imp_img_dir) if 'Cell'+str(cell_num).zfill(3)+'_003' in f]
    #         if cell_imp_imgs != []:
    #             # Read in the image
    #             print('"\033[F"Finished processing image #... '+str(ind)) 
    #             img_src_path = os.path.join(imp_img_dir,cell_imp_imgs[0])
    #             i = cv2.imread(img_src_path,0)
    #             # Image size and depth
    #             ix = i.shape[1]
    #             iy = i.shape[0]
    #             if len(i.shape) == 2:
    #                 depth = 1
    #             else:
    #                 depth = i.shape[2]
    #             if row['Success'] == 1:
    #                 class_name = "Impalement"
    #             else:
    #                 class_name = "Miss"
    #             #['ImgSrcPath','Width','Height','Depth','Class','X1','X2','Y1','Y2']
    #             df_xml_list.append([date_str,cell_imp_imgs[0],ix,iy,depth,class_name,x1,x2,y1,y2])
    #             # Extract ROI with 50 pixels beyond each side of BB for max size of 300 by 300
    #             i = self.pad_ROI(i,x1,x2,y1,y2,50,300,300)
    #             # Downsize ROI to mosaic tile size and add to numpy array for mosaic iamge
    #             ir = cv2.resize(src=i,dsize=(mosaic_img_x, mosaic_img_y),interpolation=cv2.INTER_AREA)
    #             ROI_imgs[:,:,ind] = ir
    #         else:
    #             #['ImgSrcPath','Width','Height','Depth','Class','X1','X2','Y1','Y2']
    #             df_xml_list.append([None,None, None, None, None, None,  None, None, None, None])
    #     # Create data frame from df xml data
    #     df_xml = pd.DataFrame(df_xml_list,columns=df_xml_column_names)
    #     df_associate = df

    #     return df_associate, df_xml, ROI_imgs

    # def copy_data_and_annotate(self,df_xml):
    #     for ind,row in df_xml.iterrows():
    #         # Get destination info
    #         img_name = row['ImgName']
    #         date_str = row['Date']
    #         if img_name is not None:
    #             imp_img_dir = os.path.join(self.img_dir, date_str)
    #             img_src_path = os.path.join(imp_img_dir,img_name)
    #             img_dst_path = os.path.join(self.ml_image_dir,img_name)
    #             # Copy image to destination
    #             if not os.path.exists(img_dst_path):
    #                 shutil.copyfile(img_src_path, img_dst_path)
    #                 print('Copied image to new directory.',img_dst_path)
    #             # xml info
    #             width = int(row['Width'])
    #             height = int(row['Height'])
    #             depth = int(row['Depth'])
    #             class_name = row['Class']
    #             x1 = int(row['X1'])
    #             x2 = int(row['X2'])
    #             y1 = int(row['Y1'])
    #             y2 = int(row['Y2'])
    #             # Write xml annotation to destination
    #             img_name_no_ext = img_name[:img_name.find(".")]
    #             xml_dst_path = os.path.join(self.ml_image_dir,img_name_no_ext+".xml")
    #             print("Saved xml.",xml_dst_path)
    #             writer = Writer(img_dst_path,width,height,depth=depth)
    #             writer.addObject(name=class_name,xmin=x1,ymin=y1,xmax=x2,ymax=y2)
    #             writer.save(xml_dst_path)


    # def pad_ROI(self,i,x1,x2,y1,y2,expand_pix,xsize,ysize):
    #     '''
    #     Extracts image ROI with desired expansion beyond ROI and restricts
    #     to specic size
        
    #     Keyword Arguments:
    #     i: Numpy array of input image
    #     x1: Bounding box x coordinate of upper left corner
    #     y1: Bounding box y coordinate of upper left corner
    #     x2: Bounding box x coordinate of lower right corner
    #     y2: Bounding box y coordiante of lower right corner
    #     expand_pix: Number of pixels to expand ROI on each side
    #     xsize: Pixel width of desired/cropped ROI
    #     ysize: Pixel height of desired/cropped ROI
        
    #     Return
    #     i_ROI: Numpy array of padded/cropped/expanded ROI
    #     '''

    #     # Get ROI upper/lower corner coordinates
    #     x1 = max(x1-expand_pix,0)
    #     x2 = min(x2+expand_pix,i.shape[1]-1)
    #     y1 = max(y1-expand_pix,0)
    #     y2 = min(y2+expand_pix,i.shape[0]-1)

    #     # Crop image to ROI
    #     i_ROI = i[y1:y2,x1:x2]

    #     # Size of ROI
    #     i_ROIx = i_ROI.shape[1]
    #     i_ROIy = i_ROI.shape[0]

    #     # Compress or pad y axis to desired size
    #     if i_ROIy > ysize:
    #         i_ROI = i_ROI[int(i_ROIy/2 - ysize/2):int(i_ROIy/2 + ysize/2),:]
    #     elif i_ROIy < ysize:
    #         i0 = np.zeros([ysize,i_ROIx],dtype=np.uint8)
    #         i0[int(ysize/2 - i_ROIy/2):int(ysize/2 + i_ROIy/2),:] = i_ROI
    #         i_ROI = i0

    #     # Compress or pad x axis to desired size
    #     if i_ROIx > xsize:
    #         i_ROI = i_ROI[:,int(i_ROIx/2 - xsize/2):int(i_ROIx/2 + xsize/2)]
    #     elif i_ROIx < xsize:
    #         i0 = np.zeros([ysize,xsize],dtype=np.uint8)
    #         i0[:,int(xsize/2 - i_ROIx/2):int(xsize/2 + i_ROIx/2)] = i_ROI
    #         i_ROI = i0

    #     return i_ROI 

app = DataProcessor(img_dir="E:/My Drive/Graduate/BSBRL Research/Robotics for Spat Trans/Results/Pictures/EPI Cells",
                img_stack_dir="E:/My Drive/Graduate/BSBRL Research/Robotics for Spat Trans/Results/Pictures/Image Stacks/EPI",
                inj_data_dir="E:/My Drive/Graduate/BSBRL Research/Robotics for Spat Trans/Results/Data Association/txt score files",
                ml_image_dir = "D:/Documents/TensorFlow2/workspace/ImpDetectYolo/images")
