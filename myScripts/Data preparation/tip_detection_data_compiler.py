import os
from pathlib import Path
from datetime import datetime
import time
import re
import pandas as pd
import numpy as np
import cv2
import math
import shutil
from pascal_voc_writer import Writer
from PIL import Image
from PIL import ImageTk
import PIL.Image
import PIL.ImageTk
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk

class GUIWindow(tk.Frame):
    '''
    Window to display mosaic image and select images
    '''

    def __init__(self, master=None,img_dir=None,img_stack_dir=None,inj_data_dir=None,ml_image_dir=None):
        '''
        Initialize class. Set class attributes for various directories

        Keyword Arguments:
        master: Frame to place GUI
        img_dir: String for directory containing all folders with images corresponding to injection process (EPI Cells)
        img_stack_dir: String for directory containing all image stacks for cell detectin (and annotation info of detected cell)
        inj_data_dir: String for directory containing text file data of success inject cells (impalement success data)
        ml_image_dir: String for directory containing all images for the machine learning model to be trained
        '''
        tk.Frame.__init__(self, master)
        self.master = master
        self.master.title("Data Compiler")
        self.option_add("*Font", "helvetica 9")
        self.IDC = ImpDetCompiler(img_dir=img_dir,
                                  img_stack_dir=img_stack_dir,
                                  inj_data_dir=inj_data_dir,
                                  ml_image_dir = ml_image_dir)
        self.init_vars() # initiate gui variables
        self.layout_gui() # Setup frames
        self.populate_widgets() # add widgets to frames
        self.update_display()

    def init_vars(self):
        self.fully_defined_clicked_indices = [] # Position of clicked indices [[mosaic_num, local_index]]
        self.clicked_tip_indices_and_coords = [] # Position of clicked indices [[mosaic_num, local_index, local_clicked_coord]]
        self.global_clicked_indices = [] # Index of clicked image in global roi images 
        self.pix_per_img_x = 100 # width of each img tile in mosiac
        self.pix_per_img_y = 100 # height of each img tile in mosiac
        self.disp_w = 1700 # Width of canvas dispaly (mosaic)
        self.disp_h = 1000 # height of canvas display (mosaic)
        self.imgs_per_mosaic_x = int(self.disp_w / self.pix_per_img_x) # number of img tiles side to side across mosaic
        self.imgs_per_mosaic_y = int(self.disp_h / self.pix_per_img_y) # number of img tiles top to bottom across mosaic 
        self.imgs_per_mosaic = int(self.imgs_per_mosaic_y*self.imgs_per_mosaic_x) # total number of img tiles in each mosaic
        self.cell_N = 0 # Total cell number
        self.cell_l = 0 # lower cell number
        self.cell_u = 0 # upper cell number
        self.mosaic_N = 0 # Total number of mosaic
        self.mosaic_num = 0 # Current mosaic number
        self.cursor_pos = None # Postion of cursor in canvas
        self.df_read = None # Pandas dataframe for reading indata
        self.df_associate = None # Pandas dataframe for compiled data
        self.df_xml = None # Pandas dataframe for xml annotation
        self.df_tip_coords_yolo = None # Pandas dataframe for yolo tip coords
        self.time_str = None # time string for data
        self.date_str = None # date string for saved data

    def layout_gui(self):
        ''' Layout frames of GUI'''
        # Init frames
        self.frame_display = ttk.LabelFrame(master=self.master, text='Display')
        self.frame_aux1 = ttk.LabelFrame(master=self.master, text='Auxiliary 1')
        self.frame_aux2 = ttk.LabelFrame(master=self.master, text='Auxiliary 2')
        # Place frames
        self.frame_display.grid(row=0,column=0,padx=2,pady=2,rowspan=2,sticky=tk.NSEW)
        self.frame_aux1.grid(row=0,column=1,padx=2,pady=2,sticky=tk.NSEW)
        self.frame_aux2.grid(row=1,column=1,padx=2,pady=2,sticky=tk.NSEW)

    def populate_widgets(self):
        ''' Fill frames with widgets '''
        # Display canvas
        self.canvas_img = np.zeros((self.disp_h,self.disp_w,3),dtype=np.uint8)
        self.canvas_display = tk.Canvas(master=self.frame_display, width=self.disp_w, height=self.disp_h,cursor="none")
        self.canvas_display.grid(row=0,column=0,sticky=tk.NSEW)
        self.canvas_display.bind("<Button 1>", self.img_coord_from_clicked_coord)
        self.canvas_display.bind("<Button 3>", self.rm_img_from_list)
        self.canvas_display.bind("<Motion>", self.mouse_motion)

        # DATA ASSOCIATION WIDGETS
        # Read in data (score varied success rate text files)
        self.read_data_button = tk.Button(master=self.frame_aux1,text='Read Data', command=self.read_data)
        self.read_data_button.grid(row=0,column=0,padx=1,pady=2,sticky=tk.EW)
        # Read in data (score varied success rate text files)
        self.associate_data_button = tk.Button(master=self.frame_aux1,text='Associate Data', command=self.associate_data)
        self.associate_data_button.grid(row=1,column=0,padx=1,pady=2,sticky=tk.EW)
        # Read in data (score varied success rate text files)
        self.process_data_button = tk.Button(master=self.frame_aux1,text='Process Data', command=self.process_data)
        self.process_data_button.grid(row=2,column=0,padx=1,pady=2,sticky=tk.EW)
        self.data_type_dict = {"Impale":1, "Miss":0,"All":-1}
        self.data_type_var = tk.StringVar()
        self.data_type_var.set("All")
        self.data_type_menu = tk.OptionMenu(self.frame_aux1,self.data_type_var,*self.data_type_dict,command=self.reset_clicked_indices)
        self.data_type_menu.grid(row=2,column=1,padx=1,pady=2,sticky=tk.EW)
        # Read in data (score varied success rate text files)
        self.save_data_button = tk.Button(master=self.frame_aux1,text='Save Data', command=self.save_data)
        self.save_data_button.grid(row=3,column=0,padx=1,pady=2,sticky=tk.EW)
        # Move and annotate data
        self.copy_annotate_button = tk.Button(master=self.frame_aux1,text='Copy/Annotate Data', command=self.copy_annotate_data)
        self.copy_annotate_button.grid(row=4,column=0,padx=1,pady=2,columnspan=2,sticky=tk.EW)
        # Move and annotate data
        self.show_tip_button = tk.Button(master=self.frame_aux1,text='Show Tips', command=self.show_tip)
        self.show_tip_button.grid(row=5,column=0,padx=1,pady=2,columnspan=2,sticky=tk.EW)

        # DATA EVALUATION WIDGETS
        # Increment/decrement mosaic
        self.next_mosaic_button = tk.Button(master=self.frame_aux2,text='Next Mosiac', command=self.next_mosaic)
        self.next_mosaic_button.grid(row=0,column=0,padx=1,pady=2,sticky=tk.EW)
        self.prev_mosaic_button = tk.Button(master=self.frame_aux2,text='Prev. Mosiac', command=self.prev_mosaic)
        self.prev_mosaic_button.grid(row=0,column=1,padx=1,pady=2,sticky=tk.EW)
        # Show mosaic and cell number
        cell_label = tk.Label(master=self.frame_aux2,text='Cell #:')
        cell_label.grid(row=1,column=0,padx=1,pady=2,sticky=tk.E)
        self.cell_num_var = tk.StringVar()
        self.cell_num_var.set(str(self.cell_l) + '-' + str(self.cell_u) + ' / ' + str(self.cell_N))
        cell_num_disp = tk.Label(master=self.frame_aux2,textvariable=self.cell_num_var,relief=tk.SUNKEN)
        cell_num_disp.grid(row=1,column=1,padx=1,pady=2,sticky=tk.EW)
        mosaic_label = tk.Label(master=self.frame_aux2,text='Mosaic #:')
        mosaic_label.grid(row=2,column=0,padx=1,pady=2,sticky=tk.E)
        self.mosaic_num_var = tk.StringVar()
        self.mosaic_num_var.set(str(self.mosaic_num) + ' / ' + str(self.mosaic_N))
        mosaic_num_disp = tk.Label(master=self.frame_aux2,textvariable=self.mosaic_num_var,relief=tk.SUNKEN)
        mosaic_num_disp.grid(row=2,column=1,padx=1,pady=2,sticky=tk.EW)

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

    def process_data(self):
        ''' Process data using mosaic dir '''

        self.fully_defined_clicked_indices = [] # Position of clicked indices [[mosaic_num, local_index]]
        self.clicked_tip_indices_and_coords = [] # Position of clicked indices [[mosaic_num, local_index, local_clicked_coord]]
        self.global_clicked_indices = [] # Index of clicked image in global roi images 

        # Use IDC to compile ROI images of data
        impale_indicator = self.data_type_dict[self.data_type_var.get()]
        inj_data_path = Path(self.IDC.inj_data_dir)
        pre_fdir = inj_data_path.parent.absolute()
        filetypes = [("All Files","*.*"),("CSV Files","*.csv"),("Text Documents","*.txt")]
        filepath = filedialog.askopenfilename(initialdir=pre_fdir,filetypes=filetypes,defaultextension="*.csv")
        self.date_str = filepath[filepath.rfind("_Time")-10:filepath.rfind("_Time")]
        self.time_str = filepath[filepath.rfind("_Time")+5:-4]
        if filepath != '':
            self.df_associate, self.df_xml, self.ROI_imgs, self.ROI_bb_scale = self.IDC.evaluate_data(filepath=filepath,impale=impale_indicator,img_num=1,mosaic_img_x=self.pix_per_img_x,mosaic_img_y=self.pix_per_img_y)
            print(self.df_xml)
        
        if self.df_xml is not None:
            # Create mosaic images
            self.mosaic_imgs, self.imgs_per_mosaic_x, self.imgs_per_mosaic_y = self.create_mosaic_images(i_stack=self.ROI_imgs, outsize_x=self.disp_w,outsize_y=self.disp_h)
            # Set mosaic and cell labels
            self.mosaic_N = self.mosaic_imgs.shape[2]
            self.mosaic_num = 1
            self.mosaic_num_var.set(str(self.mosaic_num) + ' / ' + str(self.mosaic_N))
            self.cell_N = len(self.df_xml)
            self.cell_l = self.mosaic_num*self.imgs_per_mosaic - self.imgs_per_mosaic
            self.cell_u = min(self.mosaic_num*self.imgs_per_mosaic, self.cell_N)
            self.cell_num_var.set(str(self.cell_l) + '-' + str(self.cell_u) + ' / ' + str(self.cell_N))
            # Show mosaic image
            self.canvas_img = self.mosaic_imgs[:,:,self.mosaic_num-1]

    def copy_annotate_data(self):
        ''' Move and annotate data '''
        self.IDC.copy_data_and_annotate(df_xml=self.df_xml)
    
    def show_tip(self):
        ''' SHow the tip from the xml file'''
        if self.df_xml is not None:
            for i in range(0,len(self.df_xml.index)):
                img_name = self.df_xml.iloc[i]['ImgName']
                date_str = self.df_xml.iloc[i]['Date']
                X1 = self.df_xml.iloc[i]['X1']
                Y1 = self.df_xml.iloc[i]['Y1']
                if img_name is not None:
                    imp_img_dir = os.path.join(self.IDC.img_dir, date_str)
                    img_src_path = os.path.join(imp_img_dir,img_name)
                    im = cv2.imread(img_src_path)
                    cv2.circle(im,(X1,Y1),10,(0,0,255),-1)
                    cv2.imshow('TIP',cv2.resize(im,(750,500)))
                    cv2.cv2.waitKey(10)

    def create_mosaic_images(self,i_stack,outsize_x,outsize_y):
        '''
        Tile input image ROIs into mosaic and display
        
        Keyword Arguments:
        i_stack: Numpy stack of images to display in mosaic
        outsize_x: Pixel width of mosaic image
        outsize_y: Pixel height of mosaic image

        '''

        # Size if image stack
        iy = i_stack.shape[0]
        ix = i_stack.shape[1]
        iz = i_stack.shape[2]

        # Compute number of ROI in mosaic and required number of mosaic images
        imgs_per_mosaic_x = math.floor(outsize_x/ix)
        imgs_per_mosaic_y = math.floor(outsize_y/iy)
        imgs_per_mosaic = imgs_per_mosaic_x*imgs_per_mosaic_y
        num_mosaic_imgs = math.ceil(iz/imgs_per_mosaic)

        # Init mosaic image to hold tiled ROI
        mosaic_imgs = np.zeros([outsize_y,outsize_x,num_mosaic_imgs],dtype=np.uint8)

        # Iterate through all the ROI and tile them in the mosaic image
        for i in range(0,iz):
            # Find number of current ROI in mosaic image (starting left to right top to bottim)
            tile_num = i - math.floor(i/imgs_per_mosaic)*imgs_per_mosaic
            # X coordinates of ROI in image
            x1 = (tile_num % imgs_per_mosaic_x)*ix
            x2 = x1 + ix
            # Compute row for ROI
            tile_row = math.floor(tile_num/imgs_per_mosaic_x)
            # Y coordiantes of ROI in image
            y1 = (tile_num % imgs_per_mosaic_y)*iy
            y1 = tile_row*iy
            y2 = y1 + iy
            # Add ROI to the mosaic images
            tile_img_ind = math.floor(i/imgs_per_mosaic)
            mosaic_imgs[y1:y2,x1:x2,tile_img_ind] = i_stack[:,:,i]

        return mosaic_imgs, imgs_per_mosaic_x, imgs_per_mosaic_y

    def save_data(self):
        ''' Fucntion to save trimmed and retained data'''

        # Trim the data frame based on user selectiosn
        # df_trim = df.iloc[rem_imgs]
        # df_save = df.drop(labels=rem_imgs)
        # df_xml = df_xml.drop(labels=rem_imgs)
        # ROI_imgs_trim = np.delete(ROI_imgs,rem_imgs,axis=2)
        print(self.global_clicked_indices)
        df_associate_clicked_tip = self.df_associate.iloc[self.global_clicked_indices]
        df_xml_clicked_tip = self.df_xml.iloc[self.global_clicked_indices]

        df_associate_elim = self.df_associate.iloc[self.global_clicked_indices]
        df_associate_retain = self.df_associate.drop(labels=self.global_clicked_indices)
        df_xml_save = self.df_xml.drop(labels=self.global_clicked_indices)

        # File paths to save
        inj_data_path = Path(self.IDC.inj_data_dir)
        pre_fdir = inj_data_path.parent.absolute()
        retain_pre_fname = "tip_data_"+self.data_type_var.get()
        retain_xml_pre_fname = "XML_tip_data_"+self.data_type_var.get()
        retain_fpath = self.save_to_file(pre_fdir=pre_fdir,pre_fname=retain_pre_fname,time_str=self.time_str,date_str=self.date_str,ext='.csv')
        retain_xml_fpath = self.save_to_file(pre_fdir=pre_fdir,pre_fname=retain_xml_pre_fname,time_str=self.time_str,date_str=self.date_str,ext='.csv')

        # Write data to file
        if retain_fpath != '':
            _ = df_associate_clicked_tip.to_csv(retain_fpath)
            print('Saved new csv of retained data',retain_fpath)
        if retain_xml_fpath != '':
            _ = df_xml_clicked_tip.to_csv(retain_xml_fpath)
            print('Saved new csv of retained XML data',retain_xml_fpath)

    def reset_clicked_indices(self,option_val):
        self.df_associate = None
        self.fully_defined_clicked_indices = [] # Position of clicked indices [[mosaic_num, local_index]]
        self.clicked_tip_indices_and_coords = [] # Position of clicked indices [[mosaic_num, local_index, local_clicked_coord]]
        self.global_clicked_indices = [] # Index of clicked image in global roi images 

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

    def mosaic_coord_to_orig_img_coord(self,global_img_index,local_coord):
        ''' Convert the local clicked coord in the mosaic to the coord in the original image'''
        (x1,y1,x2,y2,hpadx,hpady,downscale_x,downscale_y) = self.ROI_bb_scale[global_img_index,:].tolist()
        orig_ROI_coord = (int(round(local_coord[0]*1/downscale_x -hpadx)),int(round(local_coord[1]*1/downscale_y-hpady)))
        orig_clicked_coord = ((int(x1+orig_ROI_coord[0])),int(y1+orig_ROI_coord[1]))
        return orig_clicked_coord

    def img_coord_from_clicked_coord(self, event):
        ''' add clicked image to list '''
        coord = [event.x, event.y]
        mosaic_index = self.mosaic_num - 1
        local_img_index, center_coord, local_coord = self.clicked_coord_to_box(coord)
        fully_defined_indices = (int(mosaic_index), int(local_img_index))
        if fully_defined_indices not in self.fully_defined_clicked_indices and self.mosaic_N > 0:
            self.fully_defined_clicked_indices.append((mosaic_index,local_img_index))
            self.clicked_tip_indices_and_coords.append((mosaic_index,local_img_index,local_coord))
            global_img_index = self.local_to_global(local_index=local_img_index, mosaic_index=mosaic_index,imgs_per_mosaic=self.imgs_per_mosaic)
            self.global_clicked_indices.append(global_img_index)
            X1, Y1 = self.mosaic_coord_to_orig_img_coord(global_img_index=global_img_index,local_coord=local_coord)
            # img_name = self.df_xml.iloc[global_img_index]['ImgName']
            # date_str = self.df_xml.iloc[global_img_index]['Date']
            # if img_name is not None:
            #     imp_img_dir = os.path.join(self.IDC.img_dir, date_str)
            #     img_src_path = os.path.join(imp_img_dir,img_name)
            #     im = cv2.imread(img_src_path)
            #     cv2.circle(im,(X1,Y1),10,(255,255,0),-1)
            #     cv2.imshow('TIP',cv2.resize(im,(750,500)))
            #     cv2.cv2.waitKey(0)
            self.df_xml.at[global_img_index,"X1"] = X1
            self.df_xml.at[global_img_index,"X2"] = X1
            self.df_xml.at[global_img_index,"Y1"] = Y1
            self.df_xml.at[global_img_index,"Y2"] = Y1

        elif fully_defined_indices in self.fully_defined_clicked_indices and self.mosaic_N > 0:
            ind = self.fully_defined_clicked_indices.index(fully_defined_indices)
            self.clicked_tip_indices_and_coords[ind] = (mosaic_index,local_img_index,local_coord)
    
    def rm_img_from_list(self, event):
        ''' remove clicked image from list '''
        coord = [event.x, event.y]
        mosaic_index = self.mosaic_num - 1
        local_img_index, center_coord, _ = self.clicked_coord_to_box(coord)
        fully_defined_indices = (int(mosaic_index), int(local_img_index))
        if fully_defined_indices in self.fully_defined_clicked_indices and self.mosaic_N > 0:
            global_index_to_remove = self.fully_defined_clicked_indices.index(fully_defined_indices)
            self.fully_defined_clicked_indices.pop(global_index_to_remove)
            self.clicked_tip_indices_and_coords.pop(global_index_to_remove)
            self.global_clicked_indices.pop(global_index_to_remove)

    def local_to_global(self, local_index, mosaic_index, imgs_per_mosaic):
        ''' transalte the local index in the mosaic to a global index'''
        global_index = local_index + mosaic_index*imgs_per_mosaic
        return global_index

    def global_to_local(self, global_index, mosaic_index, imgs_per_mosaic):
        ''' translate global index to local index in the mosaic '''
        if global_index < ((mosaic_index+1)*imgs_per_mosaic):
            local_index = None
        else:
            local_index = global_index % ((mosaic_index+1)*imgs_per_mosaic)
        return local_index

    def local_to_coord(self,local,coord=None):
        ''' translate local index to center coordinate '''
        row_num = math.floor(local/self.imgs_per_mosaic_x)
        col_num = local - row_num*self.imgs_per_mosaic_x
        # Center coordinate of clicked img tile
        if coord is None:
            coord_ret = (int(col_num*self.pix_per_img_x + self.pix_per_img_x/2), int(row_num*self.pix_per_img_y + self.pix_per_img_y/2))
        else:
            coord_ret = (int(col_num*self.pix_per_img_x + coord[0]), int(row_num*self.pix_per_img_y + coord[1]))
        return coord_ret

    def refresh_local_clicks(self):
        ''' recompute the already clicked local indices '''
        for indices in self.clicked_tip_indices_and_coords:
            mosaic_index = indices[0]
            local_index = indices[1]
            local_coord = indices[2]
            if mosaic_index == (self.mosaic_num - 1):
                coord = self.local_to_coord(local_index,coord=local_coord)

    def clicked_coord_to_box(self, coord):
        ''' get image tile/box number of clicked coord '''

        # Image tile column and row of clicked coordinates
        col_num = math.floor(coord[0] / self.pix_per_img_x)
        row_num = math.floor(coord[1]/ self.pix_per_img_y)
        # Image number in the current mosaic
        local_img_index = row_num*self.imgs_per_mosaic_x + col_num
        # Local coordiante of clicked img_tile
        coord_local = (int(coord[0] - col_num*self.pix_per_img_x), int(coord[1] - row_num*self.pix_per_img_y))
        # Center coordinate of clicked img tile
        coord_center = (int(col_num*self.pix_per_img_x + self.pix_per_img_x/2), int(row_num*self.pix_per_img_y + self.pix_per_img_y/2))
        
        return local_img_index, coord_center, coord_local

    def draw_x(self,x_c,y_c,width,color='red'):
        ''' Draw x on canvas '''
        x_left = int(x_c - width/2 + .05*width)
        x_right = int(x_c + width/2 - .05*width)
        y_top = int(y_c - width/2 + .05*width)
        y_bot = int(y_c + width/2 - .05*width)
        self.canvas_display.create_line(x_left,y_top,x_right,y_bot,fill=color,width=3)
        self.canvas_display.create_line(x_left,y_bot,x_right,y_top,fill=color,width=3)

    def draw_cursor(self,x_c,y_c,color='grey50'):
        ''' Draw x on canvas '''
        self.canvas_display.create_line(x_c,max(y_c-100,0),x_c,max(y_c-2,0),fill=color,width=1)
        self.canvas_display.create_line(x_c,min(y_c+2,self.disp_h),x_c,min(y_c+100,self.disp_h),fill=color,width=1)
        self.canvas_display.create_line(max(x_c-100,0),y_c,max(x_c-2,0),y_c,fill=color,width=1)
        self.canvas_display.create_line(min(x_c+2,self.disp_w),y_c,min(x_c+100,self.disp_w),y_c,fill=color,width=1)
        # self.canvas_display.create_line(x_c,0,x_c,self.disp_h,fill=color,width=1)
        # self.canvas_display.create_line(0,y_c,self.disp_w,y_c,fill=color,width=1)

    def draw_tip(self,x_c,y_c,color='magenta'):
        ''' Draw tip cross on canvas'''
        self.canvas_display.create_line(x_c,y_c-10,x_c,y_c-2,fill=color,width=1)
        self.canvas_display.create_line(x_c,y_c+2,x_c,y_c+10,fill=color,width=1)
        self.canvas_display.create_line(x_c-10,y_c,x_c-2,y_c,fill=color,width=1)
        self.canvas_display.create_line(x_c+2,y_c,x_c+10,y_c,fill=color,width=1)

    def next_mosaic(self):
        ''' show next mosaic image '''
        # Not on the last mosaic (more mosaics left)
        if self.mosaic_num < self.mosaic_N:
            # Set cell and mosaic indicators
            self.mosaic_num += 1
            self.mosaic_num_var.set(str(self.mosaic_num) + ' / ' + str(self.mosaic_N))
            self.cell_l = self.mosaic_num*self.imgs_per_mosaic - self.imgs_per_mosaic
            self.cell_u = min(self.mosaic_num*self.imgs_per_mosaic, self.cell_N)
            self.cell_num_var.set(str(self.cell_l) + '-' + str(self.cell_u) + ' / ' + str(self.cell_N))
            self.canvas_img = self.mosaic_imgs[:,:,self.mosaic_num-1]
            self.refresh_local_clicks()

    def prev_mosaic(self):
        ''' show previous mosaic image '''
        # Not on the first mosaic (more mosaics left)
        if self.mosaic_num > 0:
            # Set cell and mosaic indicators
            self.mosaic_num -= 1
            self.mosaic_num_var.set(str(self.mosaic_num) + ' / ' + str(self.mosaic_N))
            self.cell_l = self.mosaic_num*self.imgs_per_mosaic - self.imgs_per_mosaic
            self.cell_u = min(self.mosaic_num*self.imgs_per_mosaic, self.cell_N)
            self.cell_num_var.set(str(self.cell_l) + '-' + str(self.cell_u) + ' / ' + str(self.cell_N))
            self.canvas_img = self.mosaic_imgs[:,:,self.mosaic_num-1]
            self.refresh_local_clicks()

    def mouse_motion(self,event):
        self.cursor_pos = (event.x, event.y)

    def update_display(self):
        self.canvas_display.delete("all")
        display_img = PIL.Image.fromarray(self.canvas_img)
        self.display_img_tk = PIL.ImageTk.PhotoImage(image=display_img)
        self.canvas_display.create_image(0, 0, image=self.display_img_tk, anchor=tk.N+tk.W)
        for indices in self.clicked_tip_indices_and_coords:
            mosaic_ind, local_ind, local_coord = indices
            if mosaic_ind == self.mosaic_num-1:
                coord = self.local_to_coord(local=local_ind,coord=local_coord)
                self.draw_tip(coord[0],coord[1])
        if self.cursor_pos is not None:
            x_c, y_c = self.cursor_pos
            self.draw_cursor(x_c,y_c,color="Springgreen2")
        self.refresh_display = self.frame_display.after(10,self.update_display)


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

        return df_cell

        # df_session = pd.DataFrame()
        # sess_dir = "E:/My Drive/Graduate/BSBRL Research/Robotics for Spat Trans/Results/Injection Trials/Individual Trials"
        # sess_files = [a for a in os.listdir(sess_dir) if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(?i)(.csv)$', a)]
        # print('Reading in '+str(len(sess_files))+' session files.')
        # for s in sess_files:
        #     datestr = s[:s.rfind("_")]
        #     datetime_date = datetime.strptime(datestr,"%m-%d-%Y")
        #     session_filepath = os.path.join(sess_dir,s)
        #     if datetime_date >= datetime.strptime("12-13-2021","%m-%d-%Y"):
        #         session_col_remove = ["Unnamed: 0","Date","Vid Name","Speed","Speed Travel","Speed Retract","Speed Poke","Back Pressure","Pressure","Duration","D offset","Maybe","Success Cells","Maybe Cells","Subs. Pipette","ML","Poke","Cut","Purge","Hyst. Comp.","Buzz Dur","Clogged","Include Data", "Note"]
        #         sess = pd.read_csv(session_filepath)
        #         sess = sess.drop(columns=session_col_remove)
        #         sess = sess.rename(columns = {"Success":"Success N"})
        #         print(sess)
        #         sess['Date'] = datetime.strftime(datetime_date,"%Y-%m-%d")
        #         df_session=pd.concat([df_session,sess])
        # df_test = pd.merge(df_cell,df_session,how="inner",on=["Date","Time"])
        # print('Session files done')
        
        # return df_test

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

    def evaluate_data(self, filepath=None, df_associate=None, impale = -1, img_num = 3, mosaic_img_x =100, mosaic_img_y=100):
        '''
        Compile all data and segment/remove data if needed.
        
        Keyword Arguments:
        filepath: String to csv file containing associated data from associate_cell_to_images
        df_associate: Pandas dataframe containing input data frame w/ associated data columns (like cell num)
        impale: Int to indicate whether compile impaled or missed cells
        img_num: Number of impalment image to use (1 should be before inject)
        mosaic_img_x: Pixel width of each tile in mosaic image
        mosaic_img_y: Pixel height of each tile in mosaic image 
        '''

        # Check that both inputs are not passed
        assert(not(filepath is None and df_associate is None)), "Cannot pass input arguments simealtaneously for filename and df_associate"

        # Set dataframe
        if filepath is not None:
            fpath = Path(filepath)
            base_dir = fpath.parent.absolute()
            df = pd.read_csv(filepath)
        else:
            df = df_associate

        # Read in only data that has been successfully injected and reset indices
        if impale == 1:
            success_indicator = 1
            df = df.loc[df['Success'] == success_indicator]
        elif impale == 0:
            success_indicator = 0
            df = df.loc[df['Success'] == success_indicator]
        else:
            pass # Use whole data frame (impaled and missed)
        df = df.reset_index()

        # Init numpy array for mosaic of images
        ROI_imgs = np.zeros([mosaic_img_y,mosaic_img_x,len(df.index)],dtype=np.uint8)
        ROI_bb_scale = -1*np.ones((len(df.index),8)) #[[x1 y1 x2 y2 padx pady downscalex, downscaley], [x1 y1 x2 y2 padx pady downscalex downscale y],...]
        
        # Init list to add to dataframe for xml
        df_xml_column_names = ['Date','ImgName','Width','Height','Depth','Class','X1','X2','Y1','Y2']
        df_xml_list = []
        print('Processing '+str(len(df.index))+' images.\n')     
        # Iterate through all the rows (cells) in dataframe and extract ROI of impalement
        for ind,row in df.iterrows():
            # Get date, cell number and bounding box data
            date_str = row['Date']
            cell_num = row['Cell']
            x1 = row['x1']
            x2 = row['x2']
            y1 = row['y1']
            y2 = row['y2']
            # Path to directory containing injection images for this row/cell's date
            imp_img_dir = os.path.join(self.img_dir, date_str)
            # Compile all image names that correspond to the third image of the specific injection
            img_num_str = str(img_num).zfill(3)
            cell_imp_imgs = [f for f in os.listdir(imp_img_dir) if 'Cell'+str(cell_num).zfill(3)+'_'+img_num_str in f]
            if cell_imp_imgs != []:
                # Read in the image
                print('"\033[F"Finished processing image #... '+str(ind)) 
                img_src_path = os.path.join(imp_img_dir,cell_imp_imgs[0])
                i = cv2.imread(img_src_path,0)
                # Image size and depth
                ix = i.shape[1]
                iy = i.shape[0]
                if len(i.shape) == 2:
                    depth = 1
                else:
                    depth = i.shape[2]
                class_name = "Tip"
                #['ImgSrcPath','Width','Height','Depth','Class','X1','X2','Y1','Y2']
                df_xml_list.append([date_str,cell_imp_imgs[0],ix,iy,depth,class_name,x1,x2,y1,y2])
                # Extract ROI with 50 pixels beyond each side of BB for max size of 300 by 300
                i, bb, pad = self.pad_ROI(i,x1,x2,y1,y2,50,300,300)
                # Downsize ROI to mosaic tile size and add to numpy array for mosaic iamge
                ir = cv2.resize(src=i,dsize=(mosaic_img_x, mosaic_img_y),interpolation=cv2.INTER_AREA)
                scale_x = mosaic_img_x/ i.shape[1]
                scale_y = mosaic_img_y/ i.shape[0]
                ROI_imgs[:,:,ind] = ir
                bb_array = np.array(bb).reshape(-1)
                pad_array = np.array(pad).reshape(-1)
                scale_array = np.array((scale_x,scale_y)).reshape(-1)
                ROI_bb_scale[ind,:] = np.concatenate((bb_array,pad_array,scale_array))
            else:
                #['ImgSrcPath','Width','Height','Depth','Class','X1','X2','Y1','Y2']
                df_xml_list.append([None,None, None, None, None, None,  None, None, None, None])
        # Create data frame from df xml data
        df_xml = pd.DataFrame(df_xml_list,columns=df_xml_column_names)
        df_associate = df
        # Overwrite xml file if it exists
        fpath2 = filepath.replace('\\', '/')
        fname = fpath2[fpath2.rfind('/')+1:]
        xml_fname = "XML_"+fname
        date_time_ID = fpath2[fpath2.rfind("_Time")-10:]
        dir_files = os.listdir(base_dir)
        for f in dir_files:
            if xml_fname in f:
                df_xml = pd.read_csv(os.path.join(base_dir,f))
                print("OVERWRITE XML WITH "+xml_fname)

        return df_associate, df_xml, ROI_imgs, ROI_bb_scale

    def copy_data_and_annotate(self,df_xml):
        for ind,row in df_xml.iterrows():
            # Get destination info
            img_name = row['ImgName']
            date_str = row['Date']
            if img_name is not None:
                imp_img_dir = os.path.join(self.img_dir, date_str)
                img_src_path = os.path.join(imp_img_dir,img_name)
                img_dst_path = os.path.join(self.ml_image_dir,img_name)
                # Copy image to destination
                if not os.path.exists(img_dst_path):
                    shutil.copyfile(img_src_path, img_dst_path)
                    print('Copied image to new directory.',img_dst_path)
                # xml info
                width = int(row['Width'])
                height = int(row['Height'])
                depth = int(row['Depth'])
                class_name = row['Class']
                x1 = int(row['X1'])
                x2 = int(row['X2'])
                y1 = int(row['Y1'])
                y2 = int(row['Y2'])
                # Write xml annotation to destination
                img_name_no_ext = img_name[:img_name.find(".")]
                xml_dst_path = os.path.join(self.ml_image_dir,img_name_no_ext+".xml")
                print("Saved xml.",xml_dst_path)
                writer = Writer(img_dst_path,width,height,depth=depth)
                writer.addObject(name=class_name,xmin=x1,ymin=y1,xmax=x2,ymax=y2)
                writer.save(xml_dst_path)


    def pad_ROI(self,i,x1,x2,y1,y2,expand_pix,xsize,ysize):
        '''
        Extracts image ROI with desired expansion beyond ROI and restricts
        to specic size
        
        Keyword Arguments:
        i: Numpy array of input image
        x1: Bounding box x coordinate of upper left corner
        y1: Bounding box y coordinate of upper left corner
        x2: Bounding box x coordinate of lower right corner
        y2: Bounding box y coordiante of lower right corner
        expand_pix: Number of pixels to expand ROI on each side
        xsize: Pixel width of desired/cropped ROI
        ysize: Pixel height of desired/cropped ROI
        
        Return
        i_ROI: Numpy array of padded/cropped/expanded ROI
        '''

        # Get ROI upper/lower corner coordinates
        x1 = max(x1-expand_pix,0)
        x2 = min(x2+expand_pix,i.shape[1]-1)
        y1 = max(y1-expand_pix,0)
        y2 = min(y2+expand_pix,i.shape[0]-1)

        # Crop image to ROI
        i_ROI = i[y1:y2,x1:x2]

        # Size of ROI
        i_ROIx = i_ROI.shape[1]
        i_ROIy = i_ROI.shape[0]

        # Compress or pad y axis to desired size
        if i_ROIy > ysize:
            y1 = y1 + int((i_ROIy-ysize)/2) # actual y1 is contracted slightly towards center of ROI
            y2 = y2 - int((i_ROIy-ysize)/2) # actual y1 is contracted slightly towards center of ROI
            i_ROI = i_ROI[int(i_ROIy/2 - ysize/2):int(i_ROIy/2 + ysize/2),:]
            half_pady = 0
        elif i_ROIy < ysize:
            i0 = np.zeros([ysize,i_ROIx],dtype=np.uint8)
            i0[int(ysize/2 - i_ROIy/2):int(ysize/2 + i_ROIy/2),:] = i_ROI
            i_ROI = i0
            half_pady = int(round((ysize - i_ROIy)/2))
        else:
            half_pady=0

        # Compress or pad x axis to desired size
        if i_ROIx > xsize:
            x1 = x1 + int((i_ROIx-xsize)/2) # actual x1 is contracted slightly towards center of ROI
            x2 = x2 - int((i_ROIx-xsize)/2) # actual x1 is contracted slightly towards center of ROI
            i_ROI = i_ROI[:,int(i_ROIx/2 - xsize/2):int(i_ROIx/2 + xsize/2)]
            half_padx = 0
        elif i_ROIx < xsize:
            i0 = np.zeros([ysize,xsize],dtype=np.uint8)
            i0[:,int(xsize/2 - i_ROIx/2):int(xsize/2 + i_ROIx/2)] = i_ROI
            i_ROI = i0
            half_padx = int(round((xsize - i_ROIx)/2))
        else:
            half_padx=0


        bb = (x1, y1, x2, y2)
        pad = (half_padx, half_pady)
        return i_ROI, bb, pad

root = tk.Tk()
app = GUIWindow(master=root,
                img_dir="E:/My Drive/Graduate/BSBRL Research/Robotics for Spat Trans/Results/Pictures/EPI Cells",
                img_stack_dir="E:/My Drive/Graduate/BSBRL Research/Robotics for Spat Trans/Results/Pictures/Image Stacks/EPI",
                inj_data_dir="E:/My Drive/Graduate/BSBRL Research/Robotics for Spat Trans/Results/Data Association/txt score files",
                ml_image_dir = "D:/Documents/TensorFlow2/workspace/TipDetectYolo/images")
app.mainloop()