# -*- coding: utf-8 -*- 
import win32api, win32con, win32gui, win32ui
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageChops, ImageOps
from twisted.internet import reactor
from twisted.internet.task import LoopingCall
import cv2 as cv
import cv2
import os
import time
import numpy  
import _thread as thread
import subprocess
import re
get_screenshot = False
GAME_RECT = {'x0': 35, 'y0': 42, 'dx': 384, 'dy': 448}
global centerx, centery
def take_screenshot(x0, y0, dx, dy):
    """
    Takes a screenshot of the region of the active window starting from
    (x0, y0) with width dx and height dy.
    """

    hwnd = win32gui.GetForegroundWindow()   # Window handle
    wDC = win32gui.GetWindowDC(hwnd)        # Window device context
    dcObj = win32ui.CreateDCFromHandle(wDC)
    cDC = dcObj.CreateCompatibleDC()

    dataBitMap = win32ui.CreateBitmap()     # PyHandle object
    dataBitMap.CreateCompatibleBitmap(dcObj, dx, dy)
    cDC.SelectObject(dataBitMap)
    cDC.BitBlt((0,0),(dx, dy) , dcObj, (x0, y0), win32con.SRCCOPY)
    image = dataBitMap.GetBitmapBits(1)

    dcObj.DeleteDC()
    cDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, wDC)
    win32gui.DeleteObject(dataBitMap.GetHandle())
    get_screenshot = True
    return Image.frombuffer("RGBA", (384, 448), image, "raw", "RGBA", 0, 1)


class Radar(object):
    def __init__(self, hit_x, hit_y):
        self.x0 = GAME_RECT['x0']
        self.y0 = GAME_RECT['y0']
        self.dx = GAME_RECT['dx']
        self.dy = GAME_RECT['dy']
        # TODO: Keep updating center to match character's hitbox
        global centerx, centery
        centerx, centery = (hit_x, hit_y)
        self.recommend = True
        self.center_x, self.center_y = (hit_x, hit_y)
        self.apothem = 50         # Distance within which to check for hostiles
        self.curr_fov = take_screenshot(self.x0, self.y0, self.dx, self.dy)
        self.obj_dists = (np.empty(0), np.empty(0))  # distances of objects in fov
        self.blink_time = .01           # Pause between screenshots
        self.diff_threhold = 90        # Diffs above this are dangerous
        # TODO: Call self.scan_fov only when self.curr_fov is updated)
        self.scanner = LoopingCall(self.scan_fov)
        self.boxlist = [[],[],[],[]]
    def update_fov(self):
        """Takes a screenshot and makes it the current fov."""
        # TODO: Only need to record the part we actually examine in scan_fov
        self.curr_fov = take_screenshot(self.x0, self.y0, self.dx, self.dy)
        #img = numpy.array(self.curr_fov) 
        

        
        # self.curr_fov.show()

    def get_diff(self):
        """Takes a new screenshots and compares it with the current one."""
        # time.sleep(.03) # TODO: Make this non-blocking
        old_fov = self.curr_fov
        # old_fov.show()
        self.update_fov()  
        # self.curr_fov.show()
        diff_img = ImageChops.difference(old_fov, self.curr_fov)
        #diff_img.show()
        return ImageOps.grayscale(diff_img)

    def scan_fov(self):
        """
        Updates self.object_locs with a NumPy array of (x, y) coordinates
        (in terms of the current fov) of detected objects.
        """
        diff_array = np.array(self.get_diff())
        #is_visited = [[False for x in range(diff_array.shape[0])] for y in range(diff_array.shape[1])]
        # Get the slice of the array representing the fov
        # NumPy indexing: array[rows, cols]
        global centerx, centery
        x = int(self.center_x)
        y = int(self.center_y)
        #print("center",x, y)
        #"""
        minx, miny = max(1,y-5), min(y+5,diff_array.shape[0])
        maxx, maxy = max(1,x-5), min(x+5,diff_array.shape[1])
        for i in range (max(1,y-5),min(y+5,int(diff_array.shape[0]))):
            for j in range (max(1,x-5),min(x+5,diff_array.shape[1])):
                if diff_array[i,j] > 80 :
                    #"""
        #dis = 10
        #maxy, miny = max(1,y-dis), min(y+dis,diff_array.shape[0])
        #maxx, minx = max(1,x-dis), min(x+dis,diff_array.shape[1])

        #for i in range (max(1,y-dis),min(y+dis,diff_array.shape[0])):
        #    for j in range (max(1,x-dis),min(x+dis,diff_array.shape[1])):
        #        if diff_array[i,j] > 50 :
                    minx = min(minx, j)
                    maxx = max(maxx, j)
                    miny = min(miny, i)
                    maxy = max(maxy, i)
        #print("minmax",minx,maxx,miny,maxy)
        self.center_x = centerx = (minx + maxx) / 2
        self.center_y = centery = (miny + maxy) / 2
        #print("fix",x-centerx, y-centery, centerx, centery)
        x = centerx
        y = centery
        apothem = self.apothem
        # Look at front, left, and right of hitbox
        fov_array = diff_array[int(x)-apothem:int(x)+apothem, int(y)-apothem:int(y)]
        fov_array[fov_array < self.diff_threhold] = 0
        self.obj_locs = np.transpose(np.nonzero(fov_array))
        fov_center = fov_array[int(int(fov_array[0].size)/2)]
        # Zero out low diff values; get the indices of non-zero values.
        # Note: fov_array is a view of diff_array that gets its own set of indices starting at 0,0
        fov_array[fov_array < self.diff_threhold] = 0
        #print np.nonzero(fov_array)
        obj_locs = np.transpose(np.nonzero(fov_array))
        #print obj_locs, obj_locs.shape

        # Update self.obj_dists with distances of currently visible objects
        if obj_locs.size > 0:
            self.obj_dists = self.get_distance(obj_locs, fov_center)
        else:
            self.obj_dists = (np.empty(0), np.empty(0))


    def get_distance(self, locs, reference):
        """Get horizontal and vertical distances of objects in fov as a pair
        of NumPy arrays."""
        h_dists = (locs[:, 0] - reference[0])
        v_dists = (locs[:, 1] - reference[1])
        #print(h_dists[0])
        return (h_dists, v_dists)

    #def update_hit(self):



    def start(self):
        
        self.curr_img = self.update_fov()
        
        self.scanner.start(self.blink_time, False)
        def opencvimg( threadName, delay):
            background = None
            backdata = None
            #img_gif = Tkinter.BitmapImage('temp.bmp')
            #label_img = Tkinter.Label(root, image = img_gif)
            #label_img.pack()
            #root.mainloop()
            es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4))
            while 1:
                try:
                    #self.curr_fov.show()
                    #img_gray = cv.cvtColor(numpy.asarray(self.curr_fov),cv.COLOR_RGB2GRAY)
                    cvimg = cv.cvtColor(numpy.asarray(self.curr_fov),cv.COLOR_RGBA2GRAY)
                    #cvimg = cv2.cvtColor(imgbgr,cv.COLOR_GRAY2BGR)
                    gray_lwpCV = cv2.GaussianBlur(cvimg, (21, 21), 0)
                    #ret, frame =cvimg
                    #cvimg = numpy.asarray(self.get_diff())  
                    #im_at_mean = cv.adaptiveThreshold(im_gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 5, 7)
                    ret, im_thre = cv.threshold(cvimg, 127, 255, cv.THRESH_BINARY)
                    try:
                        cv.circle(cvimg, (self.center_x, self.center_y), 7, (0, 0, 255), 1)
                        #cv2.rectangle(cvimg, (self.center_x-25, self.center_y-25), (self.center_x+50, self.center_y+50), (0, 255, 0), 2)
                    except:
                        pass
                    #cv.resizeWindow("OpenCV", 480, 520);
                    #edges = cv.Canny(cvimg, 30, 90)
                    if background is None:
                        background = gray_lwpCV
                        continue
                    #cv.circle(cvimg, (self.center_x, self.center_y), 7, (0, 0, 255), 1)
                    diff = cv2.absdiff(background, gray_lwpCV)
                    diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1] # 二值化阈值处理
                    diff = cv2.dilate(diff, es, iterations=2) # 形态学膨胀

                    # 显示矩形框
                    image, contours, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 该函数计算一幅图像中目标的轮廓
                    for c in contours:
                        #if cv2.contourArea(c) < 1500: # 对于矩形区域，只显示大于给定阈值的轮廓，所以一些微小的变化不会显示。对于光照不变和噪声低的摄像头可不设定轮廓最小尺寸的阈值
                        #    continue
                        (x, y, w, h) = cv2.boundingRect(c) # 该函数计算矩形的边界框
                        cv2.rectangle(cvimg, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    background = gray_lwpCV
                    
                    cv2.imshow('dis', diff)

                    #cv.imshow("OpenCV_imgbgr",imgbgr)
                    cv.imshow("OpenCV_img",cvimg)

                    #cv.imshow("OpenCV_edge",edges)
                    cv.waitKey(1)
                    #cv.destroyAllWindows()
                except:
                    pass
        def get_xy(threadName, delay):
            command = '"F:/Python Project/TouhouPlayer-master/thwatch.exe"' #可以直接在命令行中执行的命令
            r = subprocess.Popen(command,shell=True) #执行该命令
            #re.match("([0-9]{5},[0-9]{5})",r)
            #info = r.readlines()  #读取命令行的输出到一个list
            #for line in iter(r.stdout.readline, b''):
            #    print line,
            pattern = re.compile(r'\(\d{5}\,\d{5}\)')

            while 1:
                MATCH = pattern.match(str(r.stdout)).groups()
                if MATCH!=None:
                    print(MATCH)
                else:
                    pass
                    #print ""
        try:
            thread.start_new_thread( opencvimg, ("Thread-1", 2, ) )
           # thread.start_new_thread( get_xy, ("Thread-2", 2, ) )
        except:
            print ("Error: unable to start thread")

    
def main():
    radar = Radar(195, 490)
    
    #reactor.callWhenRunning(cv.imshow("OpenCV",cvimg))
    reactor.callWhenRunning(radar.start)
    reactor.run()

    # start = time.time()
    # radar.start()
    # arr = radar.scan_fov()
    # # print(arr)
    # print(time.time() - start)

if __name__ == '__main__':
    main()
