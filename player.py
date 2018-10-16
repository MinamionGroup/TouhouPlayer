# -*- coding: utf-8 -*- 
from radar import Radar
from twisted.internet import reactor
from twisted.internet.task import LoopingCall
import win32api, win32con, win32gui, win32ui
import ctypes
import random
import os
import time
import logging
import win32clipboard
import random

TH06_keyword='the Embodiment of Scarlet Devil'
touch_L=0
touch_R=0
touch_T=0
touch_D=0
SendInput = ctypes.windll.user32.SendInput
"""
logging.basicConfig(filename='thplayer.log',level=logging.DEBUG)

MOVE = {'left': 0xCB    ,   # 2 pixels each movement
        'up': 0xC8    ,
        'right': 0xCD    ,
        'down': 0xD0    }

MISC = {'shift': 0x2A,  # focus
        'esc': 0x01}

ATK = {'z': 0x2C,      # shoot
       'x': 0x2D} 

HIT_X = 192
HIT_Y = 385

def key_press(key):
    # TODO: Make this non-blocking
    win32api.keybd_event(key, 0, 0, 0)
    # reactor.callLater(.02, win32api.keybd_event,key, 0,
    #                   win32con.KEYEVENTF_KEYUP, 0)
    time.sleep(.02)
    win32api.keybd_event(key, 0, win32con.KEYEVENTF_KEYUP, 0)

def key_hold(self, key):
    win32api.keybd_event(key, 0, 0, 0)

def key_release(key):
    win32api.keybd_event(key, 0, win32con.KEYEVENTF_KEYUP, 0)

"""

SendInput = ctypes.windll.user32.SendInput
# C struct redefinitions
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

# Actuals Functions
def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_= Input_I()
    ii_.ki  = KeyBdInput( 0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

logging.basicConfig(filename='thplayer.log',level=logging.DEBUG)

MOVE = {'left': 0xCB,   # 2 pixels each movement
        'up': 0xC8,
        'right': 0xCD,
        'down': 0xD0}

MISC = {'shift': 0x10,  # focus
        'esc': 0x01}

ATK = {'z': 0x2C,      # shoot
       'x': 0x2D}      # bomb

HIT_X = 195
HIT_Y = 490

def key_press(key):
    # TODO: Make this non-blocking
        PressKey(key)
        time.sleep(.02)
        ReleaseKey(key)
        
    # reactor.callLater(.02, win32api.keybd_event,key, 0,
    #                   win32con.KEYEVENTF_KEYUP, 0)
            

def key_hold(self, key):
        PressKey(key)

def key_release(key):
    ReleaseKey(key)



class PlayerCharacter(object):
    
    def __init__(self, radar, hit_x=HIT_X, hit_y=HIT_Y, radius=3, moveleft= True):
        self.hit_x = hit_x
        self.hit_y = hit_y
        self.radius = radius
        self.width = 62
        self.height = 82    # slight overestimation
        self.radar = radar
        self.moveleft = moveleft
        bombcooldown = 0
    def move_left(self):
        # for i in range(4):
        # TODO: Hitbox should not be allowed to move outside of gameplay area
        key_press(MOVE['left'])
        self.hit_x -= 4

    def move_right(self):
        # for i in range(4):
        key_press(MOVE['right'])

        self.hit_x += 4

    def move_up(self):
        # for i in range(4):
        key_press(MOVE['up'])
        self.hit_y -= 8

    def move_down(self):
        # for i in range(4):
        key_press(MOVE['down'])
        self.hit_y += 8

    def shift(self, dir):     # Focused movement
        key_hold(MISC['shift'])
        key_press(MOVE[dir])
        key_release(MISC['shift'])

    def shoot(self):
        key_press(ATK['z'])

    def bomb(self):
        key_press(ATK['x'])

    def evade(self):
        self.hit_x = self.radar.center_x
        self.hit_y = self.radar.center_y
        h_dists, v_dists = self.radar.obj_dists
        #print(self.hit_x, self.hit_y)
        #print( h_dists.size,  v_dists.size)
        if h_dists.size > 0:
            #print('in coming!')
            #print('move to left')
            if self.hit_x < 100:
                    self.moveleft = False
            if self.hit_x >250:
                    self.moveleft = True
            if  self.moveleft:
                self.move_left()
            else:
                self.move_right()
            #if random.randint(0,20)>5:
            #    self.move_down();
            #else:
             #   self.move_up()
            
        else:
            if  self.moveleft:
                self.moveleft = False
            else:
                self.moveleft = True
        try:
            sum = 0
            for i in abs(h_dists) + abs(v_dists):
                if i < 100 and self.bombcooldown == 0:
                    sum += 1
            if sum > 50:
                self.bomb()
                self.bombcooldown = 100
            if self.bombcooldown > 0:
                self.bombcooldown -= 1
        except:
            pass


        print("h",h_dists)
        print("v",v_dists)
        #else:
            #self.move_to(192,385)
            #else:
            #    self.move_right()
            #else :
            #    if self.hit_x > 400:
            #        print('touch right')
            #        self.move_left()
        #logging.debug(h_dists, v_dists)
        #if self.hit_x < -20:
        #    touch_L = 1
         #   print('touch left')
        

    def move_to(self, x, y):
        move_x=self.hit_x -x
        move_y=self.hit_y - y
        print(move_x,move_y)
        if move_x <0:
            #while move_x>0:
            self.move_left()
        else:
            move_x=0-move_x
            #while move_x>0:
            self.move_right()
        if move_y >0:
            #while move_y>0:
            self.move_up()

        else:
            move_y=0-move_y
            #while move_x>0:
            self.move_down()
        """Bring character to (x, y)"""
        #pass

    def start(self):
        #self.shoot_constantly = LoopingCall(self.shoot)
        self.bomb_occasionally = LoopingCall(self.bomb)
        self.evader = LoopingCall(self.evade)
        #self.shoot_constantly.start(0)
        self.evader.start(.03)
        # self.bomb_occasionally.start(10, False)

def start_game():
    time.sleep(2)
    for i in range(6):
        key_press(0x2c)
        time.sleep(2)

def main():
    print ("#auto drive start#")
    start_game()
    radar = Radar(HIT_X, HIT_Y)
    player = PlayerCharacter(radar)
    reactor.callWhenRunning(player.start)
    reactor.callWhenRunning(radar.start)
    reactor.run()

if __name__ == "__main__":
    main()
