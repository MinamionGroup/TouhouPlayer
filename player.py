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

MOVE = {'left': 0x25,   # 2 pixels each movement
        'up': 0x26,
        'right': 0x27,
        'down': 0x28}

MISC = {'shift': 0x10,  # focus
        'esc': 0x1B}

ATK = {'z': 0x2C,      # shoot
       'x': 0x58}      # bomb

HIT_X = 192
HIT_Y = 385

def key_press(key):
    # TODO: Make this non-blocking
    PressKey(key)
    # reactor.callLater(.02, win32api.keybd_event,key, 0,
    #                   win32con.KEYEVENTF_KEYUP, 0)
    time.sleep(.02)
    ReleaseKey(key)

def key_hold(self, key):
    PressKey(key)

def key_release(key):
    ReleaseKey(key)



class PlayerCharacter(object):
    def __init__(self, radar, hit_x=HIT_X, hit_y=HIT_Y, radius=3):
        self.hit_x = hit_x
        self.hit_y = hit_y
        self.radius = radius
        self.width = 62
        self.height = 82    # slight overestimation
        self.radar = radar

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
        key.key_release(MISC['shift'])

    def shoot(self):
        key_press(ATK['z'])

    def bomb(self):
        key_press(ATK['x'])

    def evade(self):
        h_dists, v_dists = self.radar.obj_dists
        if h_dists.size > 0:
            self.move_left()
        #logging.debug(h_dists, v_dists)

        print(self.hit_x, self.hit_y)

    def move_to(self, x, y):
        """Bring character to (x, y)"""
        pass

    def start(self):
        self.shoot_constantly = LoopingCall(self.shoot)
        self.bomb_occasionally = LoopingCall(self.bomb)
        self.evader = LoopingCall(self.evade)

        self.shoot_constantly.start(0)
        self.evader.start(.03)
        # self.bomb_occasionally.start(10, False)

def start_game():
    time.sleep(2)
    for i in range(5):
        key_press(0x5A)
        time.sleep(1.5)

def main():
    start_game()
    radar = Radar((HIT_X, HIT_Y))
    player = PlayerCharacter(radar)

    reactor.callWhenRunning(player.start)
    reactor.callWhenRunning(radar.start)
    reactor.run()

if __name__ == "__main__":
    main()
