# Self Driving Car
# Importing the libraries
from ai import Dqn
from kivy.clock import Clock
from kivy.vector import Vector
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.cache import Cache
from kivy.core.window import Window
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time
import tkinter
from tkinter import filedialog
import os
# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line, Rectangle, InstructionGroup
from kivy.config import Config
# Make Window non-resizable to avoid app breakdown
Config.set('graphics', 'resizable', 0)
# Specify custom window size below. For default size, comment out below two lines
Config.set('graphics', 'width', '1200')
Config.set('graphics', 'height', '700')
Config.write()
# Importing the Dqn object from our AI in ai.py
# Clears map cache
Cache._categories['kv.image']['timeout'] = 1
Cache._categories['kv.texture']['timeout'] = 1
# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0
# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
brain = Dqn(5, 3, 0.9)
action2rotation = [0, 10, -10]
last_reward = 0
scores = []
# Initializing the map
first_update = True


def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    # changed variable type from float to bool to reduce NPY file size
    sand = np.zeros((longueur, largeur), dtype=np.bool)
    goal_x = 20
    goal_y = largeur - 20
    first_update = False


# Initializing the last distance
last_distance = 0
# Creating the car class


class Car(Widget):

    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)

    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate((self.angle+30) % 360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle-30) % 360) + self.pos
        self.signal1 = int(np.sum(sand[int(self.sensor1_x)-10:int(
            self.sensor1_x)+10, int(self.sensor1_y)-10:int(self.sensor1_y)+10]))/400.
        self.signal2 = int(np.sum(sand[int(self.sensor2_x)-10:int(
            self.sensor2_x)+10, int(self.sensor2_y)-10:int(self.sensor2_y)+10]))/400.
        self.signal3 = int(np.sum(sand[int(self.sensor3_x)-10:int(
            self.sensor3_x)+10, int(self.sensor3_y)-10:int(self.sensor3_y)+10]))/400.
        if self.sensor1_x > longueur-10 or self.sensor1_x < 10 or self.sensor1_y > largeur-10 or self.sensor1_y < 10:
            self.signal1 = 1.
        if self.sensor2_x > longueur-10 or self.sensor2_x < 10 or self.sensor2_y > largeur-10 or self.sensor2_y < 10:
            self.signal2 = 1.
        if self.sensor3_x > longueur-10 or self.sensor3_x < 10 or self.sensor3_y > largeur-10 or self.sensor3_y < 10:
            self.signal3 = 1.


class Ball1(Widget):
    pass


class Ball2(Widget):
    pass


class Ball3(Widget):
    pass
# Creating the game class


class Game(Widget):

    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)

    def update(self, dt):
        global brain
        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur
        longueur = self.width
        largeur = self.height
        if first_update:
            init()
        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx, yy))/180.
        last_signal = [self.car.signal1, self.car.signal2,
                       self.car.signal3, orientation, -orientation]
        action = brain.update(last_reward, last_signal)
        scores.append(brain.score())
        rotation = action2rotation[action]
        self.car.move(rotation)
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3
        if sand[int(self.car.x), int(self.car.y)] > 0:
            self.car.velocity = Vector(1, 0).rotate(self.car.angle)
            last_reward = -1
        else:  # otherwise
            self.car.velocity = Vector(6, 0).rotate(self.car.angle)
            last_reward = -0.105
            if distance < last_distance:
                last_reward = 0.1
        if self.car.x < 10:
            self.car.x = 10
            last_reward = -1
        if self.car.x > self.width - 10:
            self.car.x = self.width - 10
            last_reward = -1
        if self.car.y < 10:
            self.car.y = 10
            last_reward = -1
        if self.car.y > self.height - 10:
            self.car.y = self.height - 10
            last_reward = -1
        if distance < 100:
            goal_x = self.width-goal_x
            goal_y = self.height-goal_y
            last_reward = 2
        last_distance = distance
# Adding the painting tools


class MyPaintWidget(Widget):
    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8, 0.7, 0)
            d = 10.
            touch.ud['line'] = Line(points=(touch.x, touch.y), width=10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x), int(touch.y)] = 1

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10: int(touch.x) + 10,
                 int(touch.y) - 10: int(touch.y) + 10] = 1
            last_x = x
            last_y = y
# Adding the API Buttons (clear, save and load)


class CarApp(App):
    def build(self):
        global parent
        global clearbtn
        global savebtn
        global loadbtn
        global mapsavebtn
        global maploadbtn
        global brainbtn

        self.fig = None
        self.plotData = None

        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        Clock.schedule_interval(self.updateBrainPlot, 1.0)
        parent.painter = MyPaintWidget(size=Window.size)
        clearbtn = Button(text='Clear Map', opacity=0.7)
        maploadbtn = Button(text='Load Map', pos=(
            parent.width, 0), opacity=0.7)
        mapsavebtn = Button(text='Save Map', pos=(
            2 * parent.width, 0), opacity=0.7)
        loadbtn = Button(text='Load Brain', pos=(
            3 * parent.width, 0), opacity=0.7)
        savebtn = Button(text='Save Brain', pos=(
            4 * parent.width, 0), opacity=0.7)
        brainbtn = Button(text='Brain Graph', pos=(
            5 * parent.width, 0), opacity=0.7)
        clearbtn.bind(on_release=self.clear_canvas)
        savebtn.bind(on_release=self.save)
        loadbtn.bind(on_release=self.load)
        mapsavebtn.bind(on_release=self.mapsave)
        maploadbtn.bind(on_release=self.mapload)
        brainbtn.bind(on_release=self.braingraph)
        parent.add_widget(parent.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        parent.add_widget(mapsavebtn)
        parent.add_widget(maploadbtn)
        parent.add_widget(brainbtn)
        return parent

    def clear_canvas(self, obj):
        global sand
        parent.canvas.before.clear()
        parent.painter.canvas.clear()
        parent.painter.canvas.before.clear()
        sand = np.zeros((longueur, largeur))

    def save(self, obj):
        global pth_file_path
        pth_file_path = ""
        root = tkinter.Tk()
        root.withdraw()
        # save brain
        pth_file_path = filedialog.asksaveasfilename(
            title='Save PTH file', filetypes=(("PTH files", "*.pth"), ("all files", "*.*")))
        if pth_file_path == ():
            pth_file_path = ""
        if pth_file_path != "":
            brain.save(pth_file_path)
            plt.plot(scores)
            plt.show()
            print("Brain saved: "+pth_file_path)
        else:
            print("Specify PTH file name")

    def load(self, obj):
        global pth_file_path
        pth_file_path = ""
        root = tkinter.Tk()
        root.withdraw()
        # load brain
        pth_file_path = filedialog.askopenfilename(
            title='Open PTH file', filetypes=(("PTH files", "*.pth"), ("all files", "*.*")))
        if pth_file_path == ():
            pth_file_path = ""
        if pth_file_path != "":
            brain.load(pth_file_path)
            plt.plot(scores)
            plt.show()
            print("Brain loaded: "+pth_file_path)
        else:
            print("Specify PTH file name")

    def mapsave(self, obj):
        global sand
        global png_file_path
        global npy_file_path
        png_file_path = ""
        npy_file_path = ""
        root = tkinter.Tk()
        root.withdraw()

        # save png file
        png_file_path = filedialog.asksaveasfilename(
            title='Save PNG file', filetypes=(("PNG files", "*.png"), ("all files", "*.*")))
        if png_file_path == ():
            png_file_path = ""
        if png_file_path != "":
            parent.car.opacity = 0
            parent.ball1.opacity = 0
            parent.ball2.opacity = 0
            parent.ball3.opacity = 0
            parent.remove_widget(clearbtn)
            parent.remove_widget(savebtn)
            parent.remove_widget(loadbtn)
            parent.remove_widget(mapsavebtn)
            parent.remove_widget(maploadbtn)
            parent.remove_widget(brainbtn)
            parent.export_to_png(png_file_path)
            parent.car.opacity = 1
            parent.ball1.opacity = 1
            parent.ball2.opacity = 1
            parent.ball3.opacity = 1
            parent.add_widget(clearbtn)
            parent.add_widget(savebtn)
            parent.add_widget(loadbtn)
            parent.add_widget(mapsavebtn)
            parent.add_widget(maploadbtn)
            parent.add_widget(brainbtn)
            print("PNG file saved: " + png_file_path)
        else:
            print("PNG Filename not specified")

        # save npy file
        npy_file_path = filedialog.asksaveasfilename(
            title='Save NPY file', filetypes=(("NPY files", "*.npy"), ("all files", "*.*")))
        if npy_file_path == ():
            npy_file_path = ""
        if npy_file_path != "":
            np.save(npy_file_path, sand)
            print("NPY file saved: " + npy_file_path)
        else:
            print("NPY Filename not specified")

    def mapload(self, obj):
        global sand
        global img
        global png_file_path
        global npy_file_path
        png_file_path = ""
        npy_file_path = ""
        root = tkinter.Tk()
        root.withdraw()

        # open png file
        png_file_path = filedialog.askopenfilename(
            title='Open PNG file', filetypes=(("PNG files", "*.png"), ("all files", "*.*")))
        if png_file_path == ():
            png_file_path = ""
        if os.path.isfile(png_file_path):
            parent.canvas.before.clear()
            parent.painter.canvas.clear()
            parent.painter.canvas.before.clear()
            with parent.canvas.before:
                Rectangle(source=png_file_path, size=Window.size)
            print("PNG file loaded: " + png_file_path)
        else:
            print("PNG File not found: " + png_file_path)

        # open npy file
        npy_file_path = filedialog.askopenfilename(
            title='Open NPY file', filetypes=(("NPY files", "*.npy"), ("all files", "*.*")))
        if npy_file_path == ():
            npy_file_path = ""
        if os.path.isfile(npy_file_path):
            sand = np.load(npy_file_path)
            print("NPY file loaded: " + npy_file_path)
        else:
            print("NPY File not found: " + npy_file_path)
        # Cache.print_usage()

    def braingraph(self, obj):
        plt.plot(scores)
        plt.ion()
        plt.show()

    def updateBrainPlot(self, dt):
        plotSize = 10000
        print("update brain plot")
        if self.fig is None:
            plt.ion()
            self.fig = plt.figure()
            ax = self.fig.add_subplot(111)
            plt.ylim(-0.2, 0.2)
            x = np.linspace(0, plotSize, plotSize)
            y = np.asarray(scores)[-plotSize:]
            y2 = y.copy()
            y2.resize(plotSize, refcheck=False)
            self.plotData, = ax.plot(x, y2, label="Brain score")
            plt.show()
        else:
            y = np.asarray(scores)[-plotSize:]
            y2 = y.copy()
            y2.resize(plotSize, refcheck=False)
            self.plotData.set_ydata(y2)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()


# Running the whole thing
if __name__ == '__main__':
    CarApp().run()
