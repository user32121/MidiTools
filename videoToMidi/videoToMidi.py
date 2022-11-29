import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets
import moviepy.editor
import functools
from tqdm import tqdm
import mido

import utils

parser = argparse.ArgumentParser(description='converts a synthesia-like video into a midi file')
parser.add_argument("filename")
parser.add_argument("-d", "--detected", action="store_true", help="display detected notes")

args = parser.parse_args()
video = moviepy.editor.VideoFileClip(args.filename)
audio = video.audio
print("fps: {}".format(video.fps))
print("audio rate: {}".format(audio.fps))

#get starting reference frame and ending frame
figure = plt.figure()
subplots = figure.subplots(ncols=2)
plt.subplots_adjust(bottom=0.12)
def updateFrameRange(_):
    subplots[0].clear()
    subplots[0].set_title("select a starting reference frame when keyboard is visible")
    subplots[0].imshow(video.get_frame(sldrVidRange.val[0]))
    subplots[1].clear()
    subplots[1].set_title("select ending frame")
    subplots[1].imshow(video.get_frame(sldrVidRange.val[1]))
sldrVidRange = matplotlib.widgets.RangeSlider(plt.axes((0.15,0.10,0.7,0.03)), "", video.start, video.end-2/video.fps, valinit=(video.start, video.end-2/video.fps))
sldrVidRange.on_changed(updateFrameRange)
figure.text(0.5, 0.04, "close window to confirm", horizontalalignment="center")
updateFrameRange(None)
plt.show()
startFrameIndex = int(sldrVidRange.val[0]*video.fps)
endFrameIndex = int(sldrVidRange.val[1]*video.fps)

#configure detection regions
whiteY = int((video.size[1]-1)*0.96)
blackY = int((video.size[1]-1)*0.85)

WHITE_NOTES = [
    0,2,4,5,7,9,11,
    12,14,16,17,19,21,23,
    24,26,28,29,31,33,35,
    36,38,40,41,43,45,47,
    48,50,52,53,55,57,59,
    60,62,64,65,67,69,71,
    72,74,76,77,79,81,83,
    84,86,88,89,91,93,95,
    96,98,100,101,103,105,107,
    108,110,112,113,115,117,119,
    120,122,124,125,127,
]
BLACK_NOTES = np.setdiff1d(range(128), WHITE_NOTES)
figure = plt.figure()
subplots = figure.subplots()
plt.subplots_adjust(bottom=0.30)
def getDetectionRegions():
    regions = [None]*(max(WHITE_NOTES)+1)
    i1 = int(sldrKeyRange.val[0])
    i2 = int(sldrKeyRange.val[1])
    j1 = WHITE_NOTES.index(i1)
    j2 = WHITE_NOTES.index(i2)
    for i in range(i1, i2+1):
        if(i in WHITE_NOTES):
            j = WHITE_NOTES.index(i)-j1
            x = (j+0.5)*(video.size[0]+sldrLeftExtend.val+sldrRightExtend.val)/(j2-j1+1) - sldrLeftExtend.val
            y = sldrWhiteY.val
            width = sldrWhiteWidth.val
            height = sldrWhiteHeight.val
        else:
            j = WHITE_NOTES.index(i+1)-j1
            x = j*(video.size[0]+sldrLeftExtend.val+sldrRightExtend.val)/(j2-j1+1) - sldrLeftExtend.val
            y = sldrBlackY.val
            width = sldrBlackWidth.val
            height = sldrBlackHeight.val
            if(i%12 == 1):
                x -= sldr2SAdjustment.val
            elif(i%12 == 3):
                x += sldr2SAdjustment.val
            elif(i%12 == 6):
                x -= sldr3SAdjustment.val
            elif(i%12 == 10):
                x += sldr3SAdjustment.val
        minX = max(0, int(x-width/2))
        maxX = int(x+width/2)
        minY = max(0, int(y-height/2))
        maxY = int(y+height/2)
        regions[i] = [minX, maxX, minY, maxY]
    return regions
def redrawDetectionRegions(_):
    frame = video.get_frame(startFrameIndex/video.fps).copy()
    
    frame[sldrWhiteY.val] = (0, 0, 255)
    frame[sldrBlackY.val] = (0, 255, 0)
    regions = getDetectionRegions()
    for i in range(len(regions)):
        if(regions[i] == None):
            continue
        if(i in WHITE_NOTES):
            col = (0, 0, 255)
        else:
            col = (0, 255, 0)
        frame[regions[i][2]:regions[i][3], regions[i][0]:regions[i][1]] = col
        
    subplots.clear()
    subplots.set_title("configure detection regions")
    subplots.imshow(frame)
def enableAdvanced(_):
    btnAdvanced.ax.set_visible(False)
    plt.subplots_adjust(bottom=0.40)
    for sldr in sldrs:
        pos = sldr.ax.get_position()._points.flatten()
        pos[2:4] -= pos[0:2]
        pos[1] += 0.1
        sldr.ax.set_position(pos)
    for sldr in sldrsAdvanced:
        sldr.ax.set_visible(True)
sldrKeyRange = matplotlib.widgets.RangeSlider(plt.axes((0.15,0.22,0.7,0.03)), "note range", min(WHITE_NOTES), max(WHITE_NOTES), valinit=(21,108), valstep=WHITE_NOTES)
sldrWhiteY = matplotlib.widgets.Slider(plt.axes((0.15,0.17,0.25,0.03)), "white Y", 0, video.size[1]-1, valinit=whiteY, valstep=1)
sldrWhiteWidth = matplotlib.widgets.Slider(plt.axes((0.15,0.12,0.25,0.03)), "width", 0, 50, valinit=2, valstep=1)
sldrWhiteHeight = matplotlib.widgets.Slider(plt.axes((0.15,0.07,0.25,0.03)), "height", 0, 100, valinit=20, valstep=1)
sldrBlackY = matplotlib.widgets.Slider(plt.axes((0.6,0.17,0.25,0.03)), "black Y", 0, video.size[1]-1, valinit=blackY, valstep=1)
sldrBlackWidth = matplotlib.widgets.Slider(plt.axes((0.6,0.12,0.25,0.03)), "width", 0, 50, valinit=2, valstep=1)
sldrBlackHeight = matplotlib.widgets.Slider(plt.axes((0.6,0.07,0.25,0.03)), "height", 0, 100, valinit=30, valstep=1)
sldrs = [sldrKeyRange, sldrWhiteY, sldrWhiteWidth, sldrWhiteHeight, sldrBlackY, sldrBlackWidth, sldrBlackHeight]
for sldr in sldrs:
    sldr.on_changed(redrawDetectionRegions)
btnAdvanced = matplotlib.widgets.Button(plt.axes((0.43,0.02,0.14,0.04)), "advanced")
btnAdvanced.on_clicked(enableAdvanced)
sldrLeftExtend = matplotlib.widgets.Slider(plt.axes((0.15,0.12,0.25,0.03)), "left extend", -50, 50, valinit=0, valstep=1)
sldrRightExtend = matplotlib.widgets.Slider(plt.axes((0.6,0.12,0.25,0.03)), "right extend", -50, 50, valinit=0, valstep=1)
sldr2SAdjustment = matplotlib.widgets.Slider(plt.axes((0.15,0.07,0.25,0.03)), "2# adjust", -10, 10, valinit=0, valstep=1)
sldr3SAdjustment = matplotlib.widgets.Slider(plt.axes((0.6,0.07,0.25,0.03)), "3# adjust", -10, 10, valinit=0, valstep=1)
sldrsAdvanced = [sldrRightExtend, sldrLeftExtend, sldr2SAdjustment, sldr3SAdjustment]
for sldr in sldrsAdvanced:
    sldr.ax.set_visible(False)
    sldr.on_changed(redrawDetectionRegions)
redrawDetectionRegions(None)
plt.show()

#save detection regions
noteRegions = [None]*(max(WHITE_NOTES)+1)

regions = getDetectionRegions()
frame = video.get_frame(startFrameIndex/video.fps).astype(int)
for i in range(len(regions)):
    if(regions[i] == None):
        continue
    noteRegions[i] = frame[regions[i][2]:regions[i][3], regions[i][0]:regions[i][1]]

plt.show()

#get diffs
diffs = np.zeros((endFrameIndex-startFrameIndex+1, max(WHITE_NOTES)+1))
regions = getDetectionRegions()
for frameIndex in tqdm(range(startFrameIndex, endFrameIndex+1)):
    frame = video.get_frame(frameIndex/video.fps).astype(int)
    for i in range(len(regions)):
        if(regions[i] == None):
            continue
        diff = np.sum(abs(noteRegions[i] - frame[regions[i][2]:regions[i][3], regions[i][0]:regions[i][1]])) / np.prod(noteRegions[i].shape)
        diffs[frameIndex-startFrameIndex, i] = diff

#configure detection threshold
figure = plt.figure()
subplots = figure.subplots(ncols=2)
plt.subplots_adjust(bottom=0.20)
subplots[0].hist(diffs[WHITE_NOTES].flatten(), bins=40)
subplots[0].set_yscale("log")
subplots[0].set_title("white diffs")
subplots[1].hist(diffs[BLACK_NOTES].flatten(), bins=40)
subplots[1].set_yscale("log")
subplots[1].set_title("black diffs")
figure.text(0.5, 0.95, "select rgb distance threshold", horizontalalignment="center")
sldrWhiteThreshold = matplotlib.widgets.Slider(plt.axes((0.15,0.10,0.7,0.03)), "white threshold", 0, int(subplots[0].get_xlim()[1])+1, valinit=50, valstep=1)
sldrBlackThreshold = matplotlib.widgets.Slider(plt.axes((0.15,0.05,0.7,0.03)), "black threshold", 0, int(subplots[1].get_xlim()[1])+1, valinit=50, valstep=1)
plt.show()
whiteThreshold = sldrWhiteThreshold.val
blackThreshold = sldrBlackThreshold.val

#extract notes
notes = np.zeros((endFrameIndex-startFrameIndex+1, max(WHITE_NOTES)+1))
notes[:,WHITE_NOTES] = (diffs[:,WHITE_NOTES] >= whiteThreshold).astype(int)
notes[:,BLACK_NOTES] = (diffs[:,BLACK_NOTES] >= blackThreshold).astype(int)
noteOns = np.pad(notes[1:] - notes[:-1], ((1,0),(0,0))) > 0
noteOffs = np.pad(notes[1:] - notes[:-1], ((1,0),(0,0))) < 0

print("detected {} notes".format(np.sum(noteOns)))

mid = mido.MidiFile()
track = mido.MidiTrack()
mid.tracks.append(track)
prevMidiEventTime = 0
tempo = 500000  #default tempo, microseconds per beat
ticksPerBeat = mid.ticks_per_beat
ticksPerSecond = ticksPerBeat / (tempo / 1000000)
for i in range(endFrameIndex-startFrameIndex+1):
    seconds = i/video.fps
    ticks = int(seconds * ticksPerSecond)
    for j in range(128):
        if(noteOns[i, j]):
            track.append(mido.Message("note_on", note=j, velocity=100, time=ticks-prevMidiEventTime))
            prevMidiEventTime = ticks
        if(noteOffs[i, j]):
            track.append(mido.Message("note_off", note=j, time=ticks-prevMidiEventTime))
            prevMidiEventTime = ticks
mid.save(args.filename+".mid")

plt.imshow(np.swapaxes(notes, 0, 1))
plt.show()

#display detected notes
if(args.detected):
    figure = plt.figure()
    subplots = figure.subplots()
    plt.subplots_adjust(bottom=0.12)
    def updateDetectionFrame(_):
        frameIndex = sldrTime.val
        frame = video.get_frame(frameIndex/video.fps).copy()
        regions = getDetectionRegions()
        for i in range(len(regions)):
            if(regions[i] == None):
                continue
            if(notes[frameIndex-startFrameIndex, i]):
                if(i in WHITE_NOTES):
                    col = (0, 0, 255)
                else:
                    col = (0, 255, 0)
                frame[regions[i][2]:regions[i][3], regions[i][0]:regions[i][1]] = (255, 0, 0)
                height = regions[i][2]-regions[i][1]
                frame[int(regions[i][3]-height*abs(diff)/255):regions[i][3], regions[i][0]:regions[i][1]] = col
        subplots.clear()
        subplots.imshow(frame)
    def updateSldrTime(d, _):
        sldrTime.val += d
        updateDetectionFrame(None)
    sldrTime = matplotlib.widgets.Slider(plt.axes((0.15,0.10,0.7,0.03)), "", startFrameIndex, endFrameIndex, valinit=startFrameIndex, valstep=1)
    sldrTime.on_changed(updateDetectionFrame)
    btnPrevFrame = matplotlib.widgets.Button(plt.axes((0.45,0.05,0.05,0.03)), "<")
    btnPrevFrame.on_clicked(functools.partial(updateSldrTime, -1))
    btnNextFrame = matplotlib.widgets.Button(plt.axes((0.50,0.05,0.05,0.03)), ">")
    btnNextFrame.on_clicked(functools.partial(updateSldrTime, 1))
    updateDetectionFrame(None)
    plt.show()
