import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets
import moviepy.editor
import functools
from tqdm import tqdm
import mido

parser = argparse.ArgumentParser(description='converts a synthesia-like video into a midi file')
parser.add_argument("filename")
parser.add_argument("-d", "--detected", action="store_true", help="display detected notes 1 by 1")

args = parser.parse_args()
video = moviepy.editor.VideoFileClip(args.filename)
audio = video.audio
numVideoFrames = int(video.end*video.fps)
numAudioSamples = int(audio.end*audio.fps)
print("fps: {}".format(video.fps))
print("audio rate: {}".format(audio.fps))

#get starting reference frame
startFrameIndex = 0

figure = plt.figure()
subplots = figure.subplots()
plt.subplots_adjust(bottom=0.12)
def changeStartFrameIndex(d, _):
    global startFrameIndex
    startFrameIndex = max(0, min(numVideoFrames, startFrameIndex+d))
    subplots.clear()
    subplots.set_title("select the first frame when keyboard is visible")
    subplots.imshow(video.get_frame(startFrameIndex/video.fps))
    textStartFrameIndex._text = str(startFrameIndex)
backBackBtn = matplotlib.widgets.Button(plt.axes((0.40,0.1,0.05,0.05)), "<<")
backBackBtn.on_clicked(functools.partial(changeStartFrameIndex, -video.fps))
backBtn = matplotlib.widgets.Button(plt.axes((0.45,0.1,0.05,0.05)), "<")
backBtn.on_clicked(functools.partial(changeStartFrameIndex, -1))
nextBtn = matplotlib.widgets.Button(plt.axes((0.50,0.1,0.05,0.05)), ">")
nextBtn.on_clicked(functools.partial(changeStartFrameIndex, 1))
nextNextBtn = matplotlib.widgets.Button(plt.axes((0.55,0.1,0.05,0.05)), ">>")
nextNextBtn.on_clicked(functools.partial(changeStartFrameIndex, int(round(video.fps))))
textStartFrameIndex = figure.text(0.5, 0.06, str(startFrameIndex), horizontalalignment="center")
figure.text(0.5, 0.02, "close window to confirm", horizontalalignment="center")
changeStartFrameIndex(0, None)
plt.show()

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
figure = plt.figure()
subplots = figure.subplots()
plt.subplots_adjust(bottom=0.30)
def redrawDetectionRegions(_):
    frame = video.get_frame(startFrameIndex/video.fps).copy()
    
    frame[sldrWhiteY.val] = (0, 0, 255)
    frame[sldrBlackY.val] = (0, 255, 0)
    i1 = int(sldrKeyRange.val[0])
    i2 = int(sldrKeyRange.val[1])
    j1 = WHITE_NOTES.index(i1)
    j2 = WHITE_NOTES.index(i2)
    for i in range(i1, i2+1):
        if(i in WHITE_NOTES):
            j = WHITE_NOTES.index(i)-j1
            x = (j+0.5)*video.size[0]/(j2-j1+1)
            y = sldrWhiteY.val
            width = sldrWhiteWidth.val
            height = sldrWhiteHeight.val
            col = (0, 0, 255)
        else:
            j = WHITE_NOTES.index(i+1)-j1
            x = j*video.size[0]/(j2-j1+1)
            y = sldrBlackY.val
            width = sldrBlackWidth.val
            height = sldrBlackHeight.val    
            col = (0, 255, 0)
        minX = max(0, int(x-width/2))
        maxX = int(x+width/2)
        minY = max(0, int(y-height/2))
        maxY = int(y+height/2)
        frame[minY:maxY, minX:maxX] = col
        
    subplots.clear()
    subplots.set_title("configure detection regions")
    subplots.imshow(frame)
sldrKeyRange = matplotlib.widgets.RangeSlider(plt.axes((0.15,0.20,0.7,0.03)), "note range", min(WHITE_NOTES), max(WHITE_NOTES), valinit=(21,108), valstep=WHITE_NOTES)
sldrKeyRange.on_changed(redrawDetectionRegions)
sldrWhiteY = matplotlib.widgets.Slider(plt.axes((0.15,0.15,0.25,0.03)), "white Y", 0, video.size[1]-1, valinit=whiteY, valstep=1)
sldrWhiteY.on_changed(redrawDetectionRegions)
sldrWhiteWidth = matplotlib.widgets.Slider(plt.axes((0.15,0.10,0.25,0.03)), "width", 0, 50, valinit=2, valstep=1)
sldrWhiteWidth.on_changed(redrawDetectionRegions)
sldrWhiteHeight = matplotlib.widgets.Slider(plt.axes((0.15,0.05,0.25,0.03)), "height", 0, 100, valinit=20, valstep=1)
sldrWhiteHeight.on_changed(redrawDetectionRegions)
sldrBlackY = matplotlib.widgets.Slider(plt.axes((0.6,0.15,0.25,0.03)), "black Y", 0, video.size[1]-1, valinit=blackY, valstep=1)
sldrBlackY.on_changed(redrawDetectionRegions)
sldrBlackWidth = matplotlib.widgets.Slider(plt.axes((0.6,0.10,0.25,0.03)), "width", 0, 50, valinit=2, valstep=1)
sldrBlackWidth.on_changed(redrawDetectionRegions)
sldrBlackHeight = matplotlib.widgets.Slider(plt.axes((0.6,0.05,0.25,0.03)), "height", 0, 100, valinit=30, valstep=1)
sldrBlackHeight.on_changed(redrawDetectionRegions)
redrawDetectionRegions(None)
plt.show()

#save detection regions
noteRegions = [None]*(max(WHITE_NOTES)+1)

i1 = int(sldrKeyRange.val[0])
i2 = int(sldrKeyRange.val[1])
j1 = WHITE_NOTES.index(i1)
j2 = WHITE_NOTES.index(i2)
frame = video.get_frame(startFrameIndex/video.fps).astype(int)
for i in range(i1, i2+1):
    if(i in WHITE_NOTES):
        j = WHITE_NOTES.index(i)-j1
        x = (j+0.5)*video.size[0]/(j2-j1+1)
        y = sldrWhiteY.val
        width = sldrWhiteWidth.val
        height = sldrWhiteHeight.val
    else:
        j = WHITE_NOTES.index(i+1)-j1
        x = j*video.size[0]/(j2-j1+1)
        y = sldrBlackY.val
        width = sldrBlackWidth.val
        height = sldrBlackHeight.val    
    minX = max(0, int(x-width/2))
    maxX = int(x+width/2)
    minY = max(0, int(y-height/2))
    maxY = int(y+height/2)
    noteRegions[i] = frame[minY:maxY, minX:maxX]

plt.show()

#configure detection threshold
diffs = np.zeros((numVideoFrames, max(WHITE_NOTES)+1))
for frameIndex in tqdm(range(startFrameIndex, numVideoFrames)):
    frame = video.get_frame(frameIndex/video.fps).astype(int)
    for i in range(i1, i2+1):
        if(i in WHITE_NOTES):
            j = WHITE_NOTES.index(i)-j1
            x = (j+0.5)*video.size[0]/(j2-j1+1)
            y = sldrWhiteY.val
            width = sldrWhiteWidth.val
            height = sldrWhiteHeight.val
        else:
            j = WHITE_NOTES.index(i+1)-j1
            x = j*video.size[0]/(j2-j1+1)
            y = sldrBlackY.val
            width = sldrBlackWidth.val
            height = sldrBlackHeight.val
        minX = max(0, int(x-width/2))
        maxX = int(x+width/2)
        minY = max(0, int(y-height/2))
        maxY = int(y+height/2)
        diff = np.sum(abs(noteRegions[i] - frame[minY:maxY, minX:maxX])) / np.prod(noteRegions[i].shape)
        diffs[frameIndex, i] = diff

figure = plt.figure()
subplots = figure.subplots()
plt.subplots_adjust(bottom=0.20)
subplots.hist(diffs.flatten(), bins=40)
plt.yscale("log")
subplots.set_title("select rgb distance threshold")
sldrThreshold = matplotlib.widgets.Slider(plt.axes((0.15,0.10,0.7,0.03)), "threshold", 0, int(np.max(diffs))+1, valinit=50, valstep=1)
plt.show()
threshold = sldrThreshold.val

#extract notes
notes = (diffs >= threshold).astype(int)
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
for frameIndex in range(startFrameIndex, numVideoFrames):
    seconds = frameIndex/video.fps
    ticks = int(seconds * ticksPerSecond)
    for i in range(128):
        if(noteOns[frameIndex, i]):
            track.append(mido.Message("note_on", note=i, velocity=100, time=ticks-prevMidiEventTime))
            prevMidiEventTime = ticks
        if(noteOffs[frameIndex, i]):
            track.append(mido.Message("note_off", note=i, time=ticks-prevMidiEventTime))
            prevMidiEventTime = ticks
mid.save(args.filename+".mid")

plt.imshow(np.swapaxes(notes, 0, 1))
plt.show()

#display detected notes
if(args.detected):
    for frameIndex in range(startFrameIndex, numVideoFrames):
        if(noteOns[frameIndex].any()):
            frame = video.get_frame(frameIndex/video.fps).copy()
            for i in range(i1, i2+1):
                if(notes[frameIndex, i]):
                    if(i in WHITE_NOTES):
                        j = WHITE_NOTES.index(i)-j1
                        x = (j+0.5)*video.size[0]/(j2-j1+1)
                        y = sldrWhiteY.val
                        width = sldrWhiteWidth.val
                        height = sldrWhiteHeight.val
                        col = (0, 0, 255)
                    else:
                        j = WHITE_NOTES.index(i+1)-j1
                        x = j*video.size[0]/(j2-j1+1)
                        y = sldrBlackY.val
                        width = sldrBlackWidth.val
                        height = sldrBlackHeight.val
                        col = (0, 255, 0)
                    minX = max(0, int(x-width/2))
                    maxX = int(x+width/2)
                    minY = max(0, int(y-height/2))
                    maxY = int(y+height/2)
                    diff = np.sum(noteRegions[i] - frame[minY:maxY, minX:maxX]) / np.prod(noteRegions[i].shape)
                    frame[minY:maxY, minX:maxX] = (255, 0, 0)
                    frame[int(maxY-height*abs(diff)/255):maxY, minX:maxX] = col
            plt.imshow(frame)
            plt.show()
