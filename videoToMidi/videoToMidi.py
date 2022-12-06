import argparse
import functools
import colorsys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets
import moviepy.editor
from tqdm import tqdm
import mido
import pyaudio

import utils

parser = argparse.ArgumentParser(description='converts a synthesia-like video into a midi file')
parser.add_argument("filename")
parser.add_argument("-d", "--detected", action="store_true", help="display detected notes (ignores -s)")
parser.add_argument("-s", "--skip-GUI", action="store_true", help="skip any matplotlib GUI and use defaults")

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
if(not args.skip_GUI):
    plt.show()
startFrameIndex = int(sldrVidRange.val[0]*video.fps)
endFrameIndex = int(sldrVidRange.val[1]*video.fps)
print("frames: {}".format(endFrameIndex-startFrameIndex+1))

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
sldrWhiteWidth = matplotlib.widgets.Slider(plt.axes((0.15,0.12,0.25,0.03)), "width", 1, 50, valinit=2, valstep=1)
sldrWhiteHeight = matplotlib.widgets.Slider(plt.axes((0.15,0.07,0.25,0.03)), "height", 1, 100, valinit=20, valstep=1)
sldrBlackY = matplotlib.widgets.Slider(plt.axes((0.6,0.17,0.25,0.03)), "black Y", 0, video.size[1]-1, valinit=blackY, valstep=1)
sldrBlackWidth = matplotlib.widgets.Slider(plt.axes((0.6,0.12,0.25,0.03)), "width", 1, 50, valinit=2, valstep=1)
sldrBlackHeight = matplotlib.widgets.Slider(plt.axes((0.6,0.07,0.25,0.03)), "height", 1, 100, valinit=30, valstep=1)
sldrs = [sldrKeyRange, sldrWhiteY, sldrWhiteWidth, sldrWhiteHeight, sldrBlackY, sldrBlackWidth, sldrBlackHeight]
for sldr in sldrs:
    sldr.on_changed(redrawDetectionRegions)
btnAdvanced = matplotlib.widgets.Button(plt.axes((0.43,0.02,0.14,0.04)), "advanced")
btnAdvanced.on_clicked(enableAdvanced)
sldrLeftExtend = matplotlib.widgets.Slider(plt.axes((0.15,0.12,0.25,0.03)), "left extend", -20, 20, valinit=0, valstep=1)
sldrRightExtend = matplotlib.widgets.Slider(plt.axes((0.6,0.12,0.25,0.03)), "right extend", -20, 20, valinit=0, valstep=1)
sldr2SAdjustment = matplotlib.widgets.Slider(plt.axes((0.15,0.07,0.25,0.03)), "2# adjust", -10, 10, valinit=1, valstep=1)
sldr3SAdjustment = matplotlib.widgets.Slider(plt.axes((0.6,0.07,0.25,0.03)), "3# adjust", -10, 10, valinit=1, valstep=1)
sldrsAdvanced = [sldrRightExtend, sldrLeftExtend, sldr2SAdjustment, sldr3SAdjustment]
for sldr in sldrsAdvanced:
    sldr.ax.set_visible(False)
    sldr.on_changed(redrawDetectionRegions)
redrawDetectionRegions(None)
if(not args.skip_GUI):
    plt.show()

#save detection regions
noteRegions = [None]*(max(WHITE_NOTES)+1)

regions = getDetectionRegions()
frame = video.get_frame(startFrameIndex/video.fps).astype(int)
for i in range(len(regions)):
    if(regions[i] == None):
        continue
    noteRegions[i] = frame[regions[i][2]:regions[i][3], regions[i][0]:regions[i][1]]

#get diffs
diffs = np.zeros((endFrameIndex-startFrameIndex+1, max(WHITE_NOTES)+1))
hues = np.zeros_like(diffs)
regions = getDetectionRegions()
for frameIndex in tqdm(range(startFrameIndex, endFrameIndex+1)):
    frame = video.get_frame(frameIndex/video.fps).astype(int)
    for i in range(len(regions)):
        if(regions[i] == None):
            continue
        diff = np.sum(abs(noteRegions[i] - frame[regions[i][2]:regions[i][3], regions[i][0]:regions[i][1]])) / np.size(noteRegions[i])
        diffs[frameIndex-startFrameIndex, i] = diff
        hues[frameIndex-startFrameIndex, i] = colorsys.rgb_to_hsv(*(np.mean(frame[regions[i][2]:regions[i][3], regions[i][0]:regions[i][1]], axis=(0,1))/255))[0]

#configure detection threshold
figure = plt.figure()
subplots = figure.subplots(ncols=2)
plt.subplots_adjust(bottom=0.20)
subplots[0].hist(diffs[:,WHITE_NOTES].flatten(), bins=40)
subplots[0].set_yscale("log")
subplots[0].set_title("white diffs")
subplots[1].hist(diffs[:,BLACK_NOTES].flatten(), bins=40)
subplots[1].set_yscale("log")
subplots[1].set_title("black diffs")
figure.text(0.5, 0.95, "select rgb distance threshold", horizontalalignment="center")
whiteDiffMax = int(np.max(diffs[:,WHITE_NOTES]))
sldrWhiteThreshold = matplotlib.widgets.Slider(plt.axes((0.15,0.10,0.7,0.03)), "white threshold", 0, whiteDiffMax+1, valinit=whiteDiffMax//2, valstep=1)
blackDiffMax = int(np.max(diffs[:,BLACK_NOTES]))
sldrBlackThreshold = matplotlib.widgets.Slider(plt.axes((0.15,0.05,0.7,0.03)), "black threshold", 0, blackDiffMax+1, valinit=blackDiffMax//2, valstep=1)
if(not args.skip_GUI):
    plt.show()
whiteThreshold = sldrWhiteThreshold.val
blackThreshold = sldrBlackThreshold.val

#extract notes
notes = np.zeros_like(diffs, dtype=int)
notes[:,WHITE_NOTES] = (diffs[:,WHITE_NOTES] >= whiteThreshold).astype(int)
notes[:,BLACK_NOTES] = (diffs[:,BLACK_NOTES] >= blackThreshold).astype(int)
#auto cluster notes
notes, numTracks = utils.clusterByHue(notes, hues)
noteOns = np.zeros_like(notes)
noteOns[1:] = np.logical_and(notes[:-1] != notes[1:], notes[1:] != 0)
noteOns[0] = notes[0]
noteOffs = np.zeros_like(notes)
noteOffs[1:] = np.logical_and(notes[:-1] != notes[1:], notes[:-1] != 0)
noteOffs[-1] = notes[-1]

print("detected {} notes".format(np.sum(noteOns > 0)))

#get audio
startAudioIndex = int(sldrVidRange.val[0]*audio.fps)
endAudioIndex = int(sldrVidRange.val[1]*audio.fps)
print("audio samples: {}".format(endAudioIndex-startAudioIndex+1))
audioSamples = np.zeros((endAudioIndex-startAudioIndex+1, audio.nchannels))
for i in tqdm(range(startAudioIndex, endAudioIndex+1)):
    audioSamples[i-startAudioIndex] = audio.get_frame(i/audio.fps)

#get first note
noteAudioIndex = -1
for i in range(startFrameIndex, endFrameIndex+1):
    for j in range(128):
        if(noteOns[i-startFrameIndex, j]):
            noteAudioIndex = int(i/video.fps*audio.fps - startAudioIndex)
            break
    else:
        continue
    break

audioStream = pyaudio.PyAudio().open(
    format = pyaudio.paFloat32,
    channels = audio.nchannels,
    rate = audio.fps,
    output = True)

#get detection offset
figure = plt.figure()
subplots = figure.subplots()
plt.subplots_adjust(bottom=0.30)
def displayFFTWindow(_):
    audioSample = audioSamples[int(noteAudioIndex+sldrFFTWindow.val[0]*audio.fps):int(noteAudioIndex+sldrFFTWindow.val[1]*audio.fps+1)]
    subplots.clear()
    subplots.set_title("configure fft window")
    subplots.plot(audioSample)
def playAudio(_):
    audioSample = audioSamples[int(noteAudioIndex+sldrFFTWindow.val[0]*audio.fps):int(noteAudioIndex+sldrFFTWindow.val[1]*audio.fps+1)]
    audioStream.write(audioSample.astype(np.float32).tobytes())
sldrFFTWindow = matplotlib.widgets.RangeSlider(plt.axes((0.15,0.10,0.7,0.03)), "", -1, 1, valinit=(0, 0.3))
sldrFFTWindow.on_changed(displayFFTWindow)
btnPlay = matplotlib.widgets.Button(plt.axes((0.43,0.05,0.14,0.04)), "play")
btnPlay.on_clicked(playAudio)
displayFFTWindow(None)
if(not args.skip_GUI):
    plt.show()

#TODO extract velocity information from audio
fftWindowOffsetStart = sldrFFTWindow.val[0]*audio.fps
fftWindowOffsetEnd = sldrFFTWindow.val[1]*audio.fps
print("fft window offset: {} samples ({} frames)".format(fftWindowOffsetStart, fftWindowOffsetStart / audio.fps * video.fps))
print("fft window size: {} samples ({} frames)".format(fftWindowOffsetEnd-fftWindowOffsetStart, (fftWindowOffsetEnd-fftWindowOffsetStart) / audio.fps * video.fps))
volumes = np.zeros_like(noteOns, dtype=float)
for i in tqdm(range(startFrameIndex, endFrameIndex+1)):
    for j in range(128):
        if(noteOns[i-startFrameIndex, j]):
            fftWindowStart = max(int(i / video.fps * audio.fps + fftWindowOffsetStart), startAudioIndex)
            fftWindowEnd = min(int(i / video.fps * audio.fps) + fftWindowOffsetEnd, endAudioIndex)
            fft = np.fft.fft(audioSamples[int(fftWindowStart-startAudioIndex):int(fftWindowEnd-startAudioIndex)], axis=0) #, norm="ortho")
            #https://dsp.stackexchange.com/a/72077
            N = len(fft)
            T = 1/audio.fps
            freqBins = np.fft.fftfreq(len(fft), d=T)
            targetFreq = 440 * 2**((j-69)/12)
            volume = 0
            curFreq = targetFreq
            while True:
                index, = np.nonzero(np.isclose(freqBins, curFreq, atol=1/(T*N)))
                if(index.size == 0):
                    break
                volume += np.mean(np.abs(fft[index[0]]))
                curFreq += targetFreq
            volumes[i-startFrameIndex, j] = volume
#normalize
volumes = np.maximum(1, (volumes / np.max(volumes) * 127).astype(int))

#configure volume multiplier

mid = mido.MidiFile()
for _ in range(numTracks+1):
    mid.tracks.append(mido.MidiTrack())
prevMidiEventTime = [0]*(numTracks+1)
tempo = 500000  #default tempo, microseconds per beat
ticksPerBeat = mid.ticks_per_beat
ticksPerSecond = ticksPerBeat / (tempo / 1000000)
for i in range(endFrameIndex-startFrameIndex+1):
    seconds = i/video.fps
    ticks = int(seconds * ticksPerSecond)
    for j in range(128):
        if(noteOns[i, j]):
            track = notes[i,j]
            mid.tracks[track].append(mido.Message("note_on", note=j, velocity=volumes[i,j], time=ticks-prevMidiEventTime[track]))
            prevMidiEventTime[track] = ticks
        if(noteOffs[i, j]):
            track = notes[i-1,j]
            mid.tracks[track].append(mido.Message("note_off", note=j, time=ticks-prevMidiEventTime[track]))
            prevMidiEventTime[track] = ticks
mid.save(args.filename+".mid")

if(not args.skip_GUI):
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
