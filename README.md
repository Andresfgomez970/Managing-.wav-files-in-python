# Managing-.wav-files-in-python
In the following repo I plan to add code to manage .wav files in python. 

## Getting Started (Take care: it is not even a good sketch at the moment)

I describe breifly the important (in my consideration) functions of each one the files.

### recordtransform.py

#### variables 


chunk :: represents the number of parts in which is divided the file 
formato :: represnets the format of the audio 
channels :: Number of channels to record (Inn one of them you could record the noise for example)
Rate :: Number of data that it takes per second per second (so, t_s = 1/Rate)
wavename1 :: wavefile for which you want to apply the function 
fi,ff :: In terval of frequencies in which you wan to see your graph 
norm :: choose if you wnat a normalize spectrum of frequencies (do norm='yes' if that is the case)
secondpertone
steptone
littletone
maxtone
rate


### plotonlytransform(chunk,formato,Channels,Rate,Record_seconds,wavename1,fi,ff,norm)

This funtions plot the transform for the wavename1 file between fi and ff. The normalization depend of you. 

### recordaudio(chunk,formato,Channels,Rate,Record_seconds,wave_output_name)

This is to record directly from the computer.

### generatetones(secondpertone,steptone,littletone,maxtone,rate,name)

Tis is to generate a sequence of tones with a given step. 
