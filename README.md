# Managing-.wav-files-in-python
In the following repo I plan to add code to manage .wav files in python. 

## Getting Started 

I describe breifly the important (in my consideration) functions of each one the files.

### recordtransform.py

#### variables 

--- inputs --- <br />
chunk: represents the number of parts in which is divided the file. <br />
formato: represnets the format of the audio. <br />
channels: Number of channels to record (Inn one of them you could record the noise for example). <br />
Rate: Number of data that it takes per second per second. (so, t_s = 1/Rate) <br />
wavenamei: wavefilei for which you want to apply the function. <br />
fi,ff: In terval of frequencies in which you wan to see your graph. <br />
norm: choose if you wnat a normalize spectrum of frequencies (do norm='yes' if that is the case). <br />
secondpertone: seconds that you want a given tone to long.  <br />
littletone: minimum tone that you want. <br />
maxtone: maximum value that a tone could take. <br />
steptone: difference that you want between two tones. <br />

--- outputs --- <br />
dt: separation between two consecutive times in the array time. <br /> 
time: array of all the time of the record. <br />
Data: Values of the signal in the respective time. <br />
Dataft: Values of the amplitues for a given frequency. <br />
freq: Correspondent array of frequencies for Dataft. <br />


### plotonlytransform(chunk,formato,Channels,Rate,Record_seconds,wavename1,fi,ff,norm)

This funtions plot the transform for the wavename1 file between fi and ff. The normalization depends of you. <br />

### recordaudio(chunk,formato,Channels,Rate,Record_seconds,wave_output_name)

This is to record directly from the computer.

### generatetones(secondpertone,steptone,littletone,maxtone,Rate,wave_output_name)

Tis is to generate a sequence of tones with a given step. 

### getsignal(wave_output_name,Record_seconds)

Get data, t and dt from the wav file. 

### Fourier1(time,data,dt)

Get freq and Dataft.

### dft(f,w,t,sign)

Apply the fourier transform with a simple implementation (It is not the FFT).

### plotfourier(freq,dataft,fi,ff,norm)

Plot freq vs Dataft in the interval fi and ff. 

### recordtransform(chunk,formato,Channels,Rate,Record_seconds,wave_output_name,fi,ff,norm)

Record somehing and then apply and show the transform. 

### plotonlytransform(chunk,formato,Channels,Rate,Record_seconds,wave_output_name,fi,ff,norm)

Plot the fourier transform for a given wav file. 

### zoomplotonlytransform(chunk,formato,Channels,Rate,Record_seconds,wave_output_name,fi,ff,norm)

Plot the fourier transform of a file and a zoom for the maximum frequency found. note: (Change only to search in positive frequecies).

### comparing(chunk,formato,Channels,Rate,Record_seconds,wavename1,wavename2,fi,ff,norm,tol):

Plot the fourier transform of a file and a zoom for the maximum frequencies found for both files in a same figure. note: (Change only to search in positive frequecies).

### f(wave_output_name,Record_seconds,time,Rate)

Apply the short fourier transform for a given file in intervals separated approximately time (the nearer value multiple to Record_seconds is the one taken) seconds.



