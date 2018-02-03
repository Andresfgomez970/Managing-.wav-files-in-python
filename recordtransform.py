import pyaudio
import wave
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pylab as plt
from scipy.io import wavfile
import cmath as cm
from scipy.fftpack import fft
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.io.wavfile import write
from scipy import signal



def recordaudio(chunk,formato,Channels,Rate,Record_seconds,
                            wave_output_name):
    '''Record and audio and get it as wave output.
        chunk: 
        formato: 
        Channels:  
        Rate: 
        Record_seconds:
        wave_output_name:
     '''

    p=pyaudio.PyAudio()

    stream=p.open(format=formato,
                  channels=Channels,
                  rate=Rate,
                  input=True,
                  frames_per_buffer=chunk)

    print("Recording..")

    frames=[]

    for i in range(0,int(Rate/chunk*Record_seconds)):
        data=stream.read(chunk)
        frames.append(data)

    print("Done recording.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(wave_output_name, 'wb')
    wf.setnchannels(Channels)
    wf.setsampwidth(p.get_sample_size(formato))
    wf.setframerate(Rate)
    wf.writeframes(b''.join(frames))
    wf.close()


def generatetones(secondpertone,steptone,littletone,maxtone,rate,name):
    t = np.linspace(0,secondpertone,rate*secondpertone) 

    lista = []
    for i in range(littletone,maxtone+steptone,steptone):
        data = np.sin(t*i*2*np.pi) 
        lista += list(data)  
    scaled = np.int16(lista/np.max(np.abs(lista)) * 32000)
    write(name, rate, scaled  )


def getsignal(wave_output_name,Record_seconds):
    fs, data = wavfile.read(wave_output_name)
    #Necessary parameters to the fourier transform
    try: 
        tamdata = data[:,0].size
    except: 
        tamdata = data.size
    
    dt = Record_seconds*1./tamdata
    t = np.arange(0,Record_seconds-dt/2,dt)
    try: 
        return t,data[:,0],dt
    except: 
        return t,data,dt


def Fourier1(time, data,dt):
    dataft = np.fft.fftshift(np.fft.fft(np.fft.fftshift(data)))*dt
    freq = np.arange(-1/(2*dt),1/(2*dt)-1/(2*dt*time.size),1/(dt*time.size))
    return freq,dataft

def dft(f,w,t,sign):
    if type(f)==type(w):
        F = f
    else:   
        F = f(t)
    DFT = []

    for j in w: 
        r2 = 0
        for i in range(len(t)):
            r2 += F[i]*np.exp(1j*j*t[i]*sign)
        DFT.append( (t[-1]-t[-2]) *r2)

    return t,np.array(DFT)

def plotfourier(freq,dataft,fi,ff,norm):
    if norm=='yes':
        plt.plot(freq,abs(dataft)/abs(dataft).sum(),'b',linewidth='5')
        plt.title('Normalized spectrum of frequencies',fontsize=25)
    else: 
        plt.plot(freq,abs(dataft),'b',linewidth='5')
        plt.title('Spectrum of frquencies',fontsize=25)
    plt.xlim(fi,ff)
    plt.ylabel('a.u.',fontsize=20)
    plt.xlabel('Frecuencia (Hz)',fontsize=20)
    plt.grid()

def recordtransform(chunk,formato,Channels,Rate,Record_seconds,wave_output_name,fi,ff,norm):
    recordaudio(chunk,formato,Channels,Rate,Record_seconds,wave_output_name)
    time, data,dt = getsignal(wave_output_name,Record_seconds)
    freq,dataft = Fourier1(time, data,dt)
    plotfourier(freq,dataft,fi,ff,norm)
    plt.show()
    
def plotonlytransform(chunk,formato,Channels,Rate,Record_seconds,wave_output_name,fi,ff,norm):
    time, data,dt = getsignal(wave_output_name,Record_seconds)
    freq,dataft = Fourier1(time, data,dt)
    plotfourier(freq,dataft,fi,ff,norm)
    plt.show()

def zoomplotonlytransform(chunk,formato,Channels,Rate,Record_seconds,wave_output_name,fi,ff,norm):
    time, data,dt = getsignal(wave_output_name,Record_seconds)
    freq,dataft = Fourier1(time, data,dt)
    plt.subplot(2,1,1)
    plt.plot(freq,abs(dataft)/abs(dataft).sum(),'b',linewidth='5')
    plt.title('Normalized spectrum of frequencies',fontsize=15)
    plt.xlim(fi,ff)
    plt.subplot(2,1,2)
    plt.plot(freq,abs(dataft)/abs(dataft).sum(),'b',linewidth='5')
    plt.title('Zoom to measured frequency',fontsize=15)
    con1 = abs(dataft)==abs(dataft).max()
    ft=abs(freq[con1])
    ft = ft[0]
    plt.xlim(ft-0.5,ft+0.5)
    plt.ylabel('a.u.',fontsize=20)
    plt.xlabel('Frecuencia (Hz)',fontsize=20)
    plt.grid()
    con1 = abs(dataft)==abs(dataft).max()
    print ('Frequency found at maximum value: %.2f  \n ' % (abs(freq[con1])) )
    plt.show()

def comparing(chunk,formato,Channels,Rate,Record_seconds,wavename1,
    wavename2,fi,ff,norm,tol):
    time, data,dt = getsignal(wavename1,Record_seconds)
    freq,dataft = Fourier1(time, data,dt)
    time2, data2,dt = getsignal(wavename2,Record_seconds)
    freq2,dataft2 = Fourier1(time2, data2,dt)
    plt.figure(figsize=(20,10))
    
    plt.subplot(2,2,1)
    plt.plot(freq,abs(dataft)/abs(dataft).sum(),'b',linewidth='5')
    plt.title('Normalized spectrum of frequencies',fontsize=15)
    plt.xlim(fi,ff)
    plt.ylabel('a.u.',fontsize=10)
    plt.xlabel('Frecuencia (Hz)',fontsize=10)
    plt.grid()

    plt.subplot(2,2,2)
    plt.plot(freq,abs(dataft)/abs(dataft).sum(),'b',linewidth='5')
    plt.title('Zoom to measured frequency',fontsize=15)
    con1 = abs(dataft)==abs(dataft).max()
    ft1= abs(freq[con1])
    plt.xlim(ft1-tol,ft1+tol)
    plt.ylabel('a.u.',fontsize=10)
    plt.xlabel('Frecuencia (Hz)',fontsize=10)
    plt.grid()
    
    plt.subplot(2,2,3)
    plt.plot(freq2,abs(dataft2)/abs(dataft2).sum(),'b',linewidth='5')
    plt.title('Normalized spectrum of frequencies',fontsize=15)
    plt.xlim(fi,ff)
    plt.ylabel('a.u.',fontsize=10)
    plt.xlabel('Frecuencia (Hz)',fontsize=10)
    plt.grid()

    plt.subplot(2,2,4)
    plt.plot(freq2,abs(dataft2)/abs(dataft2).sum(),'b',linewidth='5')
    plt.title('Normalized spectrum of frequencies',fontsize=15)
    con2 = abs(dataft2)==abs(dataft2).max()
    ft2=abs(freq2[con2])
    plt.xlim(ft2-tol,ft2+tol)
    plt.ylabel('a.u.',fontsize=10)
    plt.xlabel('Frecuencia (Hz)',fontsize=10)
    plt.grid()
    print  ('The difference was of %.2f Hz' %(abs(ft1-ft2)) )
    plt.show()

def f(wave_output_name,Record_seconds,time):
    t,data,dt = getsignal(wave_output_name,Record_seconds)
    datapersecond = len(data)/Record_seconds
    freqtimes = []
    dataft_times = []
    times = []
    for i in range(Record_seconds/time):
        datai = data[i*time*datapersecond:(i+1)*time*datapersecond] 
        timei  = t[i*time*datapersecond:(i+1)*time*datapersecond]
        dataft = np.fft.fftshift(np.fft.fft(np.fft.fftshift(datai)))*dt
        freq = np.arange(-1/(2*dt),1/(2*dt)-1/(2*dt*timei.size),1/(dt*timei.size))
        freqtimes.append(freq)
        dataft_times.append(dataft)
        times.append( (i+1)*time )

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    X = times
    Y = freqtimes

    for i in range(len(times)):
    #plt.plot(np.array([1,2]), np.array([1,2]), np.array([1,2]) ,'o')
       plt.plot( np.ones(len(freqtimes[i]))*times[i] , freqtimes[i]  , abs(dataft_times[i]))
    ax.set_xlabel('Time')
    ax.set_ylabel('Freq')
    ax.set_zlabel('A.U.')
    plt.show()
    for i in range(1000,20000,1000):
        plt.plot( i,freqtimes[i/1000].max() ,'ko')
    plt.show()
    for i in range(len(times)):
        plt.plot(freqtimes[i], abs(dataft_times[i] ) )
        plt.show()

def f(wave_output_name,Record_seconds,time,Rate):
    tm = 1./Rate
    a = time%tm
    if a>=tm/2.:
        time = time + (tm - time%tm)
    else: 
        time = time - time%tm
    t,data,dt = getsignal(wave_output_name,Record_seconds)
    datapersecond = len(data)/Record_seconds
    freqtimes = []
    dataft_times = []
    times = []
    for i in range( int(Record_seconds/time) ):
        s1 , s2 = int(i*time*datapersecond),int( (i+1)*time*datapersecond)
        datai = data[s1:s2] 
        timei  = t[s1:s2]
        dataft = np.fft.fftshift(np.fft.fft(np.fft.fftshift(datai)))*dt
        freq = np.arange(-1/(2*dt),1/(2*dt)-1/(2*dt*timei.size),1/(dt*timei.size))
        freqtimes.append(freq)
        dataft_times.append(dataft)
        times.append( (i+1)*time )

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    X = times
    Y = freqtimes
    
    for i in range(len(times)):
       plt.plot( np.ones(len(freqtimes[i]))*times[i] , freqtimes[i]  , abs(dataft_times[i]))
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Freq')
    ax.set_zlabel('A.U.')


if __name__ == "__main__":

    chunk=1024 #number of frames
    formato=pyaudio.paInt16 #format of the record 
    Channels=2 #Number of channels to record (this alter data)
    Rate=16040  #Number of frames per second
    Record_seconds=38 #lenghth of the recording
    wavename1="records/test1withegeneratednoise.wav" #output file name
    fi,ff=0,20000 
    norm = 'yes'
    wavename2 = "records/test1.wav" 

### Example 1 

    print("\nThe transform of the file 'test1withegeneratednoise.wav' is \n shown:\n") 

    plotonlytransform(chunk,formato,Channels,Rate,Record_seconds,wavename1,fi,ff,norm)

### Example 2

    print("\nThe transform of the file '3200.wav' is shown and also a \n zoom to the maximum value of the fourirer tranform:\n") 

    ### This part measure a given frequency that is already in a wave format in the program; in
    ### addition a zoom is made to it with some tolerance 

    Rate=44100
    Record_seconds=4.99
    wavename2 = "records/3200.wav" 
    fi, ff = 0, 10000 
    zoomplotonlytransform(chunk,formato,Channels,Rate,Record_seconds,wavename2,fi,ff,norm)


### Example 3

    ### This part record with the computer microphone and after that 
    ### show the fourier transform of the record 

    #You could change the paramters of the record that is going to be made
    Record_seconds=5
    wave_output_name = 'recorded.wav' 
    recordtransform(chunk,formato,Channels,Rate,Record_seconds,wave_output_name,fi,ff,norm)    
     
### Example 4

    ###This part plot the transform of the two wave files and permits  
    ### to compare the amplitues and the frequencies at the maximum 
    ### amplitude 

    Record_seconds= 3.0
    wavename1="records/1000.wav"
    wavename2="records/1000t.wav"
    ft = 3265
    tol = 3
    comparing(chunk,formato,Channels,Rate,Record_seconds,wavename1,
                       wavename2,fi,ff,norm,tol)



### Example 4

    ###This is basically the short fourier transform
    ### it is important to know that the algorithm 
    ### chose as step time the nearer on to the one that 
    ### you give that satisfy being a multiple of the 
    ### recorded seconds.

    wave_output_name = "records/1000.wav"
    Record_seconds = 3
    time = 0.1
    Rate = 46080

        
    f(wave_output_name,Record_seconds,time,Rate)
    plt.show()


### Example 5

    ###This algorithm compare the Fourier transform given by python 
    ### with one that I made, it is a way to test the the programed is 
    ### expected to work with some cases at least, a further analysis 
    ### could explain the differences (The graphs were scales for a value)
    ### chosen at hand.
 
    wavename = 'records/3265.wav'
    Record_seconds = 3
    t, data, dt = getsignal(wavename,Record_seconds)
    freq, dataft = Fourier1(t, data,dt)

    data = data[1000:1500]
    t = t[1000:1500]
    w = np.arange(-np.pi/dt,np.pi/dt,2*np.pi/(len(t)*dt)  )
    t, ft = dft(data,w,t,1)
    plt.plot(w/(2*np.pi),abs(ft.real)/abs(ft.real).sum()*(0.0169/0.0881) ,'b')
    plt.plot(freq,abs(dataft.real)/abs(dataft.real).sum() ,'g')
    plt.show()
