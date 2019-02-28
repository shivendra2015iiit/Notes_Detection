
## Mocking Bot - Task 1.1: Note Detection

#  Instructions
#  ------------
#
#  This file contains Main function and note_detect function. Main Function helps you to check your output
#  for practice audio files provided. Do not make any changes in the Main Function.
#  You have to complete only the note_detect function. You can add helper functions but make sure
#  that these functions are called from note_detect function. The final output should be returned
#  from the note_detect function.
#
#  Note: While evaluation we will use only the note_detect function. Hence the format of input, output
#  or returned arguments should be as per the given format.
#
#  Recommended Python version is 2.7.
#  The submitted Python file must be 2.7 compatible as the evaluation will be done on Python 2.7.
#
#  Warning: The error due to compatibility will not be entertained.
#  -------------


## Library initialisation

# Import Modules
# DO NOT import any library/module
# related to Audio Processing here
import numpy as np
import math
import wave
import os

# Teams can add helper functions
# Add all helper functions here
def dft_c(sound,file_length):
    tol = 1e-14                                                      # threshold used to compute phase
    hN = (file_length//2)+1
    X = np.fft.fft(sound)[:((file_length)//2)+1]                                     # compute FFT
    absX = abs(X[:hN])                                      # compute ansolute value of positive side
    absX[absX<np.finfo(float).eps] = np.finfo(float).eps    # if zeros add epsilon to handle log
    mX = 20 * np.log10(absX)                                # magnitude spectrum of positive frequencies in dB
    X[:hN].real[np.abs(X[:hN].real) < tol] = 0.0            # for phase calculation set to 0 the small values
    X[:hN].imag[np.abs(X[:hN].imag) < tol] = 0.0            # for phase calculation set to 0 the small values
    pX = np.unwrap(np.angle(X[:hN]))                        # unwrapped phase spectrum of positive frequencies
    return mX, pX

def peakDetection(mX, t):
    """
    Detect spectral peak locations
    mX: magnitude spectrum, t: threshold
    returns ploc: peak locations
    """

    thresh = np.where(np.greater(mX[1:-1],t), mX[1:-1], 0); # locations above threshold
    next_minor = np.where(mX[1:-1]>mX[2:], mX[1:-1], 0)     # locations higher than the next one
    prev_minor = np.where(mX[1:-1]>mX[:-2], mX[1:-1], 0)    # locations higher than the previous one
    ploc = thresh * next_minor * prev_minor                 # locations fulfilling the three criteria
    ploc = ploc.nonzero()[0] + 1                            # add 1 to compensate for previous steps
    return ploc

def peakInterp(mX, pX, ploc):
    """
    Interpolate peak values using parabolic interpolation
    mX, pX: magnitude and phase spectrum, ploc: locations of peaks
    returns iploc, ipmag, ipphase: interpolated peak location, magnitude and phase values
    """

    val = mX[ploc]                                          # magnitude of peak bin
    lval = mX[ploc-1]                                       # magnitude of bin at left
    rval = mX[ploc+1]                                       # magnitude of bin at right
    iploc = ploc + 0.5*(lval-rval)/(lval-2*val+rval)        # center of parabola
    ipmag = val - 0.25*(lval-rval)*(iploc-ploc)             # magnitude of peaks
    ipphase = np.interp(iploc, np.arange(0, pX.size), pX)   # phase of peaks by linear interpolation
    return iploc, ipmag, ipphase

def TWM_p(pfreq, pmag, f0c):
	"""
	Two-way mismatch algorithm for f0 detection (by Beauchamp&Maher)
	pfreq, pmag: peak frequencies in Hz and magnitudes,
	f0c: frequencies of f0 candidates
	returns f0, f0Error: fundamental frequency detected and its error
	"""

	p = 0.5                                          # weighting by frequency value
	q = 1.4                                          # weighting related to magnitude of peaks
	r = 0.5                                          # scaling related to magnitude of peaks
	rho = 0.33                                       # weighting of MP error
	Amax = max(pmag)                                 # maximum peak magnitude
	maxnpeaks = 10                                   # maximum number of peaks used
	harmonic = np.matrix(f0c)
	ErrorPM = np.zeros(harmonic.size)                # initialize PM errors
	MaxNPM = min(maxnpeaks, pfreq.size)
	for i in range(0, MaxNPM) :                      # predicted to measured mismatch error
	    difmatrixPM = harmonic.T * np.ones(pfreq.size)
	    difmatrixPM = abs(difmatrixPM - np.ones((harmonic.size, 1))*pfreq)
	    FreqDistance = np.amin(difmatrixPM, axis=1)    # minimum along rows
	    peakloc = np.argmin(difmatrixPM, axis=1)
	    Ponddif = np.array(FreqDistance) * (np.array(harmonic.T)**(-p))
	    PeakMag = pmag[peakloc]
	    MagFactor = 10**((PeakMag-Amax)/20)
	    ErrorPM = ErrorPM + (Ponddif + MagFactor*(q*Ponddif-r)).T
	    harmonic = harmonic+f0c

	ErrorMP = np.zeros(harmonic.size)                # initialize MP errors
	MaxNMP = min(maxnpeaks, pfreq.size)
	for i in range(0, f0c.size) :                    # measured to predicted mismatch error
	    nharm = np.round(pfreq[:MaxNMP]/f0c[i])
	    nharm = (nharm>=1)*nharm + (nharm<1)
	    FreqDistance = abs(pfreq[:MaxNMP] - nharm*f0c[i])
	    Ponddif = FreqDistance * (pfreq[:MaxNMP]**(-p))
	    PeakMag = pmag[:MaxNMP]
	    MagFactor = 10**((PeakMag-Amax)/20)
	    ErrorMP[i] = sum(MagFactor * (Ponddif + MagFactor*(q*Ponddif-r)))

	Error = (ErrorPM[0]/MaxNPM) + (rho*ErrorMP/MaxNMP)  # total error
	f0index = np.argmin(Error)                         # get the smallest error
	f0 = f0c[f0index]                                # f0 with the smallest error
	if (np.min(ErrorPM)<=np.min(ErrorMP)):
		return ErrorPM
	else:
		return ErrorMP


def getNote(freq):
    Note = ["C","C-Sharp","D","D-Sharp","E","F","F-Sharp","G","G-Sharp","A","A-Sharp","B"]
    F  =  [16.35,17.32,18.35,19.45,20.60,21.83,23.12,24.50,25.96,27.50,29.14,30.87]
    i=0
    start=16.35
    end = 30.87

    while i in range(9):

        start=16.35*2**i
        end = 30.87*2**i
        if(freq<=16.35):
            return "C0"
        elif(freq<start):
            if freq<(oldlast+start)/2:
                return "B"+str(i-1)
            else:
                return "C"+str(i)

        for j in range(12):
            if freq < end:
                if freq<=F[j]:
                    if freq<(F[j]+F[j-1])/2:
                        return Note[j-1]+str(i)
                    else:
                        return Note[j]+str(i)
            else:
                break
        F=np.array(F)*2
        oldlast=end
        i=i+1


    return i

############################### Your Code Here ##############################################

def note_detect(audio_file):

	#   Instructions
	#   ------------
	#   Input   :   audio_file -- a single test audio_file as input argument
	#   Output  :   Detected_Note -- String corresponding to the Detected Note
	#   Example :   For Audio_1.wav file, Detected_Note = "A4"
	Detected_Note = ""

	# Add your code here
	file_length = audio_file.getnframes()

	frame_rate = audio_file.getframerate()
	sound = np.zeros(file_length)
	for i in range(file_length):
	    data = audio_file.readframes(1)
	    data = wave.struct.unpack("<h", data)
	    sound[i] = int(data[0])

	if (file_length > 44100):
        file_length = 44100
    sound = np.divide(sound, float(2**15))[:file_length]
	mX, pX = dft_c(sound,file_length)


	t = -20  # threshold in decibell

	ploc= peakDetection(mX, t)
	iploc, ipmag, ipphase = peakInterp(mX, pX, ploc)
	ipfreq = frame_rate* iploc/file_length



	minf0=15
	maxf0=8000
	f0c=np.argwhere((ipfreq>minf0) & (ipfreq<maxf0))[:,0]    #candidates for fundamental frequncy

	f0cf = ipfreq[f0c]
	Error = TWM_p(ipfreq, ipmag, f0cf)

	Detected_Note = getNote(f0cf[np.argmin(Error)])

	return Detected_Note


############################### Main Function ##############################################

if __name__ == "__main__":

	#   Instructions
	#   ------------
	#   Do not edit this function.

	# code for checking output for single audio file
	path = os.getcwd()

	file_name = path + "\Task_1.1_Audio_files\Audio_1.wav"
	audio_file = wave.open(file_name)

	Detected_Note = note_detect(audio_file)

	print("\n\tDetected Note = " + str(Detected_Note))

	# code for checking output for all audio files
	x = raw_input("\n\tWant to check output for all Audio Files - Y/N: ")

	if x == 'Y':

		Detected_Note_list = []

		file_count = len(os.listdir(path + "\Task_1.1_Audio_files"))

		for file_number in range(1, file_count):

			file_name = path + "\Task_1.1_Audio_files\Audio_"+str(file_number)+".wav"
			audio_file = wave.open(file_name)

			Detected_Note = note_detect(audio_file)

			Detected_Note_list.append(Detected_Note)

		print("\n\tDetected Notes = " + str(Detected_Note_list))
