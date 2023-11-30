import librosa
import numpy as np
from fastdtw.fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import os


####FROM  https://github.com/jasminsternkopf/mel_cepstral_distance
## CLONED


def get_mcd_between_wav_files(wav_file_1: str, wav_file_2: str, hop_length: int = 256, n_fft: int = 1024,
                              window: str = 'hamming', center: bool = False, n_mels: int = 20, htk: bool = True,
                              norm= None, dtype: np.dtype = np.float64, n_mfcc: int = 16,
                              use_dtw: bool = True, sampling_rate: int = 16000): # -> Tuple[float, float, int]:
  """Compute the mel-cepstral distance between two audios, a penalty term accounting for the number of frames that has to
  be added to equal both frame numbers or to align the mel-cepstral coefficients if using Dynamic Time Warping and the
  final number of frames that are used to compute the mel-cepstral distance.

    Parameters
    ----------
    wav_file_1 : string
        path to the first input WAV file

    wav_file_2 : string
        path to the second input WAV file

    hop_length : int > 0 [scalar]
        specifies the number of audio samples between adjacent Short Term Fourier Transformation-columns, therefore
        plays a role in computing the (mel-)spectograms which are needed to compute the mel-cepstral coefficients
        See `librosa.core.stft`

    n_fft     : int > 0 [scalar]
        `n_fft/2+1` is the number of rows of the spectograms. `n_fft` should be a power of two to optimize the speed of
        the Fast Fourier Transformation

    window    : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, or number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.hanning`
        - a vector or array of length `n_fft`

        See `librosa.filters.get_window`

    center    : bool [scalar]
        - If `True`, the signal `audio_i` is padded so that frame `D[:, t]` with `D` being the Short-term Fourier
          transform of the audio is centered at `audio_i[t * hop_length]` for i=1,2
        - If `False`, then `D[:, t]` begins at `audio_i[t * hop_length]` for i=1,2

    n_mels    : int > 0 [scalar]
        number of Mel bands to generate

    htk       : bool [scalar]
        use HTK formula instead of Slaney when creating the mel-filter bank

    norm      : {None, 1, np.inf} [scalar]
        determines if and how the mel weights are normalized: if 1, divide the triangular mel weights by the width of
        the mel band (area normalization).  Otherwise, leave all the triangles aiming for a peak value of 1.0

    dtype     : np.dtype
        data type of the output

    n_mfcc    : int > 0 [scalar]
        the number of mel-cepstral coefficients that are computed per frame, starting with the first coefficient (the
        zeroth coefficient is omitted, as it is primarily affected by system gain rather than system distortion
        according to Robert F. Kubichek)

    use_dtw:  : bool [scalar]
        to compute the mel-cepstral distance, the number of frames has to be the same for both audios. If `use_dtw` is
        `True`, Dynamic Time Warping is used to align both arrays containing the respective mel-cepstral coefficients,
        otherwise the array with less columns is filled with zeros from the right side.

    sampling_rate: sampling rate in case of npy data

    Returns
    -------
    mcd        : float
        the mel-cepstral distance between the two input audios
    penalty    : float
        a term punishing for the number of frames that had to be added to align the mel-cepstral coefficient arrays
        with Dynamic Time Warping (for `use_dtw = True`) or to equal the frame numbers via filling up one mel-cepstral
        coefficient array with zeros (for `use_dtw = False`). The penalty is the sum of the number of added frames of
        each of the two arrays divided by the final frame number (see below). It lies between zero and one, zero is
        reached if no columns were added to either array.
    final_frame_number : int
        the number of columns of one of the mel-cepstral coefficient arrays after applying Dynamic Time Warping or
        filling up with zeros

    Example
    --------

    Comparing two audios to another audio using the sum of the mel-cepstral distance and the penalty

    >>> import librosa
    >>> mcd_12, penalty_12, _ = get_mcd_between_wav_files("exampleaudio_1.wav", "exampleaudio_2.wav")
    >>> mcd_13, penalty_13, _ = get_mcd_between_wav_files("exampleaudio_1.wav". "exampleaudio_3.wav")
    >>> mcd_with_penalty_12 = mcd_12 + penalty_12
    >>> mcd_with_penalty_13 = mcd_13 + penalty_13
    >>> if mcd_with_penalty_12 < mcd_with_penalty_13:
    >>>   print("Audio 2 seems to be more similar to audio 1 than audio 3.")
    >>> elif mcd_with_penalty_13 < mcd_with_penalty_12:
    >>>   print("Audio 3 seems to be more similar to audio 1 than audio 2.")
    >>> else:
    >>>   print("Audio 2 and audio 3 seem to share the same similarity to audio 1.")
    """
  if os.path.basename(wav_file_1).endswith(".npy"):
      audio_1 = np.load(wav_file_1)
      sr_1= sampling_rate
  else:
      audio_1, sr_1 = librosa.load(wav_file_1, mono=True)

  if os.path.basename(wav_file_2).endswith(".npy"):
      audio_2 = np.load(wav_file_2)
      sr_2= sampling_rate
  else:
      audio_2, sr_2 = librosa.load(wav_file_2, mono=True)
    
  return get_mcd_between_audios(
    audio_1=audio_1,
    audio_2=audio_2,
    sr_1=sr_1,
    sr_2=sr_2,
    hop_length=hop_length,
    n_fft=n_fft,
    window=window,
    center=center,
    n_mels=n_mels,
    htk=htk,
    norm=norm,
    dtype=dtype,
    n_mfcc=n_mfcc,
    use_dtw=use_dtw
  )


def get_mcd_between_audios(audio_1: np.ndarray, audio_2: np.ndarray, sr_1: int, sr_2: int, hop_length: int = 256,
                           n_fft: int = 1024, window: str = 'hamming', center: bool = False, n_mels: int = 20,
                           htk: bool = True, norm = None, dtype: np.dtype = np.float64, n_mfcc: int = 16,
                           use_dtw: bool = True): # -> Tuple[float, float, int]:
  """Compute the mel-cepstral distance between two audios, a penalty term accounting for the number of frames that has to
  be added to equal both frame numbers or to align the mel-cepstral coefficients if using Dynamic Time Warping and the
  final number of frames that are used to compute the mel-cepstral distance.

    Parameters
    ----------
    audio_1 	  : np.ndarray [shape=(n,)]
        first audio time-series

    audio_2     : np.ndarray [shape=(m,)]
        second audio time-series

    sr_1        : number > 0 [scalar]
        sampling rate of the first incoming signal

    sr_2        : number > 0 [scalar]
        sampling rate of the second incoming signal

    hop_length  : int > 0 [scalar]
        specifies the number of audio samples between adjacent Short Term Fourier Transformation-columns, therefore
        plays a role in computing the (mel-)spectograms which are needed to compute the mel-cepstral coefficients
        See `librosa.core.stft`

    n_fft     : int > 0 [scalar]
        `n_fft/2+1` is the number of rows of the spectograms. `n_fft` should be a power of two to optimize the speed of
        the Fast Fourier Transformation

    window    : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, or number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.hanning`
        - a vector or array of length `n_fft`

        See `librosa.filters.get_window`

    center    : bool [scalar]
        - If `True`, the signal `audio_i` is padded so that frame `D[:, t]` with `D` being the Short-term Fourier
          transform of the audio is centered at `audio_i[t * hop_length]` for i=1,2
        - If `False`, then `D[:, t]` begins at `audio_i[t * hop_length]` for i=1,2

    n_mels    : int > 0 [scalar]
        number of Mel bands to generate

    htk       : bool [scalar]
        use HTK formula instead of Slaney when creating the mel-filter bank

    norm      : {None, 1, np.inf} [scalar]
        determines if and how the mel weights are normalized: if 1, divide the triangular mel weights by the width of
        the mel band (area normalization).  Otherwise, leave all the triangles aiming for a peak value of 1.0

    dtype     : np.dtype
        data type of the output

    n_mfcc    : int > 0 [scalar]
        the number of mel-cepstral coefficients that are computed per frame, starting with the first coefficient (the
        zeroth coefficient is omitted, as it is primarily affected by system gain rather than system distortion
        according to Robert F. Kubichek)

    use_dtw:  : bool [scalar]
        to compute the mel-cepstral distance, the number of frames has to be the same for both audios. If `use_dtw` is
        `True`, Dynamic Time Warping is used to align both arrays containing the respective mel-cepstral coefficients,
        otherwise the array with less columns is filled with zeros from the right side.

    Returns
    -------
    mcd        : float
        the mel-cepstral distance between the two input audios
    penalty    : float
        a term punishing for the number of frames that had to be added to align the mel-cepstral coefficient arrays
        with Dynamic Time Warping (for `use_dtw = True`) or to equal the frame numbers via filling up one mel-cepstral
        coefficient array with zeros (for `use_dtw = False`). The penalty is the sum of the number of added frames of
        each of the two arrays divided by the final frame number (see below). It lies between zero and one, zero is
        reached if no columns were added to either array.
    final_frame_number : int
        the number of columns of one of the mel-cepstral coefficient arrays after applying Dynamic Time Warping or
        filling up with zeros

    Example
    --------

    Comparing two audios to another audio using the sum of the mel-cepstral distance and the penalty

    >>> import librosa
    >>> audio_1, sr_1 = librosa.load("exampleaudio_1.wav")
    >>> audio_2, sr_2 = librosa.load("exampleaudio_2.wav")
    >>> audio_3, sr_3 = librosa.load("exampleaudio_3.wav")
    >>> mcd_12, penalty_12, _ = get_mcd_between_audios(audio_1, audio_2, sr_1, sr_2)
    >>> mcd_13, penalty_13, _ = get_mcd_between_audios(audio_1, audio_3, sr_1, sr_3)
    >>> mcd_with_penalty_12 = mcd_12 + penalty_12
    >>> mcd_with_penalty_13 = mcd_13 + penalty_13
    >>> if mcd_with_penalty_12 < mcd_with_penalty_13:
    >>>   print("Audio 2 seems to be more similar to audio 1 than audio 3.")
    >>> elif mcd_with_penalty_13 < mcd_with_penalty_12:
    >>>   print("Audio 3 seems to be more similar to audio 1 than audio 2.")
    >>> else:
    >>>   print("Audio 2 and audio 3 seem to share the same similarity to audio 1.")
    """
  if sr_1 != sr_2:
    print("Warning: The sampling rates differ.")
  mfccs_1, melspectogram_1 = get_mfccs_and_mel_spectogram_of_audio(
    audio=audio_1,
    sr=sr_1,
    hop_length=hop_length,
    n_fft=n_fft,
    window=window,
    center=center,
    n_mels=n_mels,
    htk=htk,
    dtype=dtype,
    n_mfcc=n_mfcc,
    norm=norm,
  )
  mfccs_2, melspectogram_2 = get_mfccs_and_mel_spectogram_of_audio(
    audio=audio_2,
    sr=sr_2,
    hop_length=hop_length,
    n_fft=n_fft,
    window=window,
    center=center,
    n_mels=n_mels,
    htk=htk,
    dtype=dtype,
    n_mfcc=n_mfcc,
    norm=norm,
  )
  mcd, penalty, final_frame_number = get_mcd_and_penalty_and_final_frame_number( mfccs_1, mfccs_2, use_dtw)
  mcd2= mel_cepstral_distorision(melspectogram_1, melspectogram_2, "dtw", n_mfcc)
  return mcd2, mcd, penalty, final_frame_number



def get_mcd_between_audios_2(audio_1: np.ndarray, audio_2: np.ndarray, sr_1: int, sr_2: int, hop_length: int = 256,
                           n_fft: int = 1024, window: str = 'hamming', center: bool = False, n_mels: int = 20,
                           htk: bool = True, norm = None, dtype: np.dtype = np.float64, n_mfcc: int = 16,
                           use_dtw: bool = True): # -> Tuple[float, float, int]:
  """Compute the mel-cepstral distance between two audios, a penalty term accounting for the number of frames that has to
  be added to equal both frame numbers or to align the mel-cepstral coefficients if using Dynamic Time Warping and the
  final number of frames that are used to compute the mel-cepstral distance.

    Parameters
    ----------
    audio_1 	  : np.ndarray [shape=(n,)]
        first audio time-series

    audio_2     : np.ndarray [shape=(m,)]
        second audio time-series

    sr_1        : number > 0 [scalar]
        sampling rate of the first incoming signal

    sr_2        : number > 0 [scalar]
        sampling rate of the second incoming signal

    hop_length  : int > 0 [scalar]
        specifies the number of audio samples between adjacent Short Term Fourier Transformation-columns, therefore
        plays a role in computing the (mel-)spectograms which are needed to compute the mel-cepstral coefficients
        See `librosa.core.stft`

    n_fft     : int > 0 [scalar]
        `n_fft/2+1` is the number of rows of the spectograms. `n_fft` should be a power of two to optimize the speed of
        the Fast Fourier Transformation

    window    : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, or number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.hanning`
        - a vector or array of length `n_fft`

        See `librosa.filters.get_window`

    center    : bool [scalar]
        - If `True`, the signal `audio_i` is padded so that frame `D[:, t]` with `D` being the Short-term Fourier
          transform of the audio is centered at `audio_i[t * hop_length]` for i=1,2
        - If `False`, then `D[:, t]` begins at `audio_i[t * hop_length]` for i=1,2

    n_mels    : int > 0 [scalar]
        number of Mel bands to generate

    htk       : bool [scalar]
        use HTK formula instead of Slaney when creating the mel-filter bank

    norm      : {None, 1, np.inf} [scalar]
        determines if and how the mel weights are normalized: if 1, divide the triangular mel weights by the width of
        the mel band (area normalization).  Otherwise, leave all the triangles aiming for a peak value of 1.0

    dtype     : np.dtype
        data type of the output

    n_mfcc    : int > 0 [scalar]
        the number of mel-cepstral coefficients that are computed per frame, starting with the first coefficient (the
        zeroth coefficient is omitted, as it is primarily affected by system gain rather than system distortion
        according to Robert F. Kubichek)

    use_dtw:  : bool [scalar]
        to compute the mel-cepstral distance, the number of frames has to be the same for both audios. If `use_dtw` is
        `True`, Dynamic Time Warping is used to align both arrays containing the respective mel-cepstral coefficients,
        otherwise the array with less columns is filled with zeros from the right side.

    Returns
    -------
    mcd        : float
        the mel-cepstral distance between the two input audios
    penalty    : float
        a term punishing for the number of frames that had to be added to align the mel-cepstral coefficient arrays
        with Dynamic Time Warping (for `use_dtw = True`) or to equal the frame numbers via filling up one mel-cepstral
        coefficient array with zeros (for `use_dtw = False`). The penalty is the sum of the number of added frames of
        each of the two arrays divided by the final frame number (see below). It lies between zero and one, zero is
        reached if no columns were added to either array.
    final_frame_number : int
        the number of columns of one of the mel-cepstral coefficient arrays after applying Dynamic Time Warping or
        filling up with zeros

    Example
    --------

    Comparing two audios to another audio using the sum of the mel-cepstral distance and the penalty

    >>> import librosa
    >>> audio_1, sr_1 = librosa.load("exampleaudio_1.wav")
    >>> audio_2, sr_2 = librosa.load("exampleaudio_2.wav")
    >>> audio_3, sr_3 = librosa.load("exampleaudio_3.wav")
    >>> mcd_12, penalty_12, _ = get_mcd_between_audios(audio_1, audio_2, sr_1, sr_2)
    >>> mcd_13, penalty_13, _ = get_mcd_between_audios(audio_1, audio_3, sr_1, sr_3)
    >>> mcd_with_penalty_12 = mcd_12 + penalty_12
    >>> mcd_with_penalty_13 = mcd_13 + penalty_13
    >>> if mcd_with_penalty_12 < mcd_with_penalty_13:
    >>>   print("Audio 2 seems to be more similar to audio 1 than audio 3.")
    >>> elif mcd_with_penalty_13 < mcd_with_penalty_12:
    >>>   print("Audio 3 seems to be more similar to audio 1 than audio 2.")
    >>> else:
    >>>   print("Audio 2 and audio 3 seem to share the same similarity to audio 1.")
    """
  if sr_1 != sr_2:
    print("Warning: The sampling rates differ.")
  mfccs_1 = get_mfccs_of_audio(
    audio=audio_1,
    sr=sr_1,
    hop_length=hop_length,
    n_fft=n_fft,
    window=window,
    center=center,
    n_mels=n_mels,
    htk=htk,
    dtype=dtype,
    n_mfcc=n_mfcc,
    norm=norm,
  )
  mfccs_2 = get_mfccs_of_audio(
    audio=audio_2,
    sr=sr_2,
    hop_length=hop_length,
    n_fft=n_fft,
    window=window,
    center=center,
    n_mels=n_mels,
    htk=htk,
    dtype=dtype,
    n_mfcc=n_mfcc,
    norm=norm,
  )
  mcd, penalty, final_frame_number = get_mcd_and_penalty_and_final_frame_number( mfccs_1, mfccs_2, use_dtw)
  return mcd, penalty, final_frame_number

def get_mcd_between_mel_spectograms(mel_1: np.ndarray, mel_2: np.ndarray, n_mfcc: int = 16, take_log: bool = True,
                                    use_dtw: bool = True): #-> Tuple[float, float, int]:
  """Compute the mel-cepstral distance between two audios, a penalty term accounting for the number of frames that has to
  be added to equal both frame numbers or to align the mel-cepstral coefficients if using Dynamic Time Warping and the
  final number of frames that are used to compute the mel-cepstral distance.

    Parameters
    ----------
    mel_1 	  : np.ndarray [shape=(k,n)]
        first mel spectogram

    mel_2     : np.ndarray [shape=(k,m)]
        second mel spectogram

    take_log     : bool
        should be set to `False` if log10 already has been applied to the input mel spectograms, otherwise `True`

    n_mfcc    : int > 0 [scalar]
        the number of mel-cepstral coefficients that are computed per frame, starting with the first coefficient (the
        zeroth coefficient is omitted, as it is primarily affected by system gain rather than system distortion
        according to Robert F. Kubichek)

    use_dtw:  : bool [scalar]
        to compute the mel-cepstral distance, the number of frames has to be the same for both audios. If `use_dtw` is
        `True`, Dynamic Time Warping is used to align both arrays containing the respective mel-cepstral coefficients,
        otherwise the array with less columns is filled with zeros from the right side.

    Returns
    -------
    mcd         : float
        the mel-cepstral distance between the two input audios
    penalty     : float
        a term punishing for the number of frames that had to be added to align the mel-cepstral coefficient arrays
        with Dynamic Time Warping (for `use_dtw = True`) or to equal the frame numbers via filling up one mel-cepstral
        coefficient array with zeros (for `use_dtw = False`). The penalty is the sum of the number of added frames of
        each of the two arrays divided by the final frame number (see below). It lies between zero and one, zero is
        reached if no columns were added to either array.
    final_frame_number : int
        the number of columns of one of the mel-cepstral coefficient arrays after applying Dynamic Time Warping or
        filling up with zeros
    """
  if mel_1.shape[0] != mel_2.shape[0]:
    print("Warning: the numbers of mel-bands that were used to compute the corresponding mel-spectogram differ.")
  mfccs_1 = get_mfccs_of_mel_spectogram(mel_1, n_mfcc, take_log)
  mfccs_2 = get_mfccs_of_mel_spectogram(mel_2, n_mfcc, take_log)
  mcd, penalty, final_frame_number = get_mcd_and_penalty_and_final_frame_number(
    mfccs_1=mfccs_1, mfccs_2=mfccs_2, use_dtw=use_dtw)
  mcd2= mel_cepstral_distorision(mel_1, mel_2, "dtw", n_mfcc)
  return mcd2, mcd, penalty, final_frame_number


def get_mfccs_of_audio(audio: np.ndarray, sr: int, hop_length: int, n_fft: int, window: str,
                       center: bool, n_mels: int, htk: bool, dtype: np.dtype,
                       n_mfcc: int, norm= None):# -> np.ndarray:
  mel_spectogram = librosa.feature.melspectrogram(
    audio,
    sr=sr,
    hop_length=hop_length,
    n_fft=n_fft, window=window,
    center=center,
    n_mels=n_mels,
    htk=htk,
    norm=norm,
    dtype=dtype
  )
  mfccs = get_mfccs_of_mel_spectogram(mel_spectogram, n_mfcc)
  return mfccs


def get_mfccs_and_mel_spectogram_of_audio(audio: np.ndarray, sr: int, hop_length: int, n_fft: int, window: str,
                       center: bool, n_mels: int, htk: bool, dtype: np.dtype,
                       n_mfcc: int, norm= None):# -> np.ndarray:
  mel_spectogram = librosa.feature.melspectrogram(
    audio,
    sr=sr,
    hop_length=hop_length,
    n_fft=n_fft, window=window,
    center=center,
    n_mels=n_mels,
    htk=htk,
    norm=norm,
    dtype=dtype
  )
  mfccs = get_mfccs_of_mel_spectogram(mel_spectogram, n_mfcc)
  return mfccs, mel_spectogram


def get_mfccs_of_mel_spectogram(mel_spectogram: np.ndarray, n_mfcc: int, take_log: bool = True): #-> np.ndarray:
  mel_spectogram = np.log10(mel_spectogram) if take_log else mel_spectogram
  mfccs = librosa.feature.mfcc(
    S=mel_spectogram,
    n_mfcc=n_mfcc + 1,
    norm=None
  )
  # according to "Mel-Cepstral Distance Measure for Objective Speech Quality Assessment" by R. Kubichek, the zeroth
  # coefficient is omitted
  # there are different variants of the Discrete Cosine Transform Type II, the one that librosa's mfcc uses is 2 times
  # bigger than the one we want to use (which appears in Kubicheks paper)
  mfccs = 1 / 2 * mfccs[1:]
  return mfccs


def get_mcd_and_penalty_and_final_frame_number(mfccs_1: np.ndarray, mfccs_2: np.ndarray, use_dtw: bool
                                               ): #-> Tuple[float, float, int]:
  former_frame_number_1 = mfccs_1.shape[1]
  former_frame_number_2 = mfccs_2.shape[1]
  mcd, final_frame_number = equal_frame_numbers_and_get_mcd(
    mfccs_1, mfccs_2, use_dtw)
  penalty = get_penalty(former_frame_number_1,
                        former_frame_number_2, final_frame_number)
  return mcd, penalty, final_frame_number


def equal_frame_numbers_and_get_mcd(mfccs_1: np.ndarray, mfccs_2: np.ndarray,
                                    use_dtw: bool): #-> Tuple[float, int]:
  if mfccs_1.shape[0] != mfccs_2.shape[0]:
    raise Exception("The number of coefficients per frame has to be the same for both inputs.")
  if use_dtw:
    mcd, final_frame_number = get_mcd_with_dtw(mfccs_1, mfccs_2)
    return mcd, final_frame_number
  mfccs_1, mfccs_2 = fill_rest_with_zeros(mfccs_1, mfccs_2)
  mcd, final_frame_number = get_mcd(mfccs_1, mfccs_2)
  return mcd, final_frame_number


def fill_rest_with_zeros(mfccs_1: np.ndarray, mfccs_2: np.ndarray): #-> Tuple[np.ndarray, np.ndarray]:
  frame_number_1 = mfccs_1.shape[1]
  frame_number_2 = mfccs_2.shape[1]
  diff = abs(frame_number_1 - frame_number_2)
  if diff > 0:
    adding_array = np.zeros(shape=(mfccs_1.shape[0], diff))
    if frame_number_1 < frame_number_2:
      mfccs_1 = np.concatenate((mfccs_1, adding_array), axis=1)
    else:
      mfccs_2 = np.concatenate((mfccs_2, adding_array), axis=1)
  assert mfccs_1.shape == mfccs_2.shape
  return mfccs_1, mfccs_2


def get_mcd_with_dtw(mfccs_1: np.ndarray, mfccs_2: np.ndarray): #-> Tuple[float, int]:
  mfccs_1, mfccs_2 = mfccs_1.T, mfccs_2.T
  distance, path_between_mfccs = fastdtw(np.nan_to_num(mfccs_1), np.nan_to_num(mfccs_2), dist=euclidean)
  final_frame_number = len(path_between_mfccs)
  mcd = distance / final_frame_number
  return mcd, final_frame_number


def get_mcd(mfccs_1: np.ndarray, mfccs_2: np.ndarray): #-> Tuple[float, int]:
  assert mfccs_1.shape == mfccs_2.shape
  mfccs_diff = mfccs_1 - mfccs_2
  mfccs_diff_norms = np.linalg.norm(mfccs_diff, axis=0)
  mcd = np.mean(mfccs_diff_norms)
  frame_number = len(mfccs_diff_norms)
  return mcd, frame_number


def get_penalty(former_length_1: int, former_length_2: int, length_after_equaling: int): #-> float:
  # lies between 0 and 1, the smaller the better
  penalty = 2 - (former_length_1 + former_length_2) / length_after_equaling
  return penalty



def testmcd(C, C_hat):
    """C and C_hat are NumPy arrays of shape (T, D),
    representing mel-cepstral coefficients.

    """
    K = 10 / np.log(10) * np.sqrt(2)
    return K * np.mean(np.sqrt(np.sum((C - C_hat) ** 2, axis=1)))


def testmcd2(cvt_mcc_dtw, trg_mcc_dtw):
    diff2sum = np.sum((cvt_mcc_dtw - trg_mcc_dtw)**2, 1)
    mcd = np.mean(10.0 / np.log(10.0) * np.sqrt(2 * diff2sum), 0)
    return mcd


#(https://github.com/Tomiinek/Multilingual_Text_to_Speech/audio.py

def get_spectrogram_mfcc(S, num_mfcc):
    """Compute MFCCs of a mel spectrogram."""
    return librosa.feature.mfcc(n_mfcc=num_mfcc, S=(S/10))

def get_spectrogram_mfccv2(S, num_mfcc):
    return librosa.feature.mfcc(n_mfcc=num_mfcc, S=(S/10))

#(https://github.com/Tomiinek/Multilingual_Text_to_Speech/audio.py
#def get_mfcc(y):
#    """Compute MFCCs of a waveform."""
#    return get_mfcc(audio.mel_spectrogram(y))


#(https://github.com/Tomiinek/Multilingual_Text_to_Speech/audio.py
def mel_cepstral_distorision(S1, S2, mode, n_mfcc):
    """Compute Mel Cepstral Distorsion between two mel spectrograms.

    Arguments:
        S1 and S2 -- mel spectrograms
        mode -- 'cut' to cut off frames of longer seq.
                'stretch' to stretch linearly the shorter seq.
                'dtw' to compute DTW with minimal possible MCD
    """

    def mcd(s1, s2):
        diff = s1 - s2
        return np.average(np.sqrt(np.sum(diff*diff, axis=0)))

    x, y = get_spectrogram_mfcc(S1, n_mfcc)[1:], get_spectrogram_mfcc(S2, n_mfcc)[1:]

    if mode == 'cut':
        if y.shape[1] > x.shape[1]: y = y[:,:x.shape[1]]
        if x.shape[1] > y.shape[1]: x = x[:,:y.shape[1]]

    elif mode == 'stretch':
        if x.shape[1] > y.shape[1]:
            m = x.shape[1]
            y = np.array([y[:, i * y.shape[1]//m] for i in range(m)]).T
        else:
            m = y.shape[1]
            x = np.array([x[:, i * x.shape[1]//m] for i in range(m)]).T

    elif mode == 'dtw':       
        x, y = x.T, y.T
        _, path = fastdtw(x, y, dist=mcd)     
        pathx, pathy = map(list,zip(*path))    
        x, y = x[pathx].T, y[pathy].T

    return mcd(x, y)



#(https://github.com/Tomiinek/Multilingual_Text_to_Speech/audio.py
def mel_cepstral_distorision_V2(S1, S2, num_mfcc):
  
    def mcd(s1, s2):
        diff = s1 - s2
        return np.average(np.sqrt(np.sum(diff*diff, axis=0)))

    x, y = get_spectrogram_mfcc(S1, num_mfcc)[1:], get_spectrogram_mfcc(S2, num_mfcc)[1:]

    x, y = x.T, y.T
    _, path = fastdtw(x, y, dist=mcd)     
    pathx, pathy = map(list,zip(*path))    
    x, y = x[pathx].T, y[pathy].T     
        
    return mcd(x, y)


#(https://github.com/Tomiinek/Multilingual_Text_to_Speech/audio.py
def confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m-h, m+h
