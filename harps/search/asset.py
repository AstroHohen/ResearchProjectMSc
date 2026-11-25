"""
ASSET search class, copied unchanged from HR/search/asset.ipynb (only the %run lines are removed).
ASSET_HARPS at the bottom holds the one change HARPS needs.
"""
from statistics import mode

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from astropy.io import fits



class ASSET():

    Ca_K = 393.366 #Ca II K line in nm
    Ca_H = 396.847 #Ca II H line in nm
    # UVES barycentric correction fallback coordinates (ESO UT2)
    UVES_LONGITUDE = 289.5951694
    UVES_LATITUDE = -24.627166666667
    UVES_ALTITUDE = 2635.43
    # undesired_otypes = ['RotV*alf2CVn', 'SB*', 'PulsV*delSct', 'RRLyr',
    #     'PulsV*bCep', 'V*', 'PulsV*', 'EB*', 'Seyfert_1', 'RotV*', 
    #     'Cepheid', 'EllipVar', 'PulsV*RVTau', 'gammaDor', 'HMXB', 'Nova']

    def __init__(self, line, parameters = None):
        '''initialise with param.json when we want to perform search
        options for line: either 'H' or 'K'
        '''
        self.df = None
        self.target_red = None
        self.target_san = None
        self.target_harps = None

        self.line = line

        self.spectra = None
        self.spectra_err = None
        self.wavelength = None #given in Angstrom
        self.radial_velocity = None
        self.barycentric_corr = None
        self.fits_files = None
        self.pyasl = None

        self.tot_err_spec = []
        self.tot_spec = []

        self.ok = []

        self.snr_idxrange = None

        self.ccf = False #By default ccf flag is off - activated when act_cutoff on

        # Resolution-first degradation controls/state.
        self.resolution_degrade_first = True
        self.resolution_target_resFirst = None
        self.spectra_original_resFirst = None
        self.spectra_degraded_resFirst = None

        if parameters != None:
        
            self.threshold = parameters["threshold"]
            self.pts_filter = parameters["med_filter_bin"]
            self.rv_min = parameters["rv_min"]
            self.rv_max = parameters["rv_max"]
            self.cutoff = parameters["cutoff"]
            self.snr_cut = parameters.get("SNR_cut")  # only used when spec_analysis(snr_filt=True)
            self.width_filter = parameters["width_filt"]
            self.resolution_degrade_first = bool(parameters.get('resolution_degrade_first', True))


    def _row_barycentric_correction(self, fits_path):
        '''Return UVES barycentric velocity correction in km/s from FITS header or compute fallback.'''
        try:
            with fits.open(fits_path) as fitsfile:
                header = fitsfile[0].header

                try:
                    return float(header['HIERARCH ESO QC VRAD BARYCOR'])
                except KeyError:
                    if self.pyasl is None:
                        try:
                            from PyAstronomy import pyasl as _pyasl
                            self.pyasl = _pyasl
                        except ImportError:
                            raise ImportError('PyAstronomy is required to compute missing barycentric corrections.')

                    ra2000 = float(header['RA'])
                    dec2000 = float(header['DEC'])
                    jd = float(header['MJD-OBS']) + 2400000.5

                    barycorr, _ = self.pyasl.helcorr(
                        self.UVES_LONGITUDE,
                        self.UVES_LATITUDE,
                        self.UVES_ALTITUDE,
                        ra2000,
                        dec2000,
                        jd
                    )
                    return float(barycorr)
        except Exception:
            return 0.0

    def apply_barycentric_correction(self):
        '''Shift every spectrum onto the barycentric frame using UVES FITS header values.'''
        if self.df is None or self.spectra is None or self.radial_velocity is None:
            return

        if self.fits_files is None or len(self.fits_files) != len(self.spectra):
            self.df['BarycentricCorr_km_s'] = 0.0
            self.barycentric_corr = np.zeros(len(self.spectra), dtype=float)
            return

        corrections = np.array([self._row_barycentric_correction(path) for path in self.fits_files], dtype=float)
        corrected_spectra = np.zeros_like(self.spectra, dtype=float)

        for i, spec in enumerate(self.spectra):
            corr = corrections[i]
            corrected_spectra[i, :] = np.interp(
                self.radial_velocity - corr,
                self.radial_velocity,
                spec,
                left=spec[0],
                right=spec[-1]
            )

        self.spectra = corrected_spectra
        self.barycentric_corr = corrections
        self.df['BarycentricCorr_km_s'] = corrections

    @staticmethod
    def sigma_pixels_for_resolution_degradation_resFirst(R_current, R_target):
        '''Return gaussian sigma in resolution units needed to degrade from R_current to R_target.'''
        if not np.isfinite(R_current) or not np.isfinite(R_target):
            return 0.0
        if R_current <= 0 or R_target <= 0:
            return 0.0
        if R_current <= R_target:
            return 0.0

        fwhm_current = 1.0 / float(R_current)
        fwhm_target = 1.0 / float(R_target)
        fwhm_kernel = (fwhm_target ** 2 - fwhm_current ** 2) ** 0.5
        sigma_resolution_units = fwhm_kernel / 2.354820045
        return max(0.0, sigma_resolution_units)

    def degrade_one_spectrum_resFirst(self, spec, R_current, R_target):
        '''Degrade one spectrum to the target resolution using a gaussian kernel.'''
        sigma_r = self.sigma_pixels_for_resolution_degradation_resFirst(R_current, R_target)
        if sigma_r <= 0:
            return spec

        sigma_pix = max(0.5, sigma_r * len(spec))
        return gaussian_filter1d(spec, sigma=sigma_pix, mode='nearest')

    def degrade_group_resolution_resFirst(self):
        '''Degrade mixed-resolution grouped spectra before normalisation (resolution-first path).'''
        if self.df is None or self.spectra is None:
            return

        if 'SPEC_RES' not in self.df.columns:
            return

        rvals = pd.to_numeric(self.df['SPEC_RES'], errors='coerce').to_numpy()
        finite = np.isfinite(rvals) & (rvals > 0)
        if not finite.any():
            return

        r_target = float(np.nanmin(rvals[finite]))
        degraded = np.array([
            self.degrade_one_spectrum_resFirst(self.spectra[i], rvals[i], r_target)
            for i in range(len(rvals))
        ])

        self.spectra_original_resFirst = self.spectra.copy()
        self.spectra_degraded_resFirst = degraded
        self.spectra = degraded
        self.resolution_target_resFirst = r_target

        self.df['SPEC_RES_original_resFirst'] = self.df['SPEC_RES']
        self.df['SPEC_RES_resFirst'] = r_target

    def load_info(self, star_path, init = True):
        'Init goes to false when no search is done'
        self.df = pd.read_pickle(star_path+'meta/group_df.pkl')
        self.fits_files = np.load(star_path + 'meta/fits.npy', allow_pickle=True)
        # Normalise column names to handle both 'Object' and 'OBJECT' conventions
        self.df.columns = [c if c not in ('OBJECT', 'Object') else 'Object' for c in self.df.columns]
        self.target_red = mode(self.df.Reduced)
        self.target_san = mode(self.df.Sanitised)
        self.target_harps = mode(self.df.Object)

        if self.line == 'K':
            self.spectra = np.load(star_path+'spec/sK.npy')
            self.wavelength = np.load(star_path + 'wavelength/wK.npy')
            self.radial_velocity = np.load(star_path + 'wavelength/rvK.npy')
        elif self.line == 'H':
            self.spectra = np.load(star_path+'spec/sH.npy')
            self.wavelength = np.load(star_path + 'wavelength/wH.npy')
            self.radial_velocity = np.load(star_path + 'wavelength/rvH.npy')
        else:
            return NotImplementedError

        self.apply_barycentric_correction()

        if init == True:
            self.snr_idxrange = np.where((self.radial_velocity > self.rv_min) & (self.radial_velocity < self.rv_max))

    def snr_cutoff(self):
        snr = self.df.SNR.to_numpy()
        self.df = self.df[snr > self.snr_cut].reset_index(drop = True)
        self.spectra = self.spectra[snr > self.snr_cut]
        if self.spectra_degraded_resFirst is not None:
            self.spectra_degraded_resFirst = self.spectra_degraded_resFirst[snr > self.snr_cut]
        if self.spectra_original_resFirst is not None:
            self.spectra_original_resFirst = self.spectra_original_resFirst[snr > self.snr_cut]
        if self.barycentric_corr is not None:
            self.barycentric_corr = self.barycentric_corr[snr > self.snr_cut]
        if self.fits_files is not None:
            self.fits_files = self.fits_files[snr > self.snr_cut]

    def noise_cut(self, spec):
        ''' Make an initial cut on spectra that are too noisy to observe a 4sigma signal '''
        ref, _ = self.ref(spec)
        new_ref = ref[self.ok]

        new_spectrum = spec[:,self.ok]
        # print(new_ref.shape, new_spectrum.shape)

        # Determine value that defines a 1sigma signal
        cutoff = np.median(new_ref) * 0.25

        # Determine error in each spectrum
        self.spec_err(new_ref, new_spectrum)
        keep = self.spectra_err <= cutoff
        new = spec[keep]
        self.df = self.df[keep].reset_index(drop = True)
        if self.spectra_degraded_resFirst is not None:
            self.spectra_degraded_resFirst = self.spectra_degraded_resFirst[keep]
        if self.spectra_original_resFirst is not None:
            self.spectra_original_resFirst = self.spectra_original_resFirst[keep]
        if self.barycentric_corr is not None:
            self.barycentric_corr = self.barycentric_corr[keep]
        if self.fits_files is not None:
            self.fits_files = self.fits_files[keep]

        return new        

    def norm(self,rv,s):
        ''' Normalise the input spectrum array
        Returns array same shape as input
        '''
        self.ok = []
        norm_s = np.zeros_like(s)
        
        for i in range(len(rv)): #Finding indices of values in the conditions
            # if rv[i] < -100:
            if rv[i] < -200:
                self.ok.append(i)
            if rv[i] > 200:
            # if rv[i] > 100:
                self.ok.append(i)
            continue
        
        for i in range(len(s)): #Normalising all spectra
            spec = s[i,:]
            mean = np.mean(spec[self.ok])
            norm_s[i,:] = spec/mean
        
        return norm_s

    def ref(self, spectrum):
        '''Calculates reference spec and its error'''
        median = np.median(spectrum, axis = 0)
        std =  np.std(spectrum, axis = 0)
        
        return median, std
    
    def spec_err(self, ref, s):
        self.spectra_err = np.std(np.subtract(s, ref), axis=1)

    def smoothing(self, spec, pts):
        '''Smooting data using a rolling median filter
        Input 2D array of shape (x,2000)
        Return same shape array'''
        new_spec = ndimage.median_filter(spec, size=(1,pts), mode="nearest")
        return new_spec

    def snr(self, s, m, err_s, err_m):
        '''
        Input spec 1Darray, median ref
        Returns snr 1Darray of input 
        Calculates snr and adds noise in quadrature
        '''
        # new_std = np.std(s - m, axis = 0)
        noise = np.sqrt(err_s**2 + err_m**2)
        snr = (s - m)/noise
        return snr

    def spec_analysis(self, star, snr_filt = False):
        # Load necessary info for search e.g. spectra, wavelength...
        self.load_info(star)

        if len(self.df) <= 1:
            print('[{}] not enough spectra for the search!'.format(self.target_red))
            return None

        # Resolution-first path: degrade grouped spectra before any normalisation.
        if self.resolution_degrade_first:
            self.degrade_group_resolution_resFirst()

        if snr_filt == True:
            self.snr_cutoff()

        # Normalise spectra
        norm_spectra_resFirst = self.norm(self.radial_velocity, self.spectra)
        # Smooth spectra
        new_spectra_resFirst = self.smoothing(norm_spectra_resFirst, self.pts_filter)

        new_spectra_resFirst = self.noise_cut(new_spectra_resFirst)

        if len(self.df) <= 1:
            print('[{}] not enough spectra for the search after cutting noisy spectra!'.format(self.target_red))
            return None

        med_resFirst, med_err_resFirst = self.ref(new_spectra_resFirst)
        self.spec_err(med_resFirst, new_spectra_resFirst)

        cond = (self.radial_velocity > -100) & (self.radial_velocity < 100)
        filtered_med_resFirst = med_resFirst[cond]

        # Checking for stellar activity i.e. strong emission +/- 100km/s from systemic velocity
        if len(np.where(filtered_med_resFirst > self.cutoff)[0]) > 0: #Changed to cutoff when MED (not spectrum) is >cutoff
            self.ccf = True # Flag changed to No CCF
            # return None
        # else:
        #     return None

        return [new_spectra_resFirst, med_resFirst, med_err_resFirst]

    @staticmethod
    def width_left(index, arr):
        count = 0
        l_arr = arr[:index]

        if len(l_arr) > 1:
            for i in range(1, len(l_arr) + 1):
                if arr[index - i] <= -3:
                    count += 1
                else:
                    return count
        elif len(l_arr) == 1:
            if arr[index - 1] <= -3:
                count += 1

        return count

    @staticmethod
    def width_right(index, arr):
        count = 0
        r_arr = arr[index + 1:]

        if len(r_arr) > 1:
            for i in range(1, len(r_arr) + 1):
                if arr[index + i] <= -3:
                    count += 1
                else:
                    return count
        elif len(r_arr) == 1:
            if arr[index + 1] <= -3:
                count += 1

        return count

    def get_width(self, sig):
        idx_min = np.nanargmin(sig)
        width = self.width_left(idx_min, sig) + self.width_right(idx_min, sig) + 1
        return width

    def X_corr(self, med):
        tw = np.linspace(self.wavelength.min(), self.wavelength.max(), 2000)
        if self.line == 'K':
            tf = np.exp(-(tw-(self.Ca_K*10))**2/(2.*0.2**2))
        elif self.line == 'H':
            tf = np.exp(-(tw-(self.Ca_H*10))**2/(2.*0.2**2))
        else:
            return NotImplementedError

        cond = (self.radial_velocity > -100) & (self.radial_velocity < 100)
        rv = self.radial_velocity[cond]
        corr = signal.correlate(med[cond], tf[cond], mode='same')
        rv_shift = rv[np.argmax(corr)]
        
        return rv_shift
    
    def is_valid_rv_detection(self, rv_detect, lower=-50, upper=100):
        return lower < rv_detect < upper

    def quicksearch(self, star):
        ''' Quick search given target information 
            Returns name of the target if it is a candidate
                    None if not a candidate '''
        self.ccf = False
        spec_param = self.spec_analysis(star)

        # if spec_param[1] == None:
        #     return spec_param[0]
        if spec_param == None:
            return None
        else:
            new_spectra, med, med_err = spec_param

        # if self.check_otypes == 1:
        #     otype = self.filter_otypes(self.target_san)
            
        #     if otype == None:
        #         return None
        
        # CCF
        corr_med = med.copy()
        if self.ccf == True:
            rv_shift = self.X_corr(corr_med)
            # select area +/- 50km/s from systemic velocity
            cond100 = (self.radial_velocity > rv_shift-50) & (self.radial_velocity < rv_shift+50)
            # corr_med[cond100] = np.nan

        for i in range(len(new_spectra)):

            spec = new_spectra[i]
            
            snr = self.snr(spec, med, self.spectra_err[i], med_err)
                
            sd = np.std(snr)

            corr_snr = snr.copy()
            if self.ccf == True:
                corr_snr[cond100] = np.nan
            corr_snr = corr_snr[self.snr_idxrange]
            
            sig = corr_snr/sd

            min_detect = np.nanmin(sig)

            if min_detect < self.threshold:
                # print('detection----------------------')
                width = self.get_width(sig)
                if width >= self.width_filter:
                    return self.target_red
        
        return None


class ASSET_HARPS(ASSET):

    def apply_barycentric_correction(self):
        '''HARPS spectra are already barycentric (SPECSYS = BARYCENT): nothing to shift.'''
        if self.spectra is None:
            return
        self.barycentric_corr = np.zeros(len(self.spectra), dtype=float)
        self.df['BarycentricCorr_km_s'] = 0.0

    def degrade_one_spectrum_resFirst(self, spec, R_current, R_target):
        '''Fixes the sigma of ASSET, which multiplied by len(spec) as if the cut-out spanned dlambda/lambda = 1.
        sigma_r is in dlambda/lambda, so it becomes pixels through c / (km/s per pixel).'''
        sigma_r = self.sigma_pixels_for_resolution_degradation_resFirst(R_current, R_target)
        if sigma_r <= 0:
            return spec

        sigma_pix = sigma_r * 3e5 / np.median(np.diff(self.radial_velocity))
        return gaussian_filter1d(spec, sigma=sigma_pix, mode='nearest')

