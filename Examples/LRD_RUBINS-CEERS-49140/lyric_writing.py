import os
import numpy as np
from galfits.gsutils import photometry_to_img, _generate_cosmic_sfh_prior, z_to_agebin
import string 
from astropy.wcs import WCS
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
from astropy.table import Table
from astropy.io import fits, ascii
from astropy.stats import sigma_clipped_stats
from astropy.visualization import simple_norm 
from typing import Literal
from astropy.wcs.utils import proj_plane_pixel_scales


def prepare_galfits(lyric_path, obj_id,coord, z, cutsci_list, cutsigma_list, cutmask_list,
        filter_list, label_list, psf_list, convlist,zero_list, 
        use_sed, phot_row, sps_row, prior_path=None):

    ra, dec     = np.squeeze(coord.ra.deg), np.squeeze(coord.dec.deg)

    n_band  = len(cutsci_list)

    with open(lyric_path, "w") as f:

        header  = fits.getheader(cutsci_list[0], 0)
        wcs     = WCS(header)
        xmax    = int(header["NAXIS1"])
        pixel_scale     = round(proj_plane_pixel_scales(wcs)[0]*3600, 2)
        cutsize_arcsec  = round(0.5 * xmax * pixel_scale, 3)
        kron_radius_circ    = np.squeeze(phot_row["kron_radius_circ"].value)
        pa_theta            = np.squeeze(phot_row['theta_J2000'].value)
        re_galfit       =  (10.0 ** (-0.79) * kron_radius_circ ** 1.87)
        q_galfit        = np.squeeze(phot_row["b_image"].value) / np.squeeze(phot_row["a_image"].value)
        logM            = np.squeeze(sps_row["mstar_50"].value)
        age_list        = z_to_agebin(z)
        ebv             = np.squeeze(phot_row["ebv_mw"].value)

        f_cont_list     = _generate_cosmic_sfh_prior(z, age_list, logM)
        f_cont_write    = [[float(f_cont_list[i]), -8, 0, 0.1, use_sed] for i in range(len(f_cont_list)-1)]
        f_cont_write.append([float(f_cont_list[-1]), -8, 0, 0.1, use_sed])
        f_cont_write    = f"{f_cont_write}"


        f.write(f"# galfits config file for galaxy id={obj_id}\n")
        f.write(f"# Region information\n\n")


        f.write(f"R1) obj{obj_id}\n")
        f.write(f"R2) [{ra:.5f},{dec:.5f}]\n")
        f.write(f"R3) {z:.5f}\n\n")


        img_letter = np.array(list(string.ascii_lowercase[:n_band])) 
        img_letter = [str(item) for item in img_letter]
        print(img_letter)
        argindex = []
        

        for bb in range(n_band):
            img = fits.open(cutsci_list[bb])[0].data
            ybb = np.shape(img)[0]
            xbb = np.shape(img)[1]
            xcen0 = int((xbb + 1) / 2) - 1
            ycen0 = int((ybb + 1) / 2) - 1
            print(obj_id, filter_list[bb], np.shape(img), xcen0, ycen0)
            if img[ycen0][xcen0] == 0 or np.isnan(img[ycen0][xcen0]):
                continue
            else:
                argindex.append(int(bb))

        for b in range(len(argindex)):

            f.write(f"# input images for {label_list[b]} band\n")

            f.write(f"I{img_letter[b]}1) [{cutsci_list[b]},0]\n")
            f.write(f"I{img_letter[b]}2) {filter_list[b]}\n")
            f.write(f"I{img_letter[b]}3) [{cutsigma_list[b]},0]\n")
            f.write(f"I{img_letter[b]}4) [{psf_list[b]},0]\n")
            f.write(f"I{img_letter[b]}5) 1\n")
            f.write(f"I{img_letter[b]}6) [{cutmask_list[b]},0]\n")
            f.write(f"I{img_letter[b]}7) MJy/sr\n")
            f.write(f"I{img_letter[b]}8) {cutsize_arcsec} \n")
            f.write(f"I{img_letter[b]}9) {convlist[b]} \n")  
            f.write(f"I{img_letter[b]}10) {zero_list[b]}\n")
            f.write(f"I{img_letter[b]}11) uniform\n")
            f.write(f"I{img_letter[b]}12) [[0,-1e5,1e5,0.1,0]]\n")  # sky parameter, (value, min, max, step, fix), 0 for fix.
            f.write(f"I{img_letter[b]}13) 1\n")  # allow relative shifting, 0 for False.
            f.write(f"I{img_letter[b]}14) [[0,-5,5,0.1,1],[0,-5,5,0.1,1]] \n")
            f.write(f"I{img_letter[b]}15) {use_sed}\n\n")  # Use SED information

        f.write(f"# image atlas\n")
        f.write(f"Aa1) jwst \n")
        f.write(f"Aa2) {img_letter}\n")
        f.write(f"Aa3) 0\n")
        f.write(f"Aa4) 0\n")
        f.write(f"Aa5) []\n")
        f.write(f"Aa6) []\n")
        f.write(f"Aa7) []\n\n")

        f.write(f"Pa1) obj0 \n")
        f.write(f"Pa2) sersic\n")
        f.write(f"Pa3) [0,-0.3,0.3,0.01,1]\n") 
        f.write(f"Pa4) [0,-0.3,0.3,0.01,1]\n")
        f.write(f"Pa5) [{re_galfit},{0.3*pixel_scale},{cutsize_arcsec},{0.3*pixel_scale},1] \n")  # Re half-light radius [arcsec]
        f.write(f"Pa6) [{1.5},0.2,8,0.1, 1] \n") 
        f.write(f"Pa7) [{pa_theta},-180,180,1,1] \n")
        f.write(f"Pa8) [{q_galfit},0.1,1,0.02,1] \n")
        f.write(f"Pa9)  {f_cont_write}\n")
        f.write(f"Pa10) {age_list}\n")
        f.write(f"Pa11) [[0.004,0.001,0.04,0.001,{use_sed}]] \n")
        f.write(f"Pa12) [[0.,0.,5.1,0.1,0]] \n")
        f.write(f"Pa13) [0,40,200,1,0] \n")
        f.write(f"Pa14) [{logM}, 6, 12, 0.1, {use_sed}]\n")
        f.write(f"Pa15) bins\n")
        f.write(f"Pa16) [-1,-4,-2,0.1,0] \n")
        f.write(f"Pa26) [0,0,5,0.1,0] \n")
        f.write(f"Pa27) 3 \n")
        f.write(f"Pa28) [8.14,4.5,10,0.1,0] \n")
        f.write(f'Pa29) [1.0, 0.1, 50, 10, 1]\n') # Umin, minimum radiation field
        f.write(f'Pa30) [2, 0.45, 7, 2, 1]\n') # qPAH, mass fraction of PAH
        f.write(f'Pa31) [1.0, 1.0, 3.0, 0.1, 0]\n') # alpha, powerlaw slope of U
        f.write(f'Pa32) [0.01, 0, 1.0, 0.1, 1]\n') # gamma, fraction illuminated by star forming region
        f.write(f'Pa33) [[0.004,0.0001,0.04,0.001,1]]\n') 
        f.write(f'Pa34) [0.3, 0, 4, 1, 1]\n') # dust2, usuallary gaussian ~ G(0.3, 1, [0, 4])
        f.write(f'Pa35) [1, 0, 2, 0.3, 1]\n')  # dust ratio for dust1/dust2, usually ~G(1, 0.3, 0, 2)
        f.write(f'Pa36) [0.7, -1, 0.4, 0.1, 1]\n\n') # dust index for diffuse dust uniform [-1, 0.4]
        

        f.write(f"Ga1) mygal \n")
        f.write(f"Ga2) ['a'] \n")
        f.write(f"Ga3) [{z:.5f},0.01,12,0.01,0]\n")      # redshift 
        f.write(f"Ga4) {ebv:.3f} \n")
        f.write(f"Ga5) [1.,0.8,1.2,0.05,0] \n")      # normalization of spectrum when images+spec fitting
        f.write(f"Ga6) [] \n")
        f.write(f"Ga7) 1 \n\n")

        if use_sed and prior_path is not None:

            pfile = open(prior_path, "w")
            pfile.write(f"SFH_Ta1) obj0\n")
            pfile.write(f"SFH_Ta2) cosmic\n")
            pfile.close()


def gen_pSed_data_lyric(cat_path, z_cat_path, mock_dir, filter_list, zp_list, suffix="psed"):
    
    flux_cat = Table.read(cat_path, format = "ascii")
    z_cat = Table.read(z_cat_path, format = "ascii")

    first = 1
    for obj_row in flux_cat: # loop for every source in this group
        
        id = obj_row["id"] # 1-index
        
        print(f"create pSed mock data for {id}")

        mock_subdir = os.path.join(mock_dir,f"{id}")
        os.makedirs(mock_subdir, exist_ok=True)

        z_fit = float(z_cat[z_cat["id"] == id]["z_peak"].value[0])

        ebv = z_cat[z_cat["id"] == id]["ebv_mw"].value[0]
        mu = z_cat[z_cat["id"] == id]["mu"].value[0]
        logM =  float(z_cat[z_cat["id"] == id]["logM_50"].value[0])
        print(f"z-fit is {z_fit}, ebv is {ebv}, mu is {mu}")

        for fltr in filter_list: # loop for every filter.

            if obj_row[f"{fltr}_flux"] == -99.0:
                continue ## skip the lost band.
                    
            mock_impath = os.path.join(mock_subdir, f"{fltr}.fits")
            flux_mjy = obj_row[f"{fltr}_flux"] * 1e-5 # 10 * nJy = 1e-8 Jy -> 1e-3 Jy
            fluxerr_mjy = obj_row[f"{fltr}_fluxerr"] * 1e-5
            photometry_to_img(flux = flux_mjy, flux_err = fluxerr_mjy, z = z_fit, outputname = mock_impath, band = fltr, unit='mJy')
        ra,dec = 1, 2
        lyric_file = os.path.join(mock_subdir, f"obj{id}_pureSed.lyric")
        #prior file write
        pfile_path = os.path.join(mock_subdir, f"obj{id}.prior")
        
        pfile = open(pfile_path, "w")
        pfile.write(f"SFH_Ta1) obj0\n")
        pfile.write(f"SFH_Ta2) cosmic\n")
        pfile.write(f"GP) ['obj0_dust2', 'obj0_dust_ratio', 'qPAH_obj0', 'Umin_obj0', 'gammad_obj0']")
        pfile.close()

        imgatlas = []


        param_file = open(lyric_file,'w')
        param_file.write(f"# This is a galfitS configuration file for galaxy {id}\n")
        param_file.write("# The config file provide a galfitS setup to perform a single sersic SED fitting with multi-band images.\n")

        # Region information

        param_file.write("# Region information\n")
        param_file.write(f'R1) obj{id}_{suffix}'+'\n')  # name of the target
        param_file.write('R2) ['+str(ra)+','+str(dec)+']\n')  # sky coordinate of the target [RA, Dec]
        param_file.write('R3) '+str(z_fit)+' \n\n') # redshift of the target

        imagels = list(string.ascii_lowercase)[:len(filter_list)] + ["ab"]
        
        for i, (fltr, zp) in enumerate(zip(filter_list, zp_list)):

            imagel = imagels[i]

            if obj_row[f"{fltr}_flux"] == -99.0:
                continue
            else:
                imgatlas.append(imagel)

            mockfile = os.path.join(mock_subdir, f'{fltr}.fits') # mock path 

            param_file.write('# Image '+imagel.upper()+' \n')
            param_file.write('I'+imagel+'1)  [' + mockfile + ',0] \n') #sci image
            param_file.write('I'+imagel+f'2)  {fltr}\n') # band name

            param_file.write('I'+imagel+'3)  [' + mockfile + ',2] \n') # sigma image
            param_file.write('I'+imagel+'4)  [' + mockfile + ',3]\n') #psf image
            param_file.write('I'+imagel+'5)  1\n') # PSF fine sampling factor relative to data
            param_file.write('I'+imagel+'6)  [Noimg,0]\n') #mask image
            param_file.write('I'+imagel+'7)  cR\n') # unit of the image
            param_file.write('I'+imagel+'8)  -1 \n') # size to make cutout image region for fitting, unit arcsec
            param_file.write('I'+imagel+'9)  1 \n') # Conversion from image unit to flambda, -1 for default                 ## why ? for pure sed fitting, it must be specified as 1.
            param_file.write('I'+imagel+f'10) {zp} \n') # Magnitude photometric zeropoint                                 ## mag zp calculate
            param_file.write('I'+imagel+'11) uniform\n') # sky model
            param_file.write('I'+imagel+'12) [[0,-0.5,0.5,0.1,0]]\n') # sky parameter, (value, min, max, step)
            param_file.write('I'+imagel+'13) 0\n') # allow relative shifting
            param_file.write('I'+imagel+'14) [[0,-5,5,0.1,0],[0,-5,5,0.1,0]]\n') # [shiftx, shifty]
            param_file.write('I'+imagel+'15) 1\n\n') # Use SED information
   

        age_list    = z_to_agebin(z_fit)
        if first > 0:
            print(f"age_list is {age_list}")
            first -= 1
        f_cont_list     = _generate_cosmic_sfh_prior(z_fit, age_list, logM, )
        # f_cont_list     = [0,0,0,0,0,0,0]
        f_cont_write    = [[float(f_cont_list[i]), -5, 0, 0.1, 1] for i in range(len(f_cont_list)-1)]
        f_cont_write.append([float(f_cont_list[-1]), -5, 0, 0.1, 1])
        f_cont_write    = f"{f_cont_write}"
        
        age_write = f"{[float(age_list[i]) for i in range(len(age_list))]}"
        
        param_file.write("# Image atlas\n")
        param_file.write("Aa1) 'all'\n") # name of the image atlas
        param_file.write("Aa2) "+str(imgatlas)+"\n") # images in this atlas
        param_file.write('Aa3) 0\n') # whether the images have same pixel size
        param_file.write('Aa4) 0\n') # link relative shiftings
        param_file.write('Aa5) []\n') # spectra
        param_file.write('Aa6) []\n') # aperture size
        param_file.write('Aa7) []\n\n') # references images
        
        param_file.write("# Profile A\n")
        param_file.write(f'Pa1) obj0\n') # name of the component
        param_file.write('Pa2) sersic\n') # profile type
        param_file.write('Pa3) [0,-0.3,0.3,0.1,0]\n') # x-center [arcsec]
        param_file.write('Pa4) [0,-0.3,0.3,0.1,0]\n') # y-center [arcsec]
        param_file.write('Pa5) [0.2,0.1,1.7,0.1,0]\n') # effective radius [arcsec]
        param_file.write('Pa6) [2,0.5,6,0.1,0]\n') # Sersic index
        param_file.write('Pa7) [0,-90,90,1,0]\n') # position angle (PA) [degrees: Up=0, Left=90]
        param_file.write('Pa8) [0.8,0.5,1,0.01,0]\n') # axis ratio (b/a) [0.1=round, 1=flat]
        param_file.write(f'Pa9) {f_cont_write}\n') # contemporary log star formation fraction         ## sfr
        param_file.write(f'Pa10) {age_write}\n') # burst stellar age [Gyr]          ## age 
        param_file.write('Pa11) [[0.004,0.0001,0.04,0.001,1]]\n') # metallicity [Z=0.0142=Solar]
        param_file.write('Pa12) [[0.7,0,5.1,0.1,0]]\n') # Av dust extinction [mag]
        param_file.write('Pa13) [300,40,500,1,0]\n') # stellar velocity dispersion
        param_file.write(f'Pa14) [{logM},6,12,0.1,1]\n') # log stellar mass
        param_file.write('Pa15) bins \n') # star formation history type: burst/conti                    ## change to bins 
        param_file.write('Pa16) [-1,-4,-2,0.1,0]\n') # logU nebular ionization parameter
        param_file.write('Pa26) [0,0,5,0.1,0]\n') # amplitude of the 2175A bump on extinction curve
        param_file.write(f'Pa27) 3\n') # SED model, 0: stellar; 1: steNeb ; 2: stellar + duste; 3: steNeb + duste
        param_file.write(f'Pa28) [6,4.5,10,0.1,0]\n') # log dust mass
        param_file.write(f'Pa29) [1.0, 0.1, 50, 10, 1]\n') # Umin, minimum radiation field
        param_file.write(f'Pa30) [2, 0.45, 7, 2, 1]\n') # qPAH, mass fraction of PAH
        param_file.write(f'Pa31) [1.0, 1.0, 3.0, 0.1, 0]\n') # alpha, powerlaw slope of U
        param_file.write(f'Pa32) [0.01, 0, 1.0, 0.1, 1]\n') # gamma, fraction illuminated by star forming region
        param_file.write(f'Pa33) [[0.004,0.0001,0.04,0.001,1]]\n') 
        param_file.write(f'Pa34) [0.3, 0, 4, 1, 1]\n') # dust2, usuallary gaussian ~ G(0.3, 1, [0, 4])
        param_file.write(f'Pa35) [1, 0, 2, 0.3, 1]\n')  # dust ratio for dust1/dust2, usually ~G(1, 0.3, 0, 2)
        param_file.write(f'Pa36) [0.7, -1, 0.4, 0.1, 1]\n\n') # dust index for diffuse dust uniform [-1, 0.4]
        

        # Galaixes
        param_file.write("# Galaxy A\n")
        param_file.write('Ga1) mygal\n') # name of the galaxy
        param_file.write("Ga2) ['a']\n") # profile component
        param_file.write('Ga3) ['+str(z_fit)+',0.01,12.0,0.01,0]\n') # galaxy redshift
        param_file.write(f'Ga4) {ebv:.3f}\n') # the EB-V of Galactic dust reddening 
        param_file.write('Ga5) [1.0,0.5,2,0.05,0]\n') # normalization of spectrum when images+spec fitting
        param_file.write('Ga6) []\n') # narrow lines in nebular
        param_file.write('Ga7) 1\n\n') # number of components for narrow lines
        param_file.close()
