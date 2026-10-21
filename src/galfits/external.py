"""External tool helpers.

Currently contains helpers for generating configuration files for third-party
astronomy tools used alongside GalfitS.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple


_SEXTRACTOR_PARAM_STANDARD = """\
NUMBER
ID_PARENT
X_IMAGE
Y_IMAGE
XMIN_IMAGE
YMIN_IMAGE
XMAX_IMAGE
YMAX_IMAGE
XWIN_IMAGE
YWIN_IMAGE
FLUX_ISO
FLUXERR_ISO
MAG_ISO
MAGERR_ISO
FLUX_AUTO[2]
FLUXERR_AUTO[2]
MAG_AUTO[2]
MAGERR_AUTO[2]
FLUX_APER[1]
FLUXERR_APER[1]
MAG_APER[1]
MAGERR_APER[1]
KRON_RADIUS
FLUX_RADIUS
FLAGS
FLAGS_WIN
NITER_WIN
SNR_WIN
FLAGS
ELONGATION
CLASS_STAR
IMAFLAGS_ISO
"""


def _pick_default_nnw_path() -> str:
	"""Pick a reasonable default `default.nnw` path.

	Returns an absolute path string if found, otherwise the original template
	path (so callers can override by editing output config if needed).
	"""
	candidates = [
		Path("/nfs/home/liruancun/Softwares/default.nnw"),
		Path("/home/liruancun/Softwares/default.nnw"),
	]
	for path in candidates:
		if path.exists():
			return str(path)
	return str(candidates[-1])


def write_sex_config(
	catname: str,
	minarea: int,
	photaperture: float,
	gain: float,
	pixelscale: float,
	seeingfwhm: float,
	flagimag: str,
	weightimage: str,
) -> Tuple[Path, Path]:
	"""Write a standard SExtractor config and corresponding param file.

	This uses a fixed "standard" template (provided by the user) and replaces
	the following keywords:
	- CATALOG_NAME
	- PARAMETERS_NAME (points to the generated param file)
	- DETECT_MINAREA
	- PHOT_APERTURES
	- GAIN
	- PIXEL_SCALE
	- SEEING_FWHM
	- FLAG_IMAGE
	- WEIGHT_IMAGE

	Files are written next to `catname`:
	- config: same stem as catalog, suffix `.sex`
	- param:  same stem as catalog, suffix `.param`

	Returns
	-------
	(config_path, param_path)
		Paths to the written files.
	"""
	cat_path = Path(catname)
	base_dir = cat_path.parent if str(cat_path.parent) not in ("", ".") else Path(".")
	config_path = base_dir / f"{cat_path.stem}.sex"
	param_path = base_dir / f"{cat_path.stem}.param"

	# Write param file first so config can reference it.
	param_path.write_text(_SEXTRACTOR_PARAM_STANDARD, encoding="utf-8")

	nnw_path = _pick_default_nnw_path()
	flagimag_value = "NONE" if flagimag is None else str(flagimag)
	weightimage_value = "NONE" if weightimage is None else str(weightimage)

	# Keep the template very close to the provided "standard input".
	# Whitespace in SExtractor configs is flexible; we use tabs for readability.
	config_text = (
		"# SExtractor configuration file\n"
		"# For use with \"KECK\" images wregister'ed to ACS coordinates\n"
		"# (*) indicates parameters which can be omitted from this config file.\n\n"
		"#-------------------------------- Catalog ------------------------------------\n\n"
		f"CATALOG_NAME\t{catname}\t# name of the output catalog\n"
		"CATALOG_TYPE\tFITS_LDAC\t# \"ASCII_HEAD\",\"ASCII\",\"FITS_1.0\" or \"FITS_LDAC\"\n"
		f"PARAMETERS_NAME\t{param_path.resolve()}\t# name of the file containing catalog contents\n"
		"#ASSOC_NAME      cross_list.txt\n"
		"#ASSOC_PARAMS    1,2\n"
		"#ASSOC_TYPE      NEAREST\n"
		"#------------------------------- Extraction ----------------------------------\n\n"
		"DETECT_TYPE\tCCD\t\t# \"CCD\" or \"PHOTO\" (*)\n"
		f"DETECT_MINAREA\t{minarea}\t\t# minimum number of pixels above threshold\n"
		"DETECT_THRESH\t2           \t# <sigmas> or <threshold>,<ZP> in mag.arcsec-2\n"
		"ANALYSIS_THRESH\t2\t\t# <sigmas> or <threshold>,<ZP> in mag.arcsec-2\n\n"
		"FILTER\t\tN\t\t# apply filter for detection (\"Y\" or \"N\")?\n"
		"FILTER_NAME\tdefault.conv # name of the file containing the filter\n\n"
		"DEBLEND_NTHRESH\t64\t\t# Number of deblending sub-thresholds\n"
		"DEBLEND_MINCONT\t0.002\t        # Minimum contrast parameter for deblending\n\n"
		"CLEAN\t\tY\t\t# Clean spurious detections? (Y or N)?\n"
		"CLEAN_PARAM\t1\t\t# Cleaning efficiency\n"
		"MASK_TYPE\tCORRECT\t\t# Blank detected objects (Y or N)?\n\n"
		"#------------------------------ Photometry -----------------------------------\n\n"
		f"PHOT_APERTURES\t{photaperture}\t\t\t\t# MAG_APER aperture diameter(s) in pixels\n"
		"PHOT_AUTOPARAMS\t2.5, 3.5\t# MAG_AUTO parameters: <Kron_fact>,<min_radius>\n"
		"PHOT_FLUXFRAC\t0.5\t\t# Fraction-of-light for FLUX_RADIUS\n\n"
		"SATUR_LEVEL\t1e9\t\t# level (in ADUs) at which arises saturation ***\n\n"
		"MAG_ZEROPOINT\t28.9 # magnitude zero-point *** FROM COMMAND LINE\n"
		"MAG_GAMMA\t4.0\t\t# gamma of emulsion (for photographic scans)\n"
		f"GAIN\t\t{gain}\t\t# detector gain in e-/ADU.\n"
		f"PIXEL_SCALE\t{pixelscale}\t\t# size of pixel in arcsec (0=use FITS WCS info).\n\n"
		"#------------------------- Star/Galaxy Separation ----------------------------\n\n"
		f"SEEING_FWHM\t{seeingfwhm}\t\t# stellar FWHM in arcsec\n"
		f"STARNNW_NAME\t{nnw_path}\t# Neural-Network_Weight table filename\n\n"
		"#------------------------------ Background -----------------------------------\n\n"
		"# For detection only\n"
		"BACK_TYPE\tAUTO\t\t# Use the value supplied in BACK_VALUE\n"
		"BACK_VALUE\t0\t\t# Value to be subtracted from image\n\n"
		"BACK_SIZE\t50\t\t# Background mesh: <size> or <width>,<height>\n"
		"BACK_FILTERSIZE\t3\t\t# Background filter: <size> or <width>,<height>\n\n"
		"BACKPHOTO_TYPE\tLOCAL\t\t# can be \"GLOBAL\" or \"LOCAL\" (*)\n"
		"BACKPHOTO_THICK 24\t\t# thickness of the background LOCAL annulus (*)\n\n"
		"#INTERP_TYPE\tNONE\t\t# Interpolation for variance/weight maps\n\n"
		"#--------------------- Memory (change with caution!) -------------------------\n\n"
		"MEMORY_OBJSTACK\t3000\t\t# number of objects in stack\n"
		"MEMORY_PIXSTACK\t300000\t\t# number of pixels in stack\n"
		"MEMORY_BUFSIZE\t1024\t\t# number of lines in buffer\n\n"
		"#------------------------------ Check Image ----------------------------------\n\n"
		"CHECKIMAGE_TYPE\tNONE\t# \"NONE\", \"BACKGROUND\"\n"
		"                                # \"MINIBACKGROUND\", \"-BACKGROUND\", \"OBJECTS\",\n"
		"                                # \"-OBJECTS\", \"SEGMENTATION\", \"APERTURES\",\n"
		"                                # or \"FILTERED\" (*)\n"
		"CHECKIMAGE_NAME\tnircam_segm.fits\t# CHECK IMAGE\n\n"
		"#------------------------------- Flag -----------------------------------\n\n"
		"FLAG_TYPE\tOR\t\t# COMBINATION OF INTERNAL + EXTERNAL FLAGS\n"
		f"FLAG_IMAGE\t{flagimag_value}\t# FLAG IMAGE (INTEGER) *** THIS WILL COME FROM COMMAND LINE\n\n"
		"#------------------------------- Weight -----------------------------------\n\n"
		"WEIGHT_TYPE\tMAP_WEIGHT\n"
		"RESCALE_WEIGHTS Y\n"
		f"WEIGHT_IMAGE    {weightimage_value}\n"
		"WEIGHT_GAIN     N\n"
	)

	config_path.write_text(config_text, encoding="utf-8")
	return config_path, param_path

