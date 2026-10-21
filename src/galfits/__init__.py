import os
import warnings


def _import_all(module_name):
	module = __import__(f"{__name__}.{module_name}", fromlist=["*"])
	names = getattr(module, "__all__", None)
	if names is None:
		names = [name for name in module.__dict__ if not name.startswith("_")]
	for name in names:
		globals()[name] = getattr(module, name)


for _module_name in ("sed_interp", "images", "galaxy"):
	try:
		_import_all(_module_name)
	except Exception as exc:
		if os.environ.get("SPHINX_BUILD", "0") == "1":
			warnings.warn(
				f"Failed to import galfits.{_module_name} during docs build: {exc}",
				RuntimeWarning,
			)
		else:
			raise
