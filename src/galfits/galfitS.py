import argparse
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import sys
import traceback
from contextlib import contextmanager, nullcontext
try:
        from . import gsutils
except ImportError:
        # Allow direct script execution (python path/to/galfitS.py)
        from galfits import gsutils
import pickle
from astropy.table import Table
import numpy as np
import jax.numpy as jnp
import time
from astropy.io import ascii


class _TeeStream:
        def __init__(self, *streams):
                self._streams = streams

        def write(self, data):
                for stream in self._streams:
                        try:
                                stream.write(data)
                        except Exception:
                                pass
                return len(data)

        def flush(self):
                for stream in self._streams:
                        try:
                                stream.flush()
                        except Exception:
                                pass

        def isatty(self):
                return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)


@contextmanager
def _tee_stdout_stderr(log_path: str):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        original_stdout, original_stderr = sys.stdout, sys.stderr
        log_file = open(log_path, "a", buffering=1)
        try:
                sys.stdout = _TeeStream(original_stdout, log_file)
                sys.stderr = _TeeStream(original_stderr, log_file)
                yield log_file
        finally:
                sys.stdout, sys.stderr = original_stdout, original_stderr
                try:
                        log_file.close()
                except Exception:
                        pass

def parse_args(argv=None):
        parser = argparse.ArgumentParser(description="Run galfitS")
        parser.add_argument("config", nargs='?', default="", help="config file of galfitS")
        parser.add_argument("--config", dest="config_option", type=str, default="", help="config file of galfitS")
        parser.add_argument("--workplace", "--work", dest="workplace", type=str, default="./", help="filepath to store galfitS outputs")
        parser.add_argument("--notfit", action='store_true', help="not fit, only generate parameter file")
        parser.add_argument("--fit_method", type=str, default="optimizer", help="chisq, flowmc, dynesty, jnesty, optimizer, ES")
        parser.add_argument("--multistart", action='store_true', help="use multistarting points for optimizer fitting")
        parser.add_argument("--NsGPU", type=int, default=10, help="number of starting points per GPU")
        parser.add_argument("--nstart", type=int, default=10, help="num of starting points for optimizer fitting")
        parser.add_argument("--num_steps", type=int, default=20000, help="num_steps for optimizer fitting")
        parser.add_argument("--learning_rate", type=float, default=0.0008, help="learning_rate for optimizer fitting")
        parser.add_argument("--baysian", action='store_true', help="using Baysian priors for optimizer fitting")
        parser.add_argument("--cal_sigma", action='store_true', help="calculate hessian_sigma for optimizer fitting")
        parser.add_argument("--chimethod", type=str, default="scipy", help="method for chisq fitting")
        parser.add_argument("--scimethod", type=str, default="SLSQP", help="method for scipy fitting")
        parser.add_argument("--constrain", action='store_true', help="constrain for chisq fitting")
        parser.add_argument("--nchains", type=int, default=30, help="nchains for flowmc")
        parser.add_argument("--rstep", type=float, default=0.1, help="parameter step for flowmc")
        parser.add_argument("--nlocalsteps", type=int, default=3000, help="nlocalsteps for flowmc")
        parser.add_argument("--nlooptraining", type=int, default=20, help="nlooptraining for flowmc")
        parser.add_argument("--nloopproduction", type=int, default=10, help="nloopproduction for flowmc")
        parser.add_argument("--sampler", type=str, default="GRW", help="sampler for flowmc")
        parser.add_argument("--nlive", type=int, default=80, help="numbe of live point for dynesty")
        parser.add_argument("--nvectorize", action='store_true', help="not vectorize the model calculation")
        parser.add_argument("--maxiters", type=int, default=5000000, help="maxiters for dynesty")
        parser.add_argument("--dlogz", type=float, default=0.02, help="dlogz for dynesty")
        parser.add_argument("--bound", type=str, default="multi", help="bounding method for jnesty: none, single, multi")
        parser.add_argument("--rwalk_K", type=int, default=None, help="random walk steps per iteration for jnesty (auto-tuned if not set)")
        parser.add_argument("--target_acceptance", type=float, default=0.5, help="target acceptance rate for jnesty scale adaptation")
        parser.add_argument("--device", type=str, default="gpu", help="device for jnesty: gpu or cpu")
        parser.add_argument("--queue_size", type=int, default=None, help="queue size for parallel likelihood evals in jnesty (auto for bound=multi)")
        parser.add_argument("--sample_method", type=str, default="rwalk", help="sample_method for dynesty")
        parser.add_argument("--dynamic", action='store_true', help="use dynamic nested sampling (dynesty or jnesty)")
        parser.add_argument("--maxbatch", type=int, default=10, help="maxbatch for dynesty or jnesty dynamic sampling")
        parser.add_argument("--n_effective", type=int, default=None, help="target number of effective posterior samples for dynamic nested sampling early stopping (dynesty or jnesty)")
        parser.add_argument("--use_stop", action='store_true', help="enable early stopping in dynamic nested sampling (dynesty or jnesty)")
        parser.add_argument("--parconstrain", type=str, default="default", help="filepath to store parameter constrain")
        parser.add_argument("--readpar", type=str, default="default", help="filepath to read parameter")
        parser.add_argument("--readsummary", type=str, default="default", help="filepath to read summary file")
        parser.add_argument("--savefull_results", action='store_true', help="save full results or not")
        parser.add_argument("--weight_spec", type=float, default=1.0, help="weight of spectrum in fitting")
        parser.add_argument("--num_generations", type=int, default=20000, help="num_generations for ES")
        parser.add_argument("--popsize", type=int, default=10, help="popsize for ES")
        parser.add_argument("--priorpath", type=str, default=None, help="path to the astrophysical prior file")
        parser.add_argument("--fixagnlinepro", action='store_true', help="fix agn line profile or not")
        parser.add_argument("--ndisagn", action='store_true', help="display agn or not in the image")
        parser.add_argument("--saveimgs", action='store_true', help="save images or not")
        parser.add_argument("--savelog", action='store_true', help="save fitting log or not")
        parser.add_argument("--sed_multiplot", action="store_true", help="plot sed figure for every source in isolated region.")
        args = parser.parse_args(argv)
        return args


def main(argv=None):
        args = parse_args(argv)
        config_file_path = args.config_option if args.config_option else args.config
        if config_file_path == "":
                raise ValueError("config file path not found! Please provide it as positional argument or with --config")
        workplace = args.workplace
        if not os.path.exists(config_file_path):
                raise ValueError("config file path not found!")
        if not os.path.exists(workplace):
                os.mkdir(workplace)
                print ("workplace path not found! create it!")

        Myfitter,targ,fs = gsutils.read_config_file(config_file_path,workplace,priorpath=args.priorpath,fixagnlinepro = args.fixagnlinepro,vectorize = not args.nvectorize, silent=True)

        log_path = os.path.join(workplace, f"{targ}.galfitS.log")
        log_ctx = _tee_stdout_stderr(log_path) if args.savelog else nullcontext()

        with log_ctx:
                if args.savelog:
                        print(f"# galfitS log file: {log_path}")
                        print(f"# config: {config_file_path}")
                        print(f"# workplace: {workplace}")
                        print(f"# fit_method: {args.fit_method}")

                try:
                        if args.parconstrain != "default":
                                f = open(args.parconstrain,'r')
                                fs = f.read()
                        _cons_ns = {'jnp': jnp}
                        exec(fs, _cons_ns)
                        Update_Constraints = _cons_ns.get('Update_Constraints', None)

                        if Update_Constraints is None:
                                raise ValueError("Constraint function 'Update_Constraints' was not defined by constraint script")

                        if args.readpar != "default":
                                ptab = Table.read(args.readpar, format='ascii')
                                pkeys = Myfitter.pardict.keys()
                                for loop in range(len(pkeys)):
                                        el = ptab['name'][loop]
                                        if ptab['vary'][loop] == 'False':
                                                var = False
                                        else:
                                                var = True
                                        if ptab['expr'][loop] == 'None':
                                                Myfitter.lmParameters[el].set(value=ptab['value'][loop],vary=var,min=ptab['min'][loop],max=ptab['max'][loop],brute_step=ptab['step'][loop])
                                        else:
                                                Myfitter.lmParameters[el].set(value=ptab['value'][loop],vary=var,expr=ptab['expr'][loop],min=ptab['min'][loop],max=ptab['max'][loop],brute_step=ptab['step'][loop])
                                Myfitter.loose_fix_pars()
                                f = open(workplace + '/{0}.params'.format(targ), 'w')
                                print ('name value min max step vary expr', file=f)
                                for el in Myfitter.pardict:
                                        print('{0} {1} {2} {3} {4} {5} {6}'.format(el,Myfitter.pardict[el],Myfitter.lmParameters[el].min,Myfitter.lmParameters[el].max,Myfitter.lmParameters[el].brute_step,Myfitter.lmParameters[el].vary,Myfitter.lmParameters[el].expr), file=f)
                                f.close()
                                # update constraints
                                fs = gsutils.generate_cons_func(Myfitter.lmParameters)
                                _cons_ns = {'jnp': jnp}
                                exec(fs, _cons_ns)
                                Update_Constraints = _cons_ns.get('Update_Constraints', Update_Constraints)
                                f = open(workplace + '/{0}.constrain'.format(targ), 'w')
                                print(fs, file=f)
                                f.close()

                        if args.readsummary != "default":
                                sumfile = ascii.read(args.readsummary)
                                for loop in range(len(sumfile)):
                                        try:
                                                if Myfitter.lmParameters[sumfile['pname'][loop]].brute_step is None:
                                                        Myfitter.lmParameters[sumfile['pname'][loop]].brute_step = 0.1
                                                newmax = np.min(np.array([Myfitter.lmParameters[sumfile['pname'][loop]].max,sumfile['best_value'][loop]+10*Myfitter.lmParameters[sumfile['pname'][loop]].brute_step]))
                                                newmin = np.max(np.array([Myfitter.lmParameters[sumfile['pname'][loop]].min,sumfile['best_value'][loop]-10*Myfitter.lmParameters[sumfile['pname'][loop]].brute_step]))
                                                Myfitter.lmParameters[sumfile['pname'][loop]].set(value = sumfile['best_value'][loop],min = newmin,max = newmax)
                                        except KeyError:
                                                print ("{0} not in parameter list, skip it!".format(sumfile['pname'][loop]))
                                Myfitter.loose_fix_pars()
                                f = open(workplace + '/{0}.params'.format(targ), 'w')
                                print ('name value min max brute_step vary expr', file=f)
                                for el in Myfitter.pardict:
                                        print('{0} {1} {2} {3} {4} {5} {6}'.format(el,Myfitter.pardict[el],Myfitter.lmParameters[el].min,Myfitter.lmParameters[el].max,Myfitter.lmParameters[el].brute_step,Myfitter.lmParameters[el].vary,Myfitter.lmParameters[el].expr), file=f)
                                f.close()
                                # update constraints
                                #fs = gsutils.generate_cons_func(Myfitter.lmParameters)
                                #exec(fs)
                                #f = open(workplace + '/{0}.constrain'.format(targ), 'w')
                                #print(fs, file=f)
                                #f.close()

                        if not args.notfit:
                                # timeing
                                start = time.time()
                                Myfitter.Set_Update_Constraints(Update_Constraints)
                                Myfitter.set_wight_spec(args.weight_spec)
                                if args.fit_method == "chisq":
                                        fr = Myfitter.minimize(method=args.chimethod,scipymethod=args.scimethod,constrain = args.constrain)
                                elif args.fit_method == "flowmc":
                                        fr = Myfitter.flowmc(n_chains = args.nchains, r_step = args.rstep, n_local_steps = args.nlocalsteps, n_loop_training = args.nlooptraining, n_loop_production = args.nloopproduction, sampler = args.sampler)
                                elif args.fit_method == "dynesty":
                                        fr = Myfitter.nested_sampling(maxiters=args.maxiters, nlive=args.nlive,dlogz = args.dlogz,sample_method=args.sample_method,dynamical=args.dynamic,dynesty_kwargs={'maxbatch': args.maxbatch , 'use_stop': args.use_stop, 'n_effective': args.n_effective})
                                elif args.fit_method == "optimizer":
                                        if args.multistart:
                                                fr = Myfitter.optimizer_multi(num_steps=args.num_steps, learning_rate=args.learning_rate, calsigma=args.cal_sigma,bayesian=args.baysian,num_starts=args.nstart, NsGPU = args.NsGPU)
                                        else:
                                                fr = Myfitter.optimizer(num_steps=args.num_steps, learning_rate=args.learning_rate, calsigma=args.cal_sigma,bayesian=args.baysian)
                                elif args.fit_method == "ES":
                                        if args.multistart:
                                                fr = Myfitter.evolution_strategies_multi(num_generations=args.num_generations, popsize=args.popsize, calsigma=args.cal_sigma,bayesian=args.baysian,num_starts=args.nstart, NsGPU = args.NsGPU)
                                        else:
                                                fr = Myfitter.evolution_strategies_theta(num_generations=args.num_generations, popsize=args.popsize, calsigma=args.cal_sigma,bayesian=args.baysian)
                                elif args.fit_method == "jnesty":
                                        fr = Myfitter.jnesty_nested_sampling(maxiters=args.maxiters, nlive=args.nlive, dlogz=args.dlogz, bound=args.bound, rwalk_K=args.rwalk_K, target_acceptance=args.target_acceptance, device=args.device, queue_size=args.queue_size, verbose=True, workplace=workplace, dynamical=args.dynamic, maxbatch=args.maxbatch, n_effective=args.n_effective, use_stop=args.use_stop,
                                        )
                                else:
                                        raise ValueError("fit_method not supported!")
                                stop = time.time()


                                if args.savefull_results:
                                        fr_copy = {k: v for k, v in fr.items() if k != 'sampler'}
                                        result = {
                                        'resultDit': fr_copy,
                                        'pardict': Myfitter.pardict,
                                        'varnames': Myfitter.varnames,
                                        }
                                        pickle.dump(result,open(workplace + '/{0}.gsresu'.format(targ),"wb"))

                                # write summary file
                                f = open(workplace + '/{0}.gssummary'.format(targ),'w')
                                f.write('# target: {0}\n'.format(targ))
                                f.write('# config file: {0}\n'.format(config_file_path))
                                f.write('# fitting mode: {0}\n'.format(Myfitter.GSdata.fitMode))
                                if Myfitter.GSdata.fitMode == "single image - photometry":
                                        f.write('### Magnitude can be found in fixed parameters.\n')
                                        f.write('### Magnitude error can be calculated by 2.5 times the logM error.\n')
                                f.write('# fiting method: {0}\n'.format(args.fit_method))
                                f.write('# chisq: {0}\n'.format(Myfitter.chisq))
                                f.write('# reduced chisq: {0}\n'.format(Myfitter.chisq/Myfitter.dof))
                                f.write('# BIC: {0}\n'.format(Myfitter.BIC))
                                f.write('# fitting time: {0} mins\n'.format((stop-start)/60.))
                                f.write('############# data summary ############## \n')
                                pltwave, Sedcomp, Sedlabel, z = Myfitter.cal_model_image()
                                for image_atlas in Myfitter.GSdata.image_atlas_list:
                                        f.write('# image atals: {0}\n'.format(image_atlas.name))
                                        for loop,band in enumerate(image_atlas.band_list):
                                                f.write('#    image number: {0}  band: {1}  chisq: {2}  dof: {3}  reduced chisq: {4}\n'.format(loop,band,image_atlas.img_residual[loop],image_atlas.img_dof[loop],image_atlas.img_residual[loop]/image_atlas.img_dof[loop]))
                                        for loop, spec in enumerate(image_atlas.spectra):
                                                f.write('#    spectrum number: {0}  chisq: {1}  dof: {2}  reduced chisq: {3}\n'.format(loop,image_atlas.spec_residual[loop],image_atlas.spec_dof[loop],image_atlas.spec_residual[loop]/image_atlas.spec_dof[loop]))        
                                f.write('############ fitting summary ############ \n')
                                if args.fit_method == "chisq":
                                        f.write('# method: {0}  scipymethod: {1} \n'.format(args.chimethod,args.scimethod))
                                elif args.fit_method == "ES":
                                        f.write('# num_generations: {0}  popsize: {1} \n'.format(args.num_generations,args.popsize))
                                        if args.multistart:
                                                f.write('# multistart: True  nstart: {0} NsGPU: {1}\n'.format(args.nstart,args.NsGPU))
                                elif args.fit_method == "optimizer":
                                        f.write('# num_steps: {0}  learning_rate: {1} calculate_hessian: {2} \n'.format(args.num_steps,args.learning_rate,args.cal_sigma))
                                        if args.multistart:
                                                f.write('# multistart: True  nstart: {0} NsGPU: {1}\n'.format(args.nstart,args.NsGPU))
                                elif args.fit_method == "flowmc":
                                        f.write('# nchains: {0} parastep: {1} nlocalsteps: {2} nlooptraining: {3} nloopproduction: {4} sampler: {5}\n'.format(args.nchains,args.rstep,args.nlocalsteps,args.nlooptraining,args.nloopproduction,args.sampler))
                                elif args.fit_method == "dynesty":
                                        f.write('# maxiters: {0} nlive: {1} dlogz: {2} sample_method: {3} dynamic: {4} maxbatch: {5}\n'.format(args.maxiters,args.nlive,args.dlogz,args.sample_method,args.dynamic,args.maxbatch))
                                elif args.fit_method == "jnesty":
                                        f.write('# maxiters: {0} nlive: {1} dlogz: {2} bound: {3} rwalk_K: {4} target_acceptance: {5} device: {6} queue_size: {7}\n'.format(
                                            args.maxiters, args.nlive, args.dlogz, args.bound, args.rwalk_K, args.target_acceptance, args.device, args.queue_size))
                                        if args.dynamic:
                                                f.write('# dynamic: True  maxbatch: {0}\n'.format(args.maxbatch))

                                haveerror = (isinstance(fr, dict) and (fr.get('param_errors', None) is not None))
                                f.write('# free parameters:\n')
                                if haveerror:
                                        f.write('pname    best_value    param_errors\n')
                                        ncol = 3
                                else:
                                        f.write('pname    best_value\n')
                                        ncol = 2
                                theta = fr.get('theta', None) if isinstance(fr, dict) else None
                                perr = fr.get('param_errors', None) if isinstance(fr, dict) else None
                                for loop, var in enumerate(Myfitter.varnames):
                                        if theta is None:
                                                val = float(Myfitter.theta0[loop])
                                        else:
                                                val = float(theta[loop])
                                        if haveerror:
                                                err = float(perr[loop])
                                                f.write('{0}    {1:.4f}    {2:.4f}\n'.format(var, val, err))
                                        else:
                                                f.write('{0}    {1:.4f}\n'.format(var, val))
                                f.write('# fixed parameters:\n')
                                f.write('# pname    fixed_value\n')
                                for el in Myfitter.pardict.keys():
                                        if not Myfitter.lmParameters[el].vary:
                                                if ncol == 2:  
                                                        f.write('{0}    {1:.4f}\n'.format(el,np.array(Myfitter.pardict[el])))
                                                elif ncol == 3:
                                                        f.write('{0}    {1:.4f}    nan\n'.format(el,np.array(Myfitter.pardict[el])))
                                                elif ncol == 4:
                                                        f.write('{0}    {1:.4f}    nan    nan\n'.format(el,np.array(Myfitter.pardict[el])))
                                f.write('######################################### \n')
                                f.close()

                        # Generate JNesty diagnostic plots
                        if args.fit_method == "jnesty" and isinstance(fr, dict) and 'sampler' in fr:
                                try:
                                        print("\nGenerating JNesty diagnostic plots...")
                                        plot_files = Myfitter.plot_jnesty_diagnostics(fr, save_dir=workplace)
                                        print(f"{len(plot_files)} diagnostic plots saved to {workplace}")
                                except Exception as e:
                                        print(f"Warning: JNesty plot generation failed: {e}")

                        gsutils.standard_display(Myfitter, workplace, targ, ndisage = args.ndisagn,saveimgs = args.saveimgs, sed_multiplot = args.sed_multiplot)
                except Exception:
                        traceback.print_exc()
                        raise


if __name__ == "__main__":
        main()
               
