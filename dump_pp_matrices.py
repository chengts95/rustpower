import pandapower as pp
import pandapower.networks as pn
import numpy as np
import scipy.sparse as sp

try:
    from pandapower.pypower.ppoption import ppoption
    from pandapower.pypower.opf_setup import opf_setup
    from pandapower.pypower.opf_hessfcn import opf_hessfcn
except ImportError:
    import pypower.api as pa
    from pypower.ppoption import ppoption
    from pypower.opf_setup import opf_setup
    from pypower.opf_hessfcn import opf_hessfcn

net = pn.case118()
# Pandapower runopp to ensure internal ppc is ready
try:
    pp.runopp(net)
except:
    pass

ppc = net._ppc
if ppc is None:
    # Manual conversion
    from pandapower.pd2ppc import _pd2ppc
    ppc = _pd2ppc(net)

# opf_setup to get x0 and mappings
ppc_int, model, _ = opf_setup(ppc, ppoption())

x0 = model['x0']
nb = ppc_int['bus'].shape[0]
nl = ppc_int['branch'].shape[0]
# Use same point for comparison
lam = np.ones(nb * 2)
mu = np.ones(nl * 2)

H = opf_hessfcn(x0, lam, mu, 1.0, ppc_int)

sp.save_npz('pp_hessian_118.npz', H)
np.save('pp_x0_118.npy', x0)
print(f"Hessian shape: {H.shape}")
print(f"Hessian NNZ: {H.nnz}")
